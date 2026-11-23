//! Incremental write submission with one completion notification per group.
use super::{CompletionTarget, IoCommand, IoHandle, IoKind, IoPacket};
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use std::{
    io,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct State {
    // Counts live producers and outstanding requests. A producer's reference
    // prevents completion while it registers further requests.
    remaining: AtomicUsize,
    error: Mutex<Option<io::Error>>,
    finished: Sender<io::Result<()>>,
}

impl State {
    fn add(&self) {
        // Like Arc, abort before the counter could wrap around.
        if self.remaining.fetch_add(1, Ordering::Relaxed) > isize::MAX as usize {
            std::process::abort();
        }
    }

    fn finish(&self, result: io::Result<()>) {
        if let Err(error) = result {
            self.error.lock().get_or_insert(error);
        }
        // This counter is accounting only. Errors are published through the
        // mutex and the result through the channel, not through this atomic.
        if self.remaining.fetch_sub(1, Ordering::Relaxed) == 1 {
            let result = self.error.lock().take().map_or(Ok(()), Err);
            let _ = self.finished.send(result);
        }
    }
}

/// A producer of writes. Drop all producers to finish submission.
/// Writes are enqueued immediately, not buffered until the group is closed.
pub struct WriteGroup {
    io: IoHandle,
    state: Arc<State>,
}

/// Waits for all producers and all their writes, including failed writes.
/// Declare this guard after external buffer owners, so it drains before they drop.
/// All producers must be dropped before this guard is waited on or dropped.
pub struct WriteGroupCompletion(Option<Receiver<io::Result<()>>>);

impl WriteGroupCompletion {
    /// Close all producers before waiting. Errors are returned only after all
    /// requests have completed. Drop also drains, for early returns/unwinding.
    pub fn wait(&mut self) -> io::Result<()> {
        match self.0.take() {
            Some(receiver) => receiver
                .recv()
                .unwrap_or_else(|_| Err(io::Error::other("write group disconnected"))),
            None => Ok(()),
        }
    }
}

impl Drop for WriteGroupCompletion {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}

impl IoHandle {
    pub fn write_group(&self) -> (WriteGroup, WriteGroupCompletion) {
        let (finished, receiver) = crossbeam_channel::bounded(1);
        let state = Arc::new(State {
            remaining: AtomicUsize::new(1),
            error: Mutex::new(None),
            finished,
        });
        (
            WriteGroup {
                io: self.clone(),
                state,
            },
            WriteGroupCompletion(Some(receiver)),
        )
    }
}

impl Clone for WriteGroup {
    fn clone(&self) -> Self {
        self.state.add();
        Self {
            io: self.io.clone(),
            state: self.state.clone(),
        }
    }
}

impl Drop for WriteGroup {
    fn drop(&mut self) {
        self.state.finish(Ok(()));
    }
}

impl WriteGroup {
    /// Submit immediately. Raw writes retain the existing IoHandle contract:
    /// their owners and all file descriptors must outlive this group's completion
    /// guard. No reads allowed.
    pub fn send(&self, command: IoCommand) -> io::Result<()> {
        assert!(
            !matches!(command.kind, IoKind::Read(..)),
            "reads need individual completions"
        );
        self.state.add();
        let ticket = WriteTicket {
            state: Some(self.state.clone()),
        };
        let packet = IoPacket {
            command,
            completion: CompletionTarget::WriteGroup(ticket),
        };
        let Some(sender) = self.io.sender.upgrade() else {
            drop(packet);
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "I/O pool shut down",
            ));
        };
        sender
            .send(packet)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "I/O pool shut down"))
    }
}

pub(super) struct WriteTicket {
    state: Option<Arc<State>>,
}

impl WriteTicket {
    pub(super) fn complete(mut self, result: io::Result<()>) {
        let state = self.state.take().unwrap();
        drop(self);
        state.finish(result);
    }
}

impl Drop for WriteTicket {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.finish(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "write request abandoned",
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{CompleteIo, FatPage, PagePool};
    use std::{sync::Weak, time::Duration};

    fn harness() -> (IoHandle, Arc<Sender<IoPacket>>, Receiver<IoPacket>) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let sender = Arc::new(sender);
        let (completion_sender, completion_receiver) = crossbeam_channel::unbounded();
        (
            IoHandle {
                sender: Arc::downgrade(&sender),
                completion_sender,
                completion_receiver,
            },
            sender,
            receiver,
        )
    }

    fn command(page: FatPage) -> IoCommand {
        IoCommand {
            kind: IoKind::Write(-1, 0, page),
            user_data: 0,
        }
    }

    fn complete(packet: IoPacket, result: io::Result<()>) {
        packet.completion.finish(CompleteIo {
            command: packet.command,
            result,
        });
    }

    #[test]
    fn empty_group() {
        let (handle, _sender, _receiver) = harness();
        let (writes, mut done) = handle.write_group();
        drop(writes);
        done.wait().unwrap();
    }

    #[test]
    fn can_submit_after_every_previous_write_completed() {
        let (handle, _sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        for _ in 0..4 {
            writes.send(command(pool.alloc_fat_page())).unwrap();
            complete(receiver.recv().unwrap(), Ok(()));
            assert!(done.0.as_ref().unwrap().is_empty());
        }
        drop(writes);
        done.wait().unwrap();
        assert!(handle.completion_receiver.is_empty());
    }

    #[test]
    fn concurrent_producers_close_before_final_completion() {
        let (handle, _sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let writes = writes.clone();
                let pool = &pool;
                scope.spawn(move || {
                    for _ in 0..32 {
                        writes.send(command(pool.alloc_fat_page())).unwrap();
                    }
                });
            }
        });
        drop(writes);
        for _ in 0..127 {
            complete(receiver.recv().unwrap(), Ok(()));
        }
        assert!(done.0.as_ref().unwrap().is_empty());
        complete(receiver.recv().unwrap(), Ok(()));
        done.wait().unwrap();
    }

    #[test]
    fn first_error_does_not_complete_group_early() {
        let (handle, _sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        for _ in 0..3 {
            writes.send(command(pool.alloc_fat_page())).unwrap();
        }
        drop(writes);
        complete(
            receiver.recv().unwrap(),
            Err(io::Error::from_raw_os_error(libc::EIO)),
        );
        complete(
            receiver.recv().unwrap(),
            Err(io::Error::from_raw_os_error(libc::ENOSPC)),
        );
        assert!(done.0.as_ref().unwrap().is_empty());
        complete(receiver.recv().unwrap(), Ok(()));
        assert_eq!(done.wait().unwrap_err().raw_os_error(), Some(libc::EIO));
    }

    #[test]
    fn shutdown_and_abandoned_requests_report_errors() {
        let (handle, sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        writes.send(command(pool.alloc_fat_page())).unwrap();
        drop(receiver.recv().unwrap());
        drop(sender);
        assert!(writes.send(command(pool.alloc_fat_page())).is_err());
        drop(writes);
        assert_eq!(done.wait().unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[test]
    fn rejected_enqueue_retires_ticket() {
        let (handle, _sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        drop(receiver);
        assert!(writes.send(command(pool.alloc_fat_page())).is_err());
        drop(writes);
        assert!(done.wait().is_err());
    }

    #[test]
    fn completion_drop_drains_before_releasing_external_owners() {
        let (handle, _sender, receiver) = harness();
        let pool = PagePool::new();
        let owner = Arc::new(pool.alloc_fat_page());
        let weak: Weak<FatPage> = Arc::downgrade(&owner);
        let (writes, done) = handle.write_group();
        writes
            .send(IoCommand {
                kind: IoKind::WriteRaw(-1, 0, owner.page()),
                user_data: 0,
            })
            .unwrap();
        drop(writes);
        let (finished_tx, finished_rx) = crossbeam_channel::bounded(1);
        let waiter = std::thread::spawn(move || {
            drop(done);
            drop(owner);
            finished_tx.send(()).unwrap();
        });
        assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
        assert!(weak.upgrade().is_some());
        complete(
            receiver.recv().unwrap(),
            Err(io::Error::from_raw_os_error(libc::EIO)),
        );
        finished_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn retry_keeps_the_same_ticket_outstanding() {
        let (handle, sender, receiver) = harness();
        let pool = PagePool::new();
        let (writes, mut done) = handle.write_group();
        writes.send(command(pool.alloc_fat_page())).unwrap();
        drop(writes);
        let packet = receiver.recv().unwrap();
        // The worker transfers the packet unchanged to its retry queue.
        sender.send(packet).unwrap();
        assert!(done.0.as_ref().unwrap().is_empty());
        complete(receiver.recv().unwrap(), Ok(()));
        done.wait().unwrap();
    }

    #[test]
    fn real_writes_drain_without_further_submissions() {
        use std::os::{fd::AsRawFd, unix::fs::FileExt};
        // Exercise both the idle tail and refilling a full ring. An ordinary read
        // on the same handle must still return its individual completion.
        for count in [1, 1100] {
            let mut io_pool = crate::io::start_test_io_pool(1, PagePool::new());
            let handle = io_pool.make_handle();
            let file = tempfile::tempfile().unwrap();
            file.set_len(count * super::super::PAGE_SIZE as u64)
                .unwrap();
            let (writes, mut done) = handle.write_group();
            for index in 0..count {
                let mut page = io_pool.page_pool().alloc_fat_page();
                page.fill(index as u8);
                writes
                    .send(IoCommand {
                        kind: IoKind::Write(file.as_raw_fd(), index, page),
                        user_data: 0,
                    })
                    .unwrap();
            }
            drop(writes);
            done.wait().unwrap();
            let mut byte = [0];
            file.read_exact_at(&mut byte, (count - 1) * super::super::PAGE_SIZE as u64)
                .unwrap();
            assert_eq!(byte[0], (count - 1) as u8);
            handle
                .send(IoCommand {
                    kind: IoKind::Read(
                        file.as_raw_fd(),
                        count - 1,
                        io_pool.page_pool().alloc_fat_page(),
                    ),
                    user_data: 42,
                })
                .unwrap();
            let complete = handle
                .receiver()
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            complete.result.unwrap();
            assert_eq!(complete.command.user_data, 42);
            assert_eq!(complete.command.kind.unwrap_buf()[0], (count - 1) as u8);
            io_pool.shutdown();
        }
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn synchronous_backend_group() {
        use std::os::{fd::AsRawFd, unix::fs::FileExt};
        let pool = PagePool::new();
        let workers = threadpool::ThreadPool::new(1);
        let sender = super::super::unix_tests::start_io_worker(pool.clone(), &workers, 1);
        let mut io_pool = super::super::IoPool {
            sender: Some(Arc::new(sender)),
            page_pool: pool,
            io_workers_tp: workers,
        };
        let handle = io_pool.make_handle();
        let file = tempfile::tempfile().unwrap();
        let (writes, mut done) = handle.write_group();
        for index in 0..65 {
            let mut page = io_pool.page_pool().alloc_fat_page();
            page.fill(index as u8);
            writes
                .send(IoCommand {
                    kind: IoKind::Write(file.as_raw_fd(), index, page),
                    user_data: 0,
                })
                .unwrap();
        }
        drop(writes);
        done.wait().unwrap();
        let mut byte = [0];
        file.read_exact_at(&mut byte, 64 * super::super::PAGE_SIZE as u64)
            .unwrap();
        assert_eq!(byte[0], 64);
        io_pool.shutdown();
    }
}
