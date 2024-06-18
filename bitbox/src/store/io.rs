use super::{Page, Store, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use slab::Slab;
use std::{os::fd::AsRawFd, sync::Arc};

const RING_CAPACITY: u32 = 64;

// max number of inflight requests is bounded by the slab.
const MAX_IN_FLIGHT: usize = 64;

pub type HandleIndex = usize;

#[derive(Clone)]
pub enum IoKind {
    Read(PageIndex, Box<Page>),
    Write(PageIndex, Box<Page>),
    Fsync,
}

impl IoKind {
    pub fn unwrap_buf(self) -> Box<Page> {
        match self {
            IoKind::Read(_, buf) | IoKind::Write(_, buf) => buf,
            IoKind::Fsync => panic!("attempted to extract buf from fsync"),
        }
    }
}

#[derive(Clone, Copy)]
pub enum PageIndex {
    Data(u64),
    MetaBytes(u64),
    Meta,
}

impl PageIndex {
    fn index_in_store(self, store: &Store) -> u64 {
        match self {
            PageIndex::Data(i) => store.data_page_offset() + i,
            PageIndex::MetaBytes(i) => 1 + i,
            PageIndex::Meta => 0,
        }
    }
}

pub struct IoCommand {
    pub kind: IoKind,
    pub handle: HandleIndex,
    // note: this isn't passed to io_uring, it's higher-level userdata.
    pub user_data: u64,
}

pub struct CompleteIo {
    pub command: IoCommand,
    pub result: std::io::Result<()>,
}

/// Create an I/O worker managing an io_uring and sending responses back via channels to a number
/// of handles.
pub fn start_io_worker(
    store: Arc<Store>,
    num_handles: usize,
) -> (Sender<IoCommand>, Vec<Receiver<CompleteIo>>) {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::bounded(1);
    let (handle_txs, handle_rxs) = (0..num_handles)
        .map(|_| crossbeam_channel::bounded(32))
        .unzip();
    std::thread::spawn(move || run_worker(store, command_rx, handle_txs));
    (command_tx, handle_rxs)
}

fn run_worker(
    store: Arc<Store>,
    command_rx: Receiver<IoCommand>,
    handle_tx: Vec<Sender<CompleteIo>>,
) {
    let mut pending: Slab<IoCommand> = Slab::with_capacity(MAX_IN_FLIGHT);

    let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
        .setup_single_issuer()
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    loop {
        // note: dropping the queues at the end of loop iteration performs `sync` implicitly
        let (submitter, mut submit_queue, mut complete_queue) = ring.split();

        // 1. process completions.
        if !pending.is_empty() {
            // block on next completion if at capacity.
            if pending.len() == MAX_IN_FLIGHT && complete_queue.is_empty() {
                // hack: this is not exposed from the io_uring or libc libraries.
                const IORING_ENTER_GETEVENTS: u32 = 1;

                // we just get events here, not submit.
                // TODO: handle error
                unsafe {
                    submitter
                        .enter::<libc::sigset_t>(
                            0, // submit
                            1, // complete
                            IORING_ENTER_GETEVENTS,
                            None,
                        )
                        .unwrap();
                }
                complete_queue.sync();
            }

            for completion_event in complete_queue {
                let command = pending.remove(completion_event.user_data() as usize);
                let handle_idx = command.handle;
                let result = if completion_event.result() == -1 {
                    Err(std::io::Error::from_raw_os_error(completion_event.result()))
                } else {
                    Ok(())
                };
                let complete = CompleteIo { command, result };
                if let Err(_) = handle_tx[handle_idx].send(complete) {
                    // TODO: handle?
                    break;
                }
            }
        }

        // 2. accept new I/O requests when slab has space & submission queue is not full.
        let mut to_submit = false;
        while pending.len() < MAX_IN_FLIGHT && !submit_queue.is_full() {
            let next_io = if pending.is_empty() {
                // block on new I/O if nothing in-flight.
                match command_rx.recv() {
                    Ok(command) => command,
                    Err(_) => break, // disconnected
                }
            } else {
                match command_rx.try_recv() {
                    Ok(command) => command,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break, // TODO: wait on pending I/O?
                }
            };
            to_submit = true;

            let pending_index = pending.insert(next_io);

            let entry = submission_entry(
                pending.get_mut(pending_index).unwrap(),
                &*store,
                pending_index,
            );
            unsafe { submit_queue.push(&entry).unwrap() };
        }

        // 3. submit all together.
        if to_submit {
            submit_queue.sync();

            // TODO: handle this error properly.
            submitter.submit().unwrap();
        }
    }
}

fn submission_entry(command: &mut IoCommand, store: &Store, index: usize) -> squeue::Entry {
    match command.kind {
        IoKind::Read(page_index, ref mut buf) => opcode::Read::new(
            types::Fd(store.store_file.as_raw_fd()),
            buf.as_mut_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index.index_in_store(store) * PAGE_SIZE as u64)
        .build()
        .user_data(index as u64),
        IoKind::Write(page_index, ref buf) => opcode::Write::new(
            types::Fd(store.store_file.as_raw_fd()),
            buf.as_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index.index_in_store(store) * PAGE_SIZE as u64)
        .build()
        .user_data(index as u64),
        IoKind::Fsync => opcode::Fsync::new(types::Fd(store.store_file.as_raw_fd()))
            .build()
            .user_data(index as u64),
    }
}
