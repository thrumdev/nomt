use super::{Page, Store, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use slab::Slab;
use std::{os::fd::AsRawFd, sync::Arc};

const RING_CAPACITY: u32 = 64;

// max number of inflight requests is bounded by the slab.
const SLAB_CAPACITY: usize = 64;

pub type HandleIndex = usize;

#[derive(Clone, Copy)]
pub enum IoKind {
    Read,
    Write,
}

pub struct IoCommand {
    pub kind: IoKind,
    pub handle: HandleIndex,
    pub page_id: u64,
    pub buf: Box<Page>,
}

pub struct CompleteIo {
    command: IoCommand,
    result: std::io::Result<()>,
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

// hack: this is not exposed from the io_uring or libc libraries.
const IORING_ENTER_GETEVENTS: u32 = 1;

fn run_worker(
    store: Arc<Store>,
    command_rx: Receiver<IoCommand>,
    handle_tx: Vec<Sender<CompleteIo>>,
) {
    let mut pending: Slab<IoCommand> = Slab::with_capacity(SLAB_CAPACITY);

    let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
        .setup_single_issuer()
        .setup_sqpoll(100)
        // .setup_iopoll
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    loop {
        // note: dropping the queues at the end of loop iteration performs `sync` implicitly
        let (submitter, mut submit_queue, mut complete_queue) = ring.split();

        // 1. process completions.
        if !pending.is_empty() {
            // block on next completion if at capacity.
            if pending.len() == SLAB_CAPACITY && complete_queue.is_empty() {
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
                let result = if completion_event.result() != 0 {
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

        // 2. accept new I/O requests when slab has space & submission queue is non-empty.
        let mut to_submit = false;
        submit_queue.sync();
        while pending.len() < SLAB_CAPACITY && !submit_queue.is_full() {
            let mut next_io = if pending.is_empty() {
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

            let page_id = next_io.page_id;
            let kind = next_io.kind;
            let buf_ptr = next_io.buf.as_mut_ptr();
            let pending_index = pending.insert(next_io);

            let entry = submission_entry(buf_ptr, page_id, kind, &*store, pending_index);
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

fn submission_entry(
    buf_ptr: *mut u8,
    page_id: u64,
    kind: IoKind,
    store: &Store,
    index: usize,
) -> squeue::Entry {
    match kind {
        IoKind::Read => opcode::Read::new(
            types::Fd(store.store_file.as_raw_fd()),
            buf_ptr,
            PAGE_SIZE as u32,
        )
        .offset(page_id * PAGE_SIZE as u64)
        .build()
        .user_data(index as u64),
        IoKind::Write => opcode::Write::new(
            types::Fd(store.store_file.as_raw_fd()),
            buf_ptr as *const u8,
            PAGE_SIZE as u32,
        )
        .offset(page_id * PAGE_SIZE as u64)
        .build()
        .user_data(index as u64),
    }
}
