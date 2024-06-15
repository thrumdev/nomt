use super::Store;
use crate::node_pages_map::{Page, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use slab::Slab;
use std::{os::fd::AsRawFd, path::PathBuf, sync::Arc};

const RING_CAPACITY: u32 = 128;
const SLAB_CAPACITY: usize = 256;

pub type WorkerIndex = usize;

#[derive(Clone, Copy)]
pub enum IoKind {
    Read,
    Write,
}

pub struct IoCommand {
    pub kind: IoKind,
    pub worker: WorkerIndex,
    pub page_id: u64,
    pub buf: Box<Page>,
}

pub struct CompleteIo {
    command: IoCommand,
    result: std::io::Result<()>,
}

pub fn start_io_worker(
    store: Arc<Store>,
    num_workers: usize,
) -> (Sender<IoCommand>, Vec<Receiver<CompleteIo>>) {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::bounded(1);
    let (worker_txs, worker_rxs) = (0..num_workers)
        .map(|_| crossbeam_channel::bounded(32))
        .unzip();
    std::thread::spawn(move || run_worker(store, command_rx, worker_txs));
    (command_tx, worker_rxs)
}

fn run_worker(
    store: Arc<Store>,
    command_rx: Receiver<IoCommand>,
    worker_tx: Vec<Sender<CompleteIo>>,
) {
    let mut pending: Slab<IoCommand> = Slab::with_capacity(SLAB_CAPACITY);

    let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
        .setup_single_issuer()
        .setup_sqpoll(100)
        // .setup_iopoll
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    loop {
        // note: dropping the queues each loop iteration performs `sync`.
        let (submitter, mut submit_queue, complete_queue) = ring.split();

        // 1. process completions.
        if !pending.is_empty() {
            for completion_event in complete_queue {
                let command = pending.remove(completion_event.user_data() as usize);
                let worker_idx = command.worker;
                let result = if completion_event.result() != 0 {
                    Err(std::io::Error::from_raw_os_error(completion_event.result()))
                } else {
                    Ok(())
                };
                let complete = CompleteIo { command, result };
                if let Err(_) = worker_tx[worker_idx].send(complete) {
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

        // 3. submit all together, waiting for at least one completion if full
        let pending_is_full = pending.len() == SLAB_CAPACITY;
        if to_submit || pending_is_full {
            submit_queue.sync();
            let wait_amount = if pending_is_full { 1 } else { 0 };

            // TODO: handle this error properly.
            submitter.submit_and_wait(wait_amount).unwrap();
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
