use super::{CompleteIo, IoCommand, IoKind, IoKindResult, IoPacket, PagePool, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, register, squeue, types, IoUring};
use parking_lot::{Condvar, Mutex};
use slab::Slab;
use std::{
    collections::{BTreeMap, VecDeque},
    os::fd::RawFd,
    sync::Arc,
};
use threadpool::ThreadPool;

const RING_CAPACITY: &'static str = "RING_CAPACITY";
const MAX_IN_FLIGHT: &'static str = "MAX_IN_FLIGHT";
const IOPOLL: &'static str = "IOPOLL";
const SQPOLL: &'static str = "SQPOLL";
const SQPOLL_IDLE: &'static str = "SQPOLL_IDLE";
const SINGLE_ISSUER: &'static str = "SINGLE_ISSUER";
const COOP_TASKRUN: &'static str = "COOP_TASKRUN";
const DEFER_TASKRUN: &'static str = "DEFER_TASKRUN";
const REGISTER_FILES: &'static str = "REGISTER_FILES";

fn is_true(env_var: &'static str) -> bool {
    std::env::var(env_var)
        .ok()
        .map(|no_shrinking| no_shrinking.to_lowercase() == "true")
        .unwrap_or(false)
}

fn get_usize(env_var: &'static str) -> usize {
    std::env::var(env_var)
        .ok()
        .map(|var| var.parse::<usize>().unwrap())
        .unwrap_or(128)
}

struct PendingIo {
    command: IoCommand,
    completion_sender: Sender<CompleteIo>,
}

pub fn start_io_worker(
    page_pool: PagePool,
    io_workers_tp: &ThreadPool,
    io_workers: usize,
) -> (Sender<IoPacket>, Option<RegisterFiles>) {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::unbounded();

    let register_files = RegisterFiles::new();

    start_workers(
        page_pool,
        io_workers_tp,
        command_rx,
        io_workers,
        register_files.clone(),
    );

    (command_tx, Some(register_files))
}

fn start_workers(
    page_pool: PagePool,
    io_workers_tp: &ThreadPool,
    command_rx: Receiver<IoPacket>,
    io_workers: usize,
    rf: RegisterFiles,
) {
    for _ in 0..io_workers {
        io_workers_tp.execute({
            let page_pool = page_pool.clone();
            let command_rx = command_rx.clone();
            let rf = rf.clone();
            move || run_worker(page_pool, command_rx, rf)
        });
    }
}

fn run_worker(page_pool: PagePool, command_rx: Receiver<IoPacket>, register_files: RegisterFiles) {
    let max_in_flight = get_usize(MAX_IN_FLIGHT);
    let mut pending: Slab<PendingIo> = Slab::with_capacity(max_in_flight);

    let ring_capacity = get_usize(RING_CAPACITY);
    let mut ring_builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();

    if is_true(IOPOLL) {
        ring_builder.setup_iopoll();
    }
    if is_true(SQPOLL) {
        ring_builder.setup_sqpoll(get_usize(SQPOLL_IDLE) as u32);
    }
    if is_true(SINGLE_ISSUER) {
        ring_builder.setup_single_issuer();
    }
    if is_true(COOP_TASKRUN) {
        ring_builder.setup_coop_taskrun();
    }
    if is_true(DEFER_TASKRUN) {
        ring_builder.setup_defer_taskrun();
    }
    let is_register_files = is_true(REGISTER_FILES);
    let is_defer_taskrun = is_true(DEFER_TASKRUN);

    let mut ring = ring_builder
        .build(ring_capacity as u32)
        .expect("Error building io_uring");

    let (submitter, mut submit_queue, mut complete_queue) = ring.split();
    let mut retries = VecDeque::<IoPacket>::new();

    // Indicates whether the worker detected that it should shutdown.
    let mut shutdown = false;

    let maybe_registered_map = if is_register_files {
        let files = register_files.wait_registration();
        let map: BTreeMap<i32, u32> = files
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, fd)| (fd, i as u32))
            .collect();
        submitter.register_files(&files).unwrap();
        Some(map)
    } else {
        None
    };

    loop {
        // 1. process completions.
        if !pending.is_empty() {
            complete_queue.sync();
            if is_defer_taskrun {
                unsafe {
                    let _ = submitter.enter::<()>(0, 0, 1 /* IORING_ENTER_GETEVENTS */, None);
                }
            }
            while let Some(completion_event) = complete_queue.next() {
                if pending.get(completion_event.user_data() as usize).is_none() {
                    continue;
                }
                let PendingIo {
                    command,
                    completion_sender,
                } = pending.remove(completion_event.user_data() as usize);

                // io_uring never uses errno to pass back error information.
                // Instead, completion_event.result() will contain what the equivalent
                // system call would have returned in case of success,
                // and in case of error completion_event.result() will contain -errno
                let io_uring_res = completion_event.result();
                let syscall_result = if io_uring_res >= 0 { io_uring_res } else { -1 };

                let result = match command.kind.get_result(syscall_result as isize) {
                    IoKindResult::Ok => Ok(()),
                    IoKindResult::Err => Err(std::io::Error::from_raw_os_error(io_uring_res.abs())),
                    IoKindResult::Retry => {
                        retries.push_back(IoPacket {
                            command,
                            completion_sender,
                        });
                        continue;
                    }
                };

                let complete = CompleteIo { command, result };
                let _ = completion_sender.send(complete);
            }
        } else if shutdown {
            // No pending IOs and we are shutting down. That means we can exit the worker.
            //
            // Why the `drop` here? Well, recall that the iou accepts commands parametrized with
            // buffers. These buffers are allocated in the page pool. If the page pool is dropped
            // before the ring is dropped, then that's a use-after-free.
            //
            // So in other words, we plumb `page_pool` all the way here and drop it here only to
            // ensure safety.
            drop(page_pool);
            return;
        }

        // 2. accept new I/O requests when slab has space & submission queue is not full.
        let mut to_submit = false;

        submit_queue.sync();
        while pending.len() < max_in_flight && !submit_queue.is_full() {
            let next_io = if !retries.is_empty() {
                // re-apply partially failed reads and writes
                // unwrap: known not empty
                retries.pop_front().unwrap()
            } else if pending.is_empty() {
                // block on new I/O if nothing in-flight.
                match command_rx.recv() {
                    Ok(command) => command,
                    Err(_) => {
                        shutdown = true;
                        break;
                    }
                }
            } else {
                match command_rx.try_recv() {
                    Ok(command) => command,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        shutdown = true;
                        break;
                    }
                }
            };

            to_submit = true;
            let pending_index = pending.insert(PendingIo {
                command: next_io.command,
                completion_sender: next_io.completion_sender,
            });

            let entry = if let Some(ref map) = maybe_registered_map {
                submission_entry_fixed(&mut pending.get_mut(pending_index).unwrap().command, map)
                    .user_data(pending_index as u64)
            } else {
                submission_entry(&mut pending.get_mut(pending_index).unwrap().command)
                    .user_data(pending_index as u64)
            };

            // unwrap: known not full
            unsafe { submit_queue.push(&entry).unwrap() };
        }

        // 3. submit all together.
        if to_submit {
            submit_queue.sync();
        }

        let wait = if pending.len() == max_in_flight { 1 } else { 0 };

        submitter.submit_and_wait(wait).unwrap();
    }
}

fn submission_entry(command: &mut IoCommand) -> squeue::Entry {
    match command.kind {
        IoKind::Read(fd, page_index, ref mut page) => {
            opcode::Read::new(types::Fd(fd), page.as_mut_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
        IoKind::Write(fd, page_index, ref page) => {
            opcode::Write::new(types::Fd(fd), page.as_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
        IoKind::WriteArc(fd, page_index, ref page) => {
            let page: &[u8] = &*page;
            opcode::Write::new(types::Fd(fd), page.as_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
        IoKind::WriteRaw(fd, page_index, ref page) => {
            opcode::Write::new(types::Fd(fd), page.as_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
    }
}

fn submission_entry_fixed(command: &mut IoCommand, map: &BTreeMap<i32, u32>) -> squeue::Entry {
    match command.kind {
        IoKind::Read(fd, page_index, ref mut page) => opcode::Read::new(
            types::Fixed(*map.get(&fd).unwrap()),
            page.as_mut_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index * PAGE_SIZE as u64)
        .build(),
        IoKind::Write(fd, page_index, ref page) => opcode::Write::new(
            types::Fixed(*map.get(&fd).unwrap()),
            page.as_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index * PAGE_SIZE as u64)
        .build(),
        IoKind::WriteArc(fd, page_index, ref page) => {
            let page: &[u8] = &*page;
            opcode::Write::new(
                types::Fixed(*map.get(&fd).unwrap()),
                page.as_ptr(),
                PAGE_SIZE as u32,
            )
            .offset(page_index * PAGE_SIZE as u64)
            .build()
        }
        IoKind::WriteRaw(fd, page_index, ref page) => opcode::Write::new(
            types::Fixed(*map.get(&fd).unwrap()),
            page.as_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index * PAGE_SIZE as u64)
        .build(),
    }
}

#[derive(Clone)]
pub struct RegisterFiles {
    files: Arc<Mutex<Option<Vec<RawFd>>>>,
    cv: Arc<Condvar>,
}

impl RegisterFiles {
    pub fn new() -> Self {
        Self {
            files: Arc::new(Mutex::new(None)),
            cv: Arc::new(Condvar::new()),
        }
    }

    pub fn regsiter(&self, files: &[RawFd]) {
        let _ = self.files.lock().replace(files.to_vec());
        // TODO: make sure to awake all waiting threads
        // I should pas shere the number of io uring threads
        self.cv.notify_all();
    }

    pub fn wait_registration(self) -> Vec<RawFd> {
        let mut files = self.files.lock();
        self.cv.wait_while(&mut files, |files| files.is_none());
        // UNWRAP: files has just been checked to be `Some`.
        files.take().unwrap()
    }
}
