use super::{CompleteIo, IoCommand, IoKind, IoKindResult, IoPacket, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use parking_lot::{Condvar, Mutex};
use slab::Slab;
use std::{collections::VecDeque, os::fd::RawFd, sync::Arc};

const RING_CAPACITY: u32 = 128;

pub const FIXED_HT: usize = 0;
pub const FIXED_BBN: usize = 1;
pub const FIXED_LN: usize = 2;

// max number of inflight requests is bounded by the slab.
const MAX_IN_FLIGHT: usize = RING_CAPACITY as usize;

struct PendingIo {
    command: IoCommand,
    completion_sender: Sender<CompleteIo>,
}

pub fn start_io_worker(
    io_workers: usize,
    defer_taskrun: bool,
) -> (Sender<IoPacket>, Option<RegisterFiles>) {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::unbounded();

    let register_files = RegisterFiles::new();
    start_workers(
        command_rx,
        io_workers,
        defer_taskrun,
        register_files.clone(),
    );

    (command_tx, Some(register_files))
}

fn start_workers(
    command_rx: Receiver<IoPacket>,
    io_workers: usize,
    defer_taskrun: bool,
    register_files: RegisterFiles,
) {
    for i in 0..io_workers {
        let command_rx = command_rx.clone();
        let rf = register_files.clone();
        let _ = std::thread::Builder::new()
            .name(format!("io_worker-{i}"))
            .spawn(move || run_worker(command_rx, defer_taskrun, rf))
            .unwrap();
    }
}

fn run_worker(command_rx: Receiver<IoPacket>, defer_taskrun: bool, register_files: RegisterFiles) {
    let mut pending: Slab<PendingIo> = Slab::with_capacity(MAX_IN_FLIGHT);

    let mut ring_builder = IoUring::<squeue::Entry, cqueue::Entry>::builder();
    ring_builder.setup_iopoll();
    if defer_taskrun {
        ring_builder
            .setup_single_issuer() // Available since 6.0
            .setup_defer_taskrun(); // Available since 6.1.
    }
    let mut ring = ring_builder
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    let (submitter, mut submit_queue, mut complete_queue) = ring.split();
    let mut retries = VecDeque::<IoPacket>::new();

    let files = register_files.wait_registration();
    //attempt using offsets
    let min_fd = *files.iter().min().unwrap();
    let max_fd = *files.iter().max().unwrap();
    let requreid_spaces = (max_fd - min_fd + 1).try_into().unwrap();
    submitter.register_files_sparse(requreid_spaces).unwrap();
    for fd in files {
        let offset = (fd - min_fd).try_into().unwrap();
        submitter.register_files_update(offset, &[fd]).unwrap();
    }
    let offset_to_remove = min_fd as u32;

    // attempt using a map
    //submitter.register_files(&files);
    //let mut map = std::collections::BTreeMap::<i32, u32>::new();
    //for (index, fd) in files.into_iter().enumerate() {
    //map.insert(fd, index as u32);
    //}

    loop {
        // 1. process completions.
        if !pending.is_empty() {
            if defer_taskrun {
                // TODO: add comment
                let _ = unsafe {
                    submitter.enter::<()>(0, 0, 1 /* IORING_ENTER_GETEVENTS */, None)
                };
            }

            complete_queue.sync();
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
        }

        // 2. accept new I/O requests when slab has space & submission queue is not full.
        let mut to_submit = false;

        submit_queue.sync();
        while pending.len() < MAX_IN_FLIGHT && !submit_queue.is_full() {
            let next_io = if !retries.is_empty() {
                // re-apply partially failed reads and writes
                // unwrap: known not empty
                retries.pop_front().unwrap()
            } else if pending.is_empty() {
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
            let pending_index = pending.insert(PendingIo {
                command: next_io.command,
                completion_sender: next_io.completion_sender,
            });

            let entry = submission_entry(
                &mut pending.get_mut(pending_index).unwrap().command,
                offset_to_remove,
                //&map,
            )
            .user_data(pending_index as u64);

            // unwrap: known not full
            unsafe { submit_queue.push(&entry).unwrap() };
        }

        // 3. submit all together.
        if to_submit {
            submit_queue.sync();
        }

        let wait = if pending.len() == MAX_IN_FLIGHT { 1 } else { 0 };

        submitter.submit_and_wait(wait).unwrap();
    }
}
//fn submission_entry(
//command: &mut IoCommand,
//map: &std::collections::BTreeMap<i32, u32>,
//) -> squeue::Entry {
//match command.kind {
//IoKind::Read(fd, page_index, ref mut page) => opcode::Read::new(
//types::Fixed(*map.get(&fd).unwrap()),
//page.as_mut_ptr(),
//PAGE_SIZE as u32,
//)
//.offset(page_index * PAGE_SIZE as u64)
//.build(),
//IoKind::Write(fd, page_index, ref page) => opcode::Write::new(
//types::Fixed(*map.get(&fd).unwrap()),
//page.as_ptr(),
//PAGE_SIZE as u32,
//)
//.offset(page_index * PAGE_SIZE as u64)
//.build(),
//IoKind::WriteArc(fd, page_index, ref page) => {
//let page: &[u8] = &*page;
//opcode::Write::new(
//types::Fixed(*map.get(&fd).unwrap()),
//page.as_ptr(),
//PAGE_SIZE as u32,
//)
//.offset(page_index * PAGE_SIZE as u64)
//.build()
//}
//IoKind::WriteRaw(fd, page_index, ref page) => opcode::Write::new(
//types::Fixed(*map.get(&fd).unwrap()),
//page.as_ptr(),
//PAGE_SIZE as u32,
//)
//.offset(page_index * PAGE_SIZE as u64)
//.build(),
//}
//}

fn submission_entry(command: &mut IoCommand, offset_to_remove: u32) -> squeue::Entry {
    match command.kind {
        IoKind::Read(fd, page_index, ref mut page) => opcode::Read::new(
            types::Fixed(fd as u32 - offset_to_remove),
            page.as_mut_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index * PAGE_SIZE as u64)
        .build(),
        IoKind::Write(fd, page_index, ref page) => opcode::Write::new(
            types::Fixed(fd as u32 - offset_to_remove),
            page.as_ptr(),
            PAGE_SIZE as u32,
        )
        .offset(page_index * PAGE_SIZE as u64)
        .build(),
        IoKind::WriteArc(fd, page_index, ref page) => {
            let page: &[u8] = &*page;
            opcode::Write::new(
                types::Fixed(fd as u32 - offset_to_remove),
                page.as_ptr(),
                PAGE_SIZE as u32,
            )
            .offset(page_index * PAGE_SIZE as u64)
            .build()
        }
        IoKind::WriteRaw(fd, page_index, ref page) => opcode::Write::new(
            types::Fixed(fd as u32 - offset_to_remove),
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
