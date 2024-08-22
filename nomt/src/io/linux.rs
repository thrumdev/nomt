use super::{CompleteIo, IoCommand, IoKind, IoKindResult, IoPacket, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use slab::Slab;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

const RING_CAPACITY: u32 = 128;

// max number of inflight requests is bounded by the slab.
const MAX_IN_FLIGHT: usize = RING_CAPACITY as usize;

struct PendingIo {
    command: IoCommand,
    completion_sender: Sender<CompleteIo>,
    start: Instant,
}

pub fn start_io_worker(io_workers: usize) -> Sender<IoPacket> {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::bounded(MAX_IN_FLIGHT * 2);

    let _ = std::thread::Builder::new()
        .name("io_ingress".to_string())
        .spawn(move || run_ingress(command_rx, io_workers))
        .unwrap();

    command_tx
}

fn run_ingress(command_rx: Receiver<IoPacket>, io_workers: usize) {
    if io_workers == 1 {
        run_worker(command_rx);
        return;
    }

    let mut worker_command_txs = Vec::with_capacity(io_workers);
    for i in 0..io_workers {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let _ = std::thread::Builder::new()
            .name(format!("io_worker-{i}"))
            .spawn(move || run_worker(command_rx))
            .unwrap();
        worker_command_txs.push(command_tx);
    }

    let mut next_worker_ix = 0;
    loop {
        match command_rx.recv() {
            Ok(command) => {
                let _ = worker_command_txs[next_worker_ix].send(command);
                next_worker_ix = (next_worker_ix + 1) % io_workers;
            }
            Err(_) => return,
        }
    }
}

fn run_worker(command_rx: Receiver<IoPacket>) {
    let mut pending: Slab<PendingIo> = Slab::with_capacity(MAX_IN_FLIGHT);

    let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
        .setup_iopoll()
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    let (submitter, mut submit_queue, mut complete_queue) = ring.split();
    let mut stats = Stats::new();
    let mut retries = VecDeque::<IoPacket>::new();

    loop {
        stats.log();

        // 1. process completions.
        if !pending.is_empty() {
            complete_queue.sync();
            while let Some(completion_event) = complete_queue.next() {
                if pending.get(completion_event.user_data() as usize).is_none() {
                    continue;
                }
                let PendingIo {
                    command,
                    completion_sender,
                    start,
                } = pending.remove(completion_event.user_data() as usize);

                stats.note_completion(start.elapsed().as_micros() as u64);

                // io_uring never uses errno to pass back error information.
                // Instead, completion_event.result() will contain what the equivalent
                // system call would have returned in case of success,
                // and in case of error completion_event.result() will contain -errno
                let io_uring_res = completion_event.result();
                let syscall_result = if io_uring_res >= 0 { io_uring_res } else { -1 };

                let result = match command.kind.get_result(syscall_result) {
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

                if let Err(_) = completion_sender.send(complete) {
                    // TODO: handle?
                    break;
                }
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

            stats.note_arrival();

            to_submit = true;
            let pending_index = pending.insert(PendingIo {
                command: next_io.command,
                completion_sender: next_io.completion_sender,
                start: Instant::now(),
            });

            let entry = submission_entry(&mut pending.get_mut(pending_index).unwrap().command)
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

struct Stats {
    last_log: Instant,
    total_inflight_us: u64,
    completions: usize,
    arrivals: usize,
}

impl Stats {
    fn new() -> Self {
        Stats {
            last_log: Instant::now(),
            total_inflight_us: 0,
            completions: 0,
            arrivals: 0,
        }
    }

    fn note_completion(&mut self, inflight_us: u64) {
        self.completions += 1;
        self.total_inflight_us += inflight_us;
    }

    fn note_arrival(&mut self) {
        self.arrivals += 1;
    }

    fn log(&mut self) {
        const LOG_DURATION: Duration = Duration::from_millis(1000);

        let elapsed = self.last_log.elapsed();
        if elapsed < LOG_DURATION {
            return;
        }

        self.last_log = Instant::now();
        let arrival_rate = self.arrivals as f64 * 1000.0 / elapsed.as_millis() as f64;
        let average_inflight = self.total_inflight_us as f64 / self.completions as f64;
        println!(
            "arrivals={} (rate {}/s) completions={} avg_inflight={}us | {}ms",
            self.arrivals,
            arrival_rate as usize,
            self.completions,
            average_inflight as usize,
            elapsed.as_millis(),
        );
        println!(
            "  estimated-QD={:.1}",
            arrival_rate * average_inflight / 1_000_000.0
        );

        self.completions = 0;
        self.arrivals = 0;
        self.total_inflight_us = 0;
    }
}

fn submission_entry(command: &mut IoCommand) -> squeue::Entry {
    match command.kind {
        IoKind::Read(fd, page_index, ref mut buf) => {
            opcode::Read::new(types::Fd(fd), buf.as_mut_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
        IoKind::Write(fd, page_index, ref buf) => {
            opcode::Write::new(types::Fd(fd), buf.as_ptr(), PAGE_SIZE as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
        IoKind::WriteRaw(fd, page_index, ptr, size) => {
            opcode::Write::new(types::Fd(fd), ptr, size as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }
    }
}
