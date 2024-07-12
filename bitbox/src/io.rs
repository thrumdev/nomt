use crossbeam_channel::{Receiver, Sender, TryRecvError};
use io_uring::{cqueue, opcode, squeue, types, IoUring};
use rand::prelude::SliceRandom;
use slab::Slab;
use std::{
    os::fd::RawFd,
    time::{Duration, Instant},
};

use crate::store::{Page, PAGE_SIZE};

const RING_CAPACITY: u32 = 128;

// max number of inflight requests is bounded by the slab.
const MAX_IN_FLIGHT: usize = RING_CAPACITY as usize;

pub type HandleIndex = usize;

#[derive(Clone)]
pub enum IoKind {
    Read(RawFd, u64, Box<Page>),
    Write(RawFd, u64, Box<Page>),
    WriteRaw(RawFd, u64, *const u8, usize),
    Fsync(RawFd),
}

impl IoKind {
    pub fn unwrap_buf(self) -> Box<Page> {
        match self {
            IoKind::Read(_, _, buf) | IoKind::Write(_, _, buf) => buf,
            IoKind::WriteRaw(_, _, _, _) => panic!("attempted to extract buf from write_raw"),
            IoKind::Fsync(_) => panic!("attempted to extract buf from fsync"),
        }
    }
}

unsafe impl Send for IoKind {}

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

struct PendingIo {
    command: IoCommand,
    start: Instant,
}

#[derive(Clone, Copy)]
pub enum Mode {
    /// actually use io_uring
    Real {
        /// The number of rings to maintain.
        num_rings: usize,
    },
    /// complete io_requests after a random latency.
    #[allow(unused)]
    Fake,
}

/// Create an I/O worker managing an io_uring and sending responses back via channels to a number
/// of handles.
pub fn start_io_worker(
    num_handles: usize,
    mode: Mode,
) -> (Sender<IoCommand>, Vec<Receiver<CompleteIo>>) {
    // main bound is from the pending slab.
    let (command_tx, command_rx) = crossbeam_channel::bounded(MAX_IN_FLIGHT * 2);
    let (handle_txs, handle_rxs) = (0..num_handles)
        .map(|_| crossbeam_channel::unbounded())
        .unzip();

    match mode {
        Mode::Real { num_rings } => {
            let _ = std::thread::Builder::new()
                .name("io_ingress".to_string())
                .spawn(move || run_ingress(command_rx, handle_txs, num_rings))
                .unwrap();
        }
        Mode::Fake => {
            let _ = std::thread::Builder::new()
                .name("io_worker".to_string())
                .spawn(move || run_fake_worker(command_rx, handle_txs))
                .unwrap();
        }
    }

    (command_tx, handle_rxs)
}

fn run_ingress(
    command_rx: Receiver<IoCommand>,
    handle_txs: Vec<Sender<CompleteIo>>,
    num_rings: usize,
) {
    if num_rings == 1 {
        run_worker(command_rx, handle_txs);
        return;
    }

    let mut worker_command_txs = Vec::with_capacity(num_rings);
    for i in 0..num_rings {
        let handle_txs = handle_txs.clone();
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let _ = std::thread::Builder::new()
            .name(format!("io_worker-{i}"))
            .spawn(move || run_worker(command_rx, handle_txs))
            .unwrap();
        worker_command_txs.push(command_tx);
    }

    let mut next_worker_ix = 0;
    loop {
        match command_rx.recv() {
            Ok(command) => {
                let _ = worker_command_txs[next_worker_ix].send(command);
                next_worker_ix = (next_worker_ix + 1) % num_rings;
            }
            Err(_) => return,
        }
    }
}

fn run_worker(command_rx: Receiver<IoCommand>, handle_tx: Vec<Sender<CompleteIo>>) {
    let mut pending: Slab<PendingIo> = Slab::with_capacity(MAX_IN_FLIGHT);

    let mut ring = IoUring::<squeue::Entry, cqueue::Entry>::builder()
        .setup_iopoll()
        .build(RING_CAPACITY)
        .expect("Error building io_uring");

    let (submitter, mut submit_queue, mut complete_queue) = ring.split();
    let mut stats = Stats::new();

    loop {
        stats.log();

        // 1. process completions.
        if !pending.is_empty() {
            complete_queue.sync();
            while let Some(completion_event) = complete_queue.next() {
                if pending.get(completion_event.user_data() as usize).is_none() {
                    continue;
                }
                let PendingIo { command, start } =
                    pending.remove(completion_event.user_data() as usize);

                stats.note_completion(start.elapsed().as_micros() as u64);

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

        submit_queue.sync();
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
            stats.note_arrival();

            to_submit = true;
            let pending_index = pending.insert(PendingIo {
                command: next_io,
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
        IoKind::Fsync(fd) => opcode::Fsync::new(types::Fd(fd)).build(),
        IoKind::WriteRaw(fd, page_index, ptr, size) => {
            opcode::Write::new(types::Fd(fd), ptr, size as u32)
                .offset(page_index * PAGE_SIZE as u64)
                .build()
        }, 
    }
}

struct FakePendingIo {
    command: IoCommand,
    duration: u64,
    end: Instant,
}

fn possible_latencies(latency_occurrences: Vec<(u64, usize)>) -> Vec<u64> {
    latency_occurrences
        .into_iter()
        .flat_map(|(l, x)| std::iter::repeat(l).take(x))
        .collect()
}

fn run_fake_worker(command_rx: Receiver<IoCommand>, handle_tx: Vec<Sender<CompleteIo>>) {
    // EV = ~319us with this distribution.
    let possible_latencies = possible_latencies(vec![
        (50, 1),
        (75, 1),
        (150, 2),
        (250, 5),
        (300, 10),
        (400, 5),
        (500, 2),
        (600, 1),
        (700, 1),
    ]);

    let mut stats = Stats::new();
    let mut pending: Slab<FakePendingIo> = Slab::with_capacity(MAX_IN_FLIGHT);
    let mut rng = rand::thread_rng();

    loop {
        stats.log();

        let now = Instant::now();
        for i in 0..MAX_IN_FLIGHT {
            if pending.get(i).map_or(false, |i| i.end <= now) {
                let item = pending.remove(i);

                stats.note_completion(item.duration as u64);

                let command = item.command;
                let handle_idx = command.handle;
                let complete = CompleteIo {
                    command,
                    result: Ok(()),
                };
                if let Err(_) = handle_tx[handle_idx].send(complete) {
                    // TODO: handle?
                    break;
                }
            }
        }

        if pending.len() == MAX_IN_FLIGHT {
            continue;
        }
        let next_io = if pending.is_empty() {
            match command_rx.recv() {
                Ok(io) => io,
                Err(_) => break,
            }
        } else {
            match command_rx.try_recv() {
                Ok(io) => io,
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => continue,
            }
        };

        let duration = *possible_latencies.choose(&mut rng).unwrap();
        let end = Instant::now() + Duration::from_micros(duration);
        pending.insert(FakePendingIo {
            command: next_io,
            duration,
            end,
        });
        stats.note_arrival();
    }
}
