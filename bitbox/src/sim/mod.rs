//! Simulation meant to emulate the workload of a merkle trie like NOMT in editing pages.
//!
//! We have a number of pages which are "in" the store, given by the range `0..num_pages`.
//! Workers each operate over pages with a unique prefix and randomly sample from this range
//! to choose pages to load, and randomly sample from `num_pages..` with `cold_rate` probability.
//!
//! The read workload is split across N workers (since they do some CPU work) and the write
//! workload happens on a single thread.
//!
//! The read workload involves reading pages and updating them, while the write workload involves
//! writing to a WAL and then doing random writes to update buckets / meta bits.
//!
//! Pages are always loaded in batches of `preload_count` and an extra page is fetched after this
//! with `load_extra_rate` probability. if `load_extra_rate` is zero it is like saying we always
//! expect to find a leaf after loading `preload_count` pages in NOMT. realistically, it should be
//! low but non-zero.
//!
//! Pages consist of the page ID followed by 126 32-byte vectors some of which are chosen to be
//! randomly updated at a rate of `page_item_update_rate`.

use ahash::RandomState;
use crossbeam_channel::{Receiver, Sender, TrySendError};

use std::collections::HashSet;
use std::sync::{Arc, Barrier, RwLock};

use crate::meta_map::MetaMap;
use crate::store::{
    io::{self as store_io, CompleteIo, IoCommand, IoKind, Mode as IoMode, PageIndex},
    Page, Store,
};

mod read;

#[derive(Clone, Copy)]
pub struct Params {
    pub num_workers: usize,
    pub num_rings: usize,
    pub num_pages: usize,
    pub workload_size: usize,
    pub cold_rate: f32,
    pub preload_count: usize,
    pub load_extra_rate: f32,
    pub page_item_update_rate: f32,
}

type PageId = [u8; 16];
type PageDiff = [u8; 16];

struct ChangedPage {
    page_id: PageId,
    bucket: Option<BucketIndex>,
    buf: Box<Page>,
    diff: PageDiff,
}

const SLOTS_PER_PAGE: usize = 126;

fn slot_range(slot_index: usize) -> std::ops::Range<usize> {
    let start = 32 + slot_index * 32;
    let end = start + 32;
    start..end
}

fn make_hasher(seed: [u8; 32]) -> RandomState {
    let extract_u64 = |range| {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&seed[range]);
        u64::from_le_bytes(buf)
    };

    RandomState::with_seeds(
        extract_u64(0..8),
        extract_u64(8..16),
        extract_u64(16..24),
        extract_u64(24..32),
    )
}

pub fn run_simulation(store: Arc<Store>, mut params: Params, meta_map: MetaMap) {
    params.num_pages /= params.num_workers;
    params.workload_size /= params.num_workers;

    let mut full_count = meta_map.full_count();
    println!("loaded map with {} buckets occupied", full_count);
    let meta_map = Arc::new(RwLock::new(meta_map));
    let (io_sender, mut io_receivers) = store_io::start_io_worker(
        store.clone(),
        params.num_workers + 1,
        IoMode::Real {
            num_rings: params.num_rings,
        },
    );
    let (page_changes_tx, page_changes_rx) = crossbeam_channel::unbounded();

    let write_io_receiver = io_receivers.pop().unwrap();

    let mut start_work_txs = Vec::new();
    for (i, io_receiver) in io_receivers.into_iter().enumerate() {
        let map = Map {
            io_sender: io_sender.clone(),
            io_receiver,
            hasher: make_hasher(store.seed()),
        };
        let meta_map = meta_map.clone();
        let page_changes_tx = page_changes_tx.clone();
        let (start_work_tx, start_work_rx) = crossbeam_channel::bounded(1);
        let _ = std::thread::Builder::new()
            .name("read_worker".to_string())
            .spawn(move || {
                read::run_worker(i, params, map, meta_map, page_changes_tx, start_work_rx)
            })
            .unwrap();
        start_work_txs.push(start_work_tx);
    }

    let io_handle_index = params.num_workers;

    let map = Map {
        io_sender,
        io_receiver: write_io_receiver,
        hasher: make_hasher(store.seed()),
    };
    loop {
        let barrier = Arc::new(Barrier::new(params.num_workers + 1));
        for tx in &start_work_txs {
            let _ = tx.send(barrier.clone());
        }

        // wait for reading to be done.
        let _ = barrier.wait();

        let mut meta_map = meta_map.write().unwrap();
        write(
            io_handle_index,
            &map,
            &page_changes_rx,
            &mut meta_map,
            &mut full_count,
        );
    }
}

type BucketIndex = u64;

#[derive(Clone, Copy)]
struct ProbeSequence {
    hash: u64,
    bucket: u64,
    step: u64,
}

struct Map {
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
    hasher: RandomState,
}

impl Map {
    fn begin_probe(&self, page_id: &PageId, meta_map: &MetaMap) -> ProbeSequence {
        let hash = self.hasher.hash_one(page_id);
        ProbeSequence {
            hash,
            bucket: hash % meta_map.len() as u64,
            step: 0,
        }
    }

    // search for the bucket the probed item may live in. returns `None` if it definitely does not
    // exist in the map. `Some` means it may exist in that bucket and needs to be probed.
    fn search(
        &self,
        meta_map: &MetaMap,
        mut probe_sequence: ProbeSequence,
    ) -> Option<ProbeSequence> {
        loop {
            probe_sequence.bucket += probe_sequence.step;
            probe_sequence.step += 1;
            probe_sequence.bucket %= meta_map.len() as u64;

            if meta_map.hint_empty(probe_sequence.bucket as usize) {
                return None;
            }

            if meta_map.hint_not_match(probe_sequence.bucket as usize, probe_sequence.hash) {
                continue;
            }
            return Some(probe_sequence);
        }
    }

    // search for the first empty bucket along a probe sequence.
    fn search_free(&self, meta_map: &MetaMap, mut probe_sequence: ProbeSequence) -> u64 {
        loop {
            probe_sequence.bucket += probe_sequence.step;
            probe_sequence.step += 1;
            probe_sequence.bucket %= meta_map.len() as u64;

            if meta_map.hint_empty(probe_sequence.bucket as usize)
                || meta_map.hint_tombstone(probe_sequence.bucket as usize)
            {
                return probe_sequence.bucket;
            }
        }
    }
}

fn write(
    io_handle_index: usize,
    map: &Map,
    changed_pages: &Receiver<ChangedPage>,
    meta_map: &mut MetaMap,
    full_count: &mut usize,
) {
    let mut changed_meta_pages = HashSet::new();
    let mut fresh_pages = HashSet::new();

    let start = std::time::Instant::now();

    let mut submitted = 0;
    let mut completed = 0;
    for changed in changed_pages.try_iter() {
        let bucket = match changed.bucket {
            Some(b) => b,
            None => {
                if !fresh_pages.insert(changed.page_id) {
                    continue;
                }
                let probe = map.begin_probe(&changed.page_id, &*meta_map);
                let bucket = map.search_free(&*meta_map, probe);
                changed_meta_pages.insert(meta_map.page_index(bucket as usize));
                meta_map.set_full(bucket as usize, probe.hash);
                bucket
            }
        };

        let command = IoCommand {
            kind: IoKind::Write(PageIndex::Data(bucket), changed.buf),
            handle: io_handle_index,
            user_data: 0, // unimportant.
        };

        submitted += 1;

        submit_write(&map.io_sender, &map.io_receiver, command, &mut completed);
    }

    for changed_meta_page in changed_meta_pages {
        let mut buf = Box::new(Page::zeroed());
        buf[..].copy_from_slice(meta_map.page_slice(changed_meta_page));
        let command = IoCommand {
            kind: IoKind::Write(PageIndex::MetaBytes(changed_meta_page as u64), buf),
            handle: io_handle_index,
            user_data: 0, // unimportant
        };

        submitted += 1;

        submit_write(&map.io_sender, &map.io_receiver, command, &mut completed);
    }

    while completed < submitted {
        await_completion(&map.io_receiver, &mut completed);
    }

    let command = IoCommand {
        kind: IoKind::Fsync,
        handle: io_handle_index,
        user_data: 0, // unimportant
    };

    submit_write(&map.io_sender, &map.io_receiver, command, &mut completed);

    submitted += 1;
    while completed < submitted {
        await_completion(&map.io_receiver, &mut completed);
    }

    *full_count += fresh_pages.len();

    println!(
        "finished write phase in {}ms, {}ios, {} IOPS, {} buckets full",
        start.elapsed().as_millis(),
        completed,
        1000.0 * completed as f64 / (start.elapsed().as_millis() as f64),
        full_count,
    );
}

fn submit_write(
    io_sender: &Sender<IoCommand>,
    io_receiver: &Receiver<CompleteIo>,
    command: IoCommand,
    completed: &mut usize,
) {
    let mut command = Some(command);
    while let Some(c) = command.take() {
        match io_sender.try_send(c) {
            Ok(()) => break,
            Err(TrySendError::Disconnected(_)) => panic!("I/O worker dropped"),
            Err(TrySendError::Full(c)) => {
                command = Some(c);
            }
        }

        await_completion(io_receiver, completed);
    }
}

fn await_completion(io_receiver: &Receiver<CompleteIo>, completed: &mut usize) {
    let completion = io_receiver.recv().expect("I/O worker dropped");
    assert!(completion.result.is_ok());
    *completed += 1;
}
