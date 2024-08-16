use crossbeam::channel::{Receiver, Sender};
use nomt_core::page_id::PageId;
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;

use crate::{
    io::{CompleteIo, IoCommand, IoKind, Page, PAGE_SIZE},
    page_cache::PageDiff,
};

use self::{
    meta_map::MetaMap,
    store::Store,
    wal::{Batch as WalBatch, ConsistencyError, Entry as WalEntry, WalWriter},
};

pub use self::store::create;

mod meta_map;
mod store;
mod wal;

const LOAD_PAGE_HANDLE_INDEX: usize = 0;
const COMMIT_HANDLE_INDEX: usize = 1;
const NUM_IO_HANDLES: usize = 2;

/// The index of a bucket within the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketIndex(u64);

pub struct DB {
    shared: Arc<Shared>,
}

pub struct Shared {
    store: Store,
    // TODO: probably RwLock can be avoided being used only during commit
    wal: RwLock<WalWriter>,
    meta_map: RwLock<MetaMap>,
    // channel used to send requests to the dispatcher
    get_page_tx: Mutex<Sender<(u64 /*bucket*/, Sender<Page>)>>,
    io_sender: Sender<IoCommand>,
    commit_page_receiver: Receiver<CompleteIo>,
    _dispatcher_tp: ThreadPool,
}

impl DB {
    /// Opens an existing bitbox database.
    pub fn open(
        sync_seqn: u32,
        num_pages: u32,
        num_rings: usize,
        path: PathBuf,
    ) -> anyhow::Result<Self> {
        // TODO: refactor to use u32.
        let sync_seqn = sync_seqn as u64;

        let wal_path = path.join("wal");

        let (store, meta_map) = match store::Store::open(num_pages, path.clone()) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening store: {e:?}");
            }
        };

        // Open the WAL, check its integrity and make sure the store is consistent with it
        let wal = wal::WalChecker::open_and_recover(wal_path.clone());
        let _pending_batch = match wal.check_consistency(sync_seqn) {
            Ok(()) => {
                println!("Wal and Store are consistent, last sequence number: {sync_seqn}");
                None
            }
            Err(ConsistencyError::LastBatchCrashed(crashed_batch)) => {
                println!(
                    "Wal and Store are not consistent, pending sequence number: {}",
                    sync_seqn + 1
                );
                Some(crashed_batch)
            }
            Err(ConsistencyError::NotConsistent(wal_seqn)) => {
                // This is useful for testing. If the WAL sequence number is zero, it means the WAL is empty.
                // For example, it could have been deleted, and it's okay to continue working on the store
                // by appending new batches to the new WAL
                if wal_seqn == 0 {
                    None
                } else {
                    panic!(
                        "Store and Wal have two inconsistent serial numbers. wal: {wal_seqn}, store: {sync_seqn}"
                    );
                }
            }
        };

        // Create a WalWriter, able to append new batch and prune older ones
        let wal = match wal::WalWriter::open(wal_path) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening wal: {e:?}")
            }
        };

        // Spawn io_workers
        let (io_sender, io_receivers) = crate::io::start_io_worker(NUM_IO_HANDLES, num_rings);

        let load_page_sender = io_sender.clone();
        let load_page_receiver = io_receivers[LOAD_PAGE_HANDLE_INDEX].clone();
        let commit_page_receiver = io_receivers[COMMIT_HANDLE_INDEX].clone();

        // Spawn PageRequest dispatcher
        let (get_page_tx, get_page_rx) = crossbeam::channel::unbounded();

        let dispatcher_tp = threadpool::Builder::new()
            .num_threads(NUM_IO_HANDLES)
            .thread_name("nomt-io-dispatcher".to_string())
            .build();

        dispatcher_tp.execute(page_dispatcher_task(
            store.clone(),
            load_page_sender,
            load_page_receiver,
            get_page_rx,
        ));

        Ok(Self {
            shared: Arc::new(Shared {
                store,
                commit_page_receiver,
                wal: RwLock::new(wal),
                meta_map: RwLock::new(meta_map),
                get_page_tx: Mutex::new(get_page_tx),
                _dispatcher_tp: dispatcher_tp,
                io_sender,
            }),
        })
    }

    /// Return a bucket allocator, used to determine the buckets which any newly inserted pages
    /// will clear.
    pub fn bucket_allocator(&self) -> BucketAllocator {
        BucketAllocator {
            shared: self.shared.clone(),
            changed_buckets: HashMap::new(),
        }
    }

    pub fn get(&self, page_id: &PageId) -> anyhow::Result<Option<(Vec<u8>, BucketIndex)>> {
        let mut probe_seq = ProbeSequence::new(page_id, &self.shared.meta_map.read());

        loop {
            match probe_seq.next(&self.shared.meta_map.read()) {
                ProbeResult::Tombstone(_) => continue,
                ProbeResult::PossibleHit(bucket) => {
                    // if this could be a match we need to fetch the page and check its id

                    // send the read command
                    let (tx, rx) = crossbeam::channel::bounded::<Page>(1);
                    {
                        let get_page_tx = self.shared.get_page_tx.lock().unwrap();
                        get_page_tx.send((bucket, tx)).unwrap();
                    }

                    // wait for the dispacther
                    let Ok(page) = rx.recv() else {
                        panic!("something went wrong requesting pages");
                    };

                    if &page[PAGE_SIZE - 32..] == page_id.encode() {
                        break Ok(Some((page.to_vec(), BucketIndex(bucket))));
                    }
                }
                ProbeResult::Empty(_bucket) => break Ok(None),
            }
        }
    }

    // TODO: update with async sync apporach
    pub fn sync_begin(
        &self,
        changes: Vec<(PageId, BucketIndex, Option<(Vec<u8>, PageDiff)>)>,
        sync_seqn: u32,
    ) -> anyhow::Result<u64> {
        // Steps are:
        // 0. Increase sequence number
        // 1. compute the WalBatch
        // 2. append WalBatch to wal and fsync
        // 4. write new pages
        // 5. write meta map
        // 6. fsync
        // 7. write meta page and fsync
        // 8. prune the wal

        let mut wal = self.shared.wal.write();
        let mut meta_map = self.shared.meta_map.write();

        let mut changed_meta_pages = HashSet::new();
        let next_sequence_number = sync_seqn as u64;
        let mut wal_batch = WalBatch::new(next_sequence_number);
        let mut bucket_writes = Vec::new();

        for (page_id, BucketIndex(bucket), page_info) in changes {
            // let's extract its bucket
            match page_info {
                Some((raw_page, page_diff)) => {
                    // the page_id must be written into the page itself
                    assert!(raw_page.len() == PAGE_SIZE);
                    let mut page = crate::bitbox::Page::zeroed();
                    page[..raw_page.len()].copy_from_slice(&raw_page);
                    page[PAGE_SIZE - 32..].copy_from_slice(&page_id.encode());

                    // update meta map with new info
                    let hash = hash_page_id(&page_id);
                    let meta_map_changed = meta_map.hint_not_match(bucket as usize, hash);
                    if meta_map_changed {
                        meta_map.set_full(bucket as usize, hash);
                        changed_meta_pages.insert(meta_map.page_index(bucket as usize));
                    }

                    // fill the WalBatch and the pages that need to be written to disk
                    wal_batch.append_entry(WalEntry::Update {
                        page_id: page_id.encode(),
                        changed: get_changed(&page, &page_diff),
                        page_diff: page_diff.get_raw(),
                        bucket_index: bucket as u64,
                    });

                    bucket_writes.push((bucket, page));
                }
                None => {
                    meta_map.set_tombstone(bucket as usize);
                    changed_meta_pages.insert(meta_map.page_index(bucket as usize));

                    wal_batch.append_entry(WalEntry::Clear {
                        bucket_index: bucket as u64,
                    });
                }
            };
        }

        let prev_wal_size = wal.file_size();
        wal_batch.data().len();
        wal.apply_batch(&wal_batch).unwrap();

        let mut submitted: u32 = 0;
        let mut completed: u32 = 0;

        // Issue all bucket writes
        for (bucket, page) in bucket_writes {
            let command = IoCommand {
                kind: IoKind::Write(
                    self.shared.store.store_fd(),
                    self.shared.store.data_page_index(bucket),
                    Box::new(page),
                ),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant.
            };
            // TODO: handle error
            self.shared.io_sender.send(command).unwrap();
            submitted += 1;
        }

        // apply changed meta pages
        for changed_meta_page in changed_meta_pages {
            let mut buf = Box::new(Page::zeroed());
            buf[..].copy_from_slice(meta_map.page_slice(changed_meta_page));
            let command = IoCommand {
                kind: IoKind::Write(
                    self.shared.store.store_fd(),
                    self.shared.store.meta_bytes_index(changed_meta_page as u64),
                    buf,
                ),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            };
            submitted += 1;
            // TODO: handle error
            self.shared.io_sender.send(command).unwrap();
        }

        // wait for all writes command to be finished
        while completed < submitted {
            let completion = self
                .shared
                .commit_page_receiver
                .recv()
                .expect("I/O worker dropped");
            assert!(completion.result.is_ok());
            completed += 1;
        }

        // sync all writes
        submit_and_wait_one(
            &self.shared.io_sender,
            &self.shared.commit_page_receiver,
            IoCommand {
                kind: IoKind::Fsync(self.shared.store.store_fd()),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            },
        );

        // sync meta page change.
        submit_and_wait_one(
            &self.shared.io_sender,
            &self.shared.commit_page_receiver,
            IoCommand {
                kind: IoKind::Fsync(self.shared.store.store_fd()),
                handle: COMMIT_HANDLE_INDEX,
                user_data: 0, // unimportant
            },
        );

        Ok(prev_wal_size)
    }

    pub fn sync_end(&self, prev_wal_size: u64) -> anyhow::Result<()> {
        // clear the WAL.
        let wal = self.shared.wal.write();
        wal.prune_front(prev_wal_size);
        Ok(())
    }
}

/// Helper used in constructing a transaction. Used for finding buckets in which to write pages.
pub struct BucketAllocator {
    shared: Arc<Shared>,
    // true: occupied. false: vacated
    changed_buckets: HashMap<u64, bool>,
}

impl BucketAllocator {
    /// Allocate a bucket for a page which is known not to exist in the hash-table.
    ///
    /// `allocate` and `free` must be called in the same order that items are passed to `commit`,
    /// or pages may silently disappear later.
    pub fn allocate(&mut self, page_id: PageId) -> BucketIndex {
        let meta_map = self.shared.meta_map.read();
        let mut probe_seq = ProbeSequence::new(&page_id, &meta_map);

        loop {
            match probe_seq.next(&meta_map) {
                ProbeResult::PossibleHit(bucket) => {
                    if self
                        .changed_buckets
                        .get(&bucket)
                        .map_or(false, |full| !full)
                    {
                        self.changed_buckets.insert(bucket, true);
                        return BucketIndex(bucket);
                    }
                }
                ProbeResult::Tombstone(bucket) => {
                    if self.changed_buckets.get(&bucket).map_or(true, |full| !full) {
                        self.changed_buckets.insert(bucket, true);
                        return BucketIndex(bucket);
                    }
                }
                ProbeResult::Empty(bucket) => {
                    self.changed_buckets.insert(bucket, true);
                    return BucketIndex(bucket);
                }
                _ => continue,
            }
        }
    }

    /// Free a bucket which is known to be occupied by the given page ID.
    pub fn free(&mut self, bucket_index: BucketIndex) {
        self.changed_buckets.insert(bucket_index.0, false);
    }
}

// call only when I/O queue is totally empty.
fn submit_and_wait_one(
    io_sender: &Sender<IoCommand>,
    io_receiver: &Receiver<CompleteIo>,
    command: IoCommand,
) {
    let _ = io_sender.send(command);
    let _ = io_receiver.recv();
}

fn get_changed(page: &Page, page_diff: &PageDiff) -> Vec<[u8; 32]> {
    page_diff
        .get_changed()
        .into_iter()
        .map(|changed| {
            let start = changed * 32;
            let end = start + 32;
            let mut node = [0; 32];
            node.copy_from_slice(&page[start..end]);
            node
        })
        .collect()
}

// A task that will receive all page requests and send back the result
// through the provided channel

// NOTE: It's not the best solution, as initial integration it avoids
// having to change the io_uring abstraction
fn page_dispatcher_task(
    store: Store,
    load_page_sender: Sender<IoCommand>,
    load_page_receiver: Receiver<CompleteIo>,
    get_page_rx: Receiver<(u64, Sender<Page>)>,
) -> impl Fn() {
    move || {
        let mut slab = slab::Slab::new();
        loop {
            crossbeam::select! {
                recv(get_page_rx) -> req => {
                    let Ok((bucket, tx)) = req else {
                        break;
                    };

                    let index = slab.insert(tx);
                    let command = IoCommand {
                        kind: IoKind::Read(store.store_fd(), store.data_page_index(bucket), Box::new(Page::zeroed())),
                        handle: LOAD_PAGE_HANDLE_INDEX,
                        user_data: index as u64,
                    };

                    // send the command
                    load_page_sender.send(command).unwrap();

                },
                recv(load_page_receiver) -> response => {
                    let Ok(CompleteIo { command, result }) = response else {
                        panic!("TODO")
                    };

                    assert!(result.is_ok());

                    let tx: Sender<Page> = slab.remove(command.user_data as usize);
                    tx.send(*command.kind.unwrap_buf()).unwrap();
                }
            }
        }
    }
}

fn hash_page_id(page_id: &PageId) -> u64 {
    let mut buf = [0u8; 8];
    // TODO: the seed of the store should be used
    buf.copy_from_slice(&blake3::hash(&page_id.encode()).as_bytes()[..8]);
    u64::from_le_bytes(buf)
}

#[derive(Clone, Copy)]
struct ProbeSequence {
    hash: u64,
    bucket: u64,
    step: u64,
}

pub enum ProbeResult {
    PossibleHit(u64),
    Empty(u64),
    Tombstone(u64),
}

impl ProbeSequence {
    pub fn new(page_id: &PageId, meta_map: &MetaMap) -> Self {
        let hash = hash_page_id(page_id);
        Self {
            hash,
            bucket: hash % meta_map.len() as u64,
            step: 0,
        }
    }

    // probe until there is a possible hit or an empty bucket is found
    pub fn next(&mut self, meta_map: &MetaMap) -> ProbeResult {
        loop {
            // Triangular probing
            self.bucket += self.step;
            self.step += 1;
            self.bucket %= meta_map.len() as u64;

            if meta_map.hint_empty(self.bucket as usize) {
                return ProbeResult::Empty(self.bucket);
            }

            if meta_map.hint_tombstone(self.bucket as usize) {
                return ProbeResult::Tombstone(self.bucket);
            }

            if meta_map.hint_not_match(self.bucket as usize, self.hash) {
                continue;
            }

            return ProbeResult::PossibleHit(self.bucket);
        }
    }

    pub fn bucket(&self) -> u64 {
        self.bucket
    }
}
