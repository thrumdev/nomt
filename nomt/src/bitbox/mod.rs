use anyhow::Context;
use crossbeam::channel::{TryRecvError, TrySendError};
use nomt_core::page_id::PageId;
use parking_lot::{ArcRwLockReadGuard, RwLock};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    os::fd::{AsRawFd, RawFd},
    sync::Arc,
};

use crate::{
    io::{IoCommand, IoHandle, IoKind, IoPool, Page, PAGE_SIZE},
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
    meta_map: Arc<RwLock<MetaMap>>,
    io_handle: IoHandle,
}

impl DB {
    /// Opens an existing bitbox database.
    pub fn open(
        io_pool: &IoPool,
        sync_seqn: u32,
        num_pages: u32,
        ht_fd: &File,
        wal_fd: &File,
    ) -> anyhow::Result<Self> {
        // TODO: refactor to use u32.
        let sync_seqn = sync_seqn as u64;

        let (store, meta_map) = match store::Store::open(num_pages, ht_fd) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening store: {e:?}");
            }
        };

        // Open the WAL, check its integrity and make sure the store is consistent with it
        let wal = wal::WalChecker::open_and_recover(wal_fd);
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
        let wal = match wal::WalWriter::open(wal_fd) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening wal: {e:?}")
            }
        };

        Ok(Self {
            shared: Arc::new(Shared {
                store,
                wal: RwLock::new(wal),
                meta_map: Arc::new(RwLock::new(meta_map)),
                io_handle: io_pool.make_handle(),
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

    pub fn sync_begin(
        &self,
        changes: Vec<(PageId, BucketIndex, Option<(Vec<u8>, PageDiff)>)>,
        sync_seqn: u32,
        ht_fd: &std::fs::File,
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
                    let mut page = Box::new(crate::bitbox::Page::zeroed());
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
                    ht_fd.as_raw_fd(),
                    self.shared.store.data_page_index(bucket),
                    page,
                ),
                user_data: 0, // unimportant.
            };
            self.shared
                .io_handle
                .send(command)
                .map_err(|_| anyhow::anyhow!("I/O Pool Disconnected"))?;
            submitted += 1;
        }

        // apply changed meta pages
        for changed_meta_page in changed_meta_pages {
            let mut buf = Box::new(Page::zeroed());
            buf[..].copy_from_slice(meta_map.page_slice(changed_meta_page));
            let command = IoCommand {
                kind: IoKind::Write(
                    ht_fd.as_raw_fd(),
                    self.shared.store.meta_bytes_index(changed_meta_page as u64),
                    buf,
                ),
                user_data: 0, // unimportant
            };
            submitted += 1;
            self.shared
                .io_handle
                .send(command)
                .map_err(|_| anyhow::anyhow!("I/O Pool Disconnected"))?;
        }

        // wait for all writes command to be finished
        while completed < submitted {
            let completion = self.shared.io_handle.recv()?;
            assert!(completion.result.is_ok());
            completed += 1;
        }

        // sync all writes
        ht_fd
            .sync_all()
            .context("ht file: error performing fsync")?;

        Ok(prev_wal_size)
    }

    pub fn sync_end(&self, prev_wal_size: u64) -> anyhow::Result<()> {
        // clear the WAL.
        let wal = self.shared.wal.write();
        wal.prune_front(prev_wal_size);
        Ok(())
    }
}

/// A utility for loading pages from bitbox.
pub struct PageLoader {
    shared: Arc<Shared>,
    meta_map: ArcRwLockReadGuard<parking_lot::RawRwLock, MetaMap>,
    io_handle: IoHandle,
}

impl PageLoader {
    /// Create a new page loader.
    pub fn new(db: &DB, io_handle: IoHandle) -> Self {
        PageLoader {
            shared: db.shared.clone(),
            meta_map: RwLock::read_arc(&db.shared.meta_map),
            io_handle,
        }
    }

    /// Create a new page load.
    pub fn start_load(&self, page_id: PageId) -> PageLoad {
        PageLoad {
            probe_sequence: ProbeSequence::new(&page_id, &self.meta_map),
            page_id,
            state: PageLoadState::Pending,
        }
    }

    /// Try to advance the state of the given page load. Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns a value indicating the state of the page load.
    /// The user_data is only relevant if `Submitted` is returned, in which case a completion will
    /// arrive with the same user data at some point.
    pub fn try_advance(
        &self,
        ht_fd: RawFd,
        load: &mut PageLoad,
        user_data: u64,
    ) -> anyhow::Result<PageLoadAdvance> {
        let bucket = match load.take_blocked() {
            Some(bucket) => bucket,
            None => loop {
                match load.probe_sequence.next(&self.meta_map) {
                    ProbeResult::Tombstone(_) => continue,
                    ProbeResult::Empty(_) => return Ok(PageLoadAdvance::GuaranteedFresh),
                    ProbeResult::PossibleHit(bucket) => break BucketIndex(bucket),
                }
            },
        };

        let data_page_index = self.shared.store.data_page_index(bucket.0);

        let command = IoCommand {
            kind: IoKind::Read(ht_fd, data_page_index, Box::new(Page::zeroed())),
            user_data,
        };

        match self.io_handle.try_send(command) {
            Ok(()) => {
                load.state = PageLoadState::Submitted;
                Ok(PageLoadAdvance::Submitted)
            }
            Err(TrySendError::Full(_)) => {
                load.state = PageLoadState::Blocked;
                Ok(PageLoadAdvance::Blocked)
            }
            Err(TrySendError::Disconnected(_)) => anyhow::bail!("I/O pool hangup"),
        }
    }

    /// Advance the state of the given page load, blocking the current thread.
    /// Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns `Ok(true)` if the page request has been submitted and a completion will be
    /// coming. `Ok(false)` means that the page is guaranteed to be fresh.
    pub fn advance(
        &self,
        ht_fd: RawFd,
        load: &mut PageLoad,
        user_data: u64,
    ) -> anyhow::Result<bool> {
        let bucket = match load.take_blocked() {
            Some(bucket) => bucket,
            None => loop {
                match load.probe_sequence.next(&self.meta_map) {
                    ProbeResult::Tombstone(_) => continue,
                    ProbeResult::Empty(_) => return Ok(false),
                    ProbeResult::PossibleHit(bucket) => break BucketIndex(bucket),
                }
            },
        };

        let data_page_index = self.shared.store.data_page_index(bucket.0);

        let command = IoCommand {
            kind: IoKind::Read(ht_fd, data_page_index, Box::new(Page::zeroed())),
            user_data,
        };

        match self.io_handle.send(command) {
            Ok(()) => {
                load.state = PageLoadState::Submitted;
                Ok(true)
            }
            Err(_) => anyhow::bail!("I/O pool hangup"),
        }
    }

    /// Try to receive the next completion, without blocking the current thread.
    ///
    /// Fails if the I/O pool is down or a request caused an I/O error.
    pub fn try_complete(&self) -> anyhow::Result<Option<PageLoadCompletion>> {
        match self.io_handle.try_recv() {
            Ok(completion) => {
                completion.result?;
                match completion.command.kind {
                    IoKind::Read(_, _, page) => Ok(Some(PageLoadCompletion {
                        page,
                        user_data: completion.command.user_data,
                    })),
                    _ => panic!(),
                }
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => anyhow::bail!("I/O pool hangup"),
        }
    }

    /// Receive the next completion, blocking the current thread.
    ///
    /// Fails if the I/O pool is down or a request caused an I/O error.
    pub fn complete(&self) -> anyhow::Result<PageLoadCompletion> {
        match self.io_handle.recv() {
            Ok(completion) => {
                completion.result?;
                match completion.command.kind {
                    IoKind::Read(_, _, page) => Ok(PageLoadCompletion {
                        page,
                        user_data: completion.command.user_data,
                    }),
                    _ => panic!(),
                }
            }
            Err(_) => anyhow::bail!("I/O pool hangup"),
        }
    }
}

/// Represents the completion of a page load.
pub struct PageLoadCompletion {
    page: Box<Page>,
    user_data: u64,
}

impl PageLoadCompletion {
    pub fn user_data(&self) -> u64 {
        self.user_data
    }

    pub fn apply_to(self, load: &mut PageLoad) -> Option<(Vec<u8>, BucketIndex)> {
        assert!(load.needs_completion());
        if self.page[PAGE_SIZE - 32..] == load.page_id.encode() {
            Some((
                self.page.to_vec(),
                BucketIndex(load.probe_sequence.bucket()),
            ))
        } else {
            load.state = PageLoadState::Pending;
            None
        }
    }
}

/// The result of advancing a page load.
pub enum PageLoadAdvance {
    /// The page load is blocked by I/O backpressure.
    Blocked,
    /// The page load was submitted and a completion will follow.
    Submitted,
    /// The page is guaranteed not to exist.
    GuaranteedFresh,
}

pub struct PageLoad {
    page_id: PageId,
    probe_sequence: ProbeSequence,
    state: PageLoadState,
}

impl PageLoad {
    pub fn needs_completion(&self) -> bool {
        self.state == PageLoadState::Submitted
    }

    pub fn page_id(&self) -> &PageId {
        &self.page_id
    }

    fn take_blocked(&mut self) -> Option<BucketIndex> {
        match std::mem::replace(&mut self.state, PageLoadState::Pending) {
            PageLoadState::Pending => None,
            PageLoadState::Blocked => Some(BucketIndex(self.probe_sequence.bucket())),
            PageLoadState::Submitted => panic!("attempted to re-submit page load"),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum PageLoadState {
    Pending,
    Blocked,
    Submitted,
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

        let mut i = 0;
        loop {
            i += 1;
            assert!(i < 10000, "hash-table full");
            match probe_seq.next(&meta_map) {
                ProbeResult::PossibleHit(bucket) => {
                    // skip unless another page has freed the bucket.
                    if self
                        .changed_buckets
                        .get(&bucket)
                        .map_or(false, |full| !full)
                    {
                        self.changed_buckets.insert(bucket, true);
                        return BucketIndex(bucket);
                    }
                }
                ProbeResult::Tombstone(bucket) | ProbeResult::Empty(bucket) => {
                    // unless some other page has taken the bucket, fill it.
                    if self.changed_buckets.get(&bucket).map_or(true, |full| !full) {
                        self.changed_buckets.insert(bucket, true);
                        return BucketIndex(bucket);
                    }
                }
            }
        }
    }

    /// Free a bucket which is known to be occupied by the given page ID.
    pub fn free(&mut self, bucket_index: BucketIndex) {
        self.changed_buckets.insert(bucket_index.0, false);
    }
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
