use crossbeam::channel::TryRecvError;
use crossbeam_channel::{Receiver, Sender};
use nomt_core::page_id::PageId;
use parking_lot::{ArcRwLockReadGuard, Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    os::{fd::AsRawFd, unix::fs::FileExt},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use threadpool::ThreadPool;

use crate::{
    io::{self, page_pool::FatPage, IoCommand, IoHandle, IoKind, PagePool, PAGE_SIZE},
    merkle,
    page_cache::PageCache,
    page_diff::PageDiff,
    store::MerkleTransaction,
};

use self::{ht_file::HTOffsets, meta_map::MetaMap};

pub use self::ht_file::create;
pub use wal::WalBlobBuilder;

mod ht_file;
mod meta_map;
mod wal;
pub(crate) mod writeout;

/// The index of a bucket within the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketIndex(u64);

#[derive(Clone)]
pub struct DB {
    shared: Arc<Shared>,
}

pub struct Shared {
    page_pool: PagePool,
    store: HTOffsets,
    seed: [u8; 16],
    meta_map: Arc<RwLock<MetaMap>>,
    wal_blob_builder: Arc<Mutex<WalBlobBuilder>>,
    occupied_buckets: AtomicUsize,
    wal_fd: File,
    ht_fd: File,
    sync_tp: ThreadPool,
}

impl DB {
    /// Opens an existing bitbox database.
    pub fn open(
        num_pages: u32,
        seed: [u8; 16],
        page_pool: PagePool,
        ht_fd: File,
        wal_fd: File,
    ) -> anyhow::Result<Self> {
        let (store, mut meta_map) = match ht_file::open(num_pages, &page_pool, &ht_fd) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening store: {e:?}");
            }
        };

        if wal_fd.metadata()?.len() > 0 {
            recover(&ht_fd, &wal_fd, &page_pool, &store, &mut meta_map, seed)?;
        }

        let occupied_buckets = meta_map.full_count();

        let wal_blob_builder = WalBlobBuilder::new()?;
        Ok(Self {
            shared: Arc::new(Shared {
                page_pool,
                store,
                seed,
                meta_map: Arc::new(RwLock::new(meta_map)),
                wal_blob_builder: Arc::new(Mutex::new(wal_blob_builder)),
                occupied_buckets: AtomicUsize::new(occupied_buckets),
                wal_fd,
                ht_fd,
                sync_tp: ThreadPool::with_name("bitbox-sync".into(), 2),
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

    pub fn sync(&self) -> SyncController {
        SyncController::new(self.clone())
    }

    fn prepare_sync(
        &self,
        page_pool: &PagePool,
        changes: Vec<(PageId, BucketIndex, Option<(FatPage, PageDiff)>)>,
        wal_blob_builder: &mut WalBlobBuilder,
    ) -> Vec<(u64, FatPage)> {
        wal_blob_builder.reset();

        let mut meta_map = self.shared.meta_map.write();

        let mut changed_meta_pages = HashSet::new();
        let mut ht_pages = Vec::new();

        let mut occupied_buckets_delta = 0isize;
        for (page_id, BucketIndex(bucket), page_info) in changes {
            // let's extract its bucket
            match page_info {
                Some((mut page, page_diff)) => {
                    page[PAGE_SIZE - 32..].copy_from_slice(&page_id.encode());

                    // update meta map with new info
                    let hash = hash_page_id(&page_id, &self.shared.seed);
                    let meta_map_changed = meta_map.hint_not_match(bucket as usize, hash);
                    if meta_map_changed {
                        occupied_buckets_delta += 1;
                        meta_map.set_full(bucket as usize, hash);
                        changed_meta_pages.insert(meta_map.page_index(bucket as usize));
                    }

                    wal_blob_builder.write_update(
                        page_id.encode(),
                        &page_diff,
                        page_diff.pack_changed_nodes(&page),
                        bucket,
                    );

                    let pn = self.shared.store.data_page_index(bucket);
                    ht_pages.push((pn, page));
                }
                None => {
                    occupied_buckets_delta -= 1;
                    meta_map.set_tombstone(bucket as usize);
                    changed_meta_pages.insert(meta_map.page_index(bucket as usize));
                    wal_blob_builder.write_clear(bucket);
                }
            };
        }

        for changed_meta_page in changed_meta_pages {
            let mut buf = page_pool.alloc_fat_page();
            buf[..].copy_from_slice(meta_map.page_slice(changed_meta_page));
            let pn = self.shared.store.meta_bytes_index(changed_meta_page as u64);
            ht_pages.push((pn, buf));
        }

        if cfg!(debug_assertions) {
            // Make sure that there are no duplicate pages.
            let orig_len = ht_pages.len();
            ht_pages.sort_unstable_by_key(|(pn, _)| *pn);
            ht_pages.dedup_by_key(|(pn, _)| *pn);
            assert_eq!(orig_len, ht_pages.len());
        }

        if occupied_buckets_delta < 0 {
            self.shared
                .occupied_buckets
                .fetch_sub(occupied_buckets_delta.abs() as usize, Ordering::Relaxed);
        } else if occupied_buckets_delta > 0 {
            self.shared
                .occupied_buckets
                .fetch_add(occupied_buckets_delta as usize, Ordering::Relaxed);
        }

        wal_blob_builder.finalize();

        ht_pages
    }
}

pub struct SyncController {
    db: DB,
    /// The channel to send the result of the WAL writeout. Option is to allow `take`.
    wal_result_tx: Option<Sender<anyhow::Result<()>>>,
    /// The channel to receive the result of the WAL writeout.
    wal_result_rx: Receiver<anyhow::Result<()>>,
    /// The pages along with their page numbers to write out to the HT file.
    ht_to_write: Arc<Mutex<Option<Vec<(u64, FatPage)>>>>,
}

impl SyncController {
    fn new(db: DB) -> Self {
        let (wal_result_tx, wal_result_rx) = crossbeam_channel::bounded(1);
        Self {
            db,
            wal_result_tx: Some(wal_result_tx),
            wal_result_rx,
            ht_to_write: Arc::new(Mutex::new(None)),
        }
    }

    /// Begins the sync process.
    ///
    /// Non-blocking.
    pub fn begin_sync(
        &mut self,
        page_cache: PageCache,
        mut merkle_tx: MerkleTransaction,
        page_diffs: merkle::PageDiffs,
    ) {
        let page_pool = self.db.shared.page_pool.clone();
        let bitbox = self.db.clone();
        let ht_to_write = self.ht_to_write.clone();
        let wal_blob_builder = self.db.shared.wal_blob_builder.clone();
        // UNWRAP: safe because begin_sync is called only once.
        let wal_result_tx = self.wal_result_tx.take().unwrap();
        self.db.shared.sync_tp.execute(move || {
            page_cache.prepare_transaction(page_diffs.into_iter(), &mut merkle_tx);

            let mut wal_blob_builder = wal_blob_builder.lock();
            let ht_pages =
                bitbox.prepare_sync(&page_pool, merkle_tx.new_pages, &mut *wal_blob_builder);
            drop(wal_blob_builder);

            Self::spawn_wal_writeout(wal_result_tx, bitbox);

            let mut ht_to_write = ht_to_write.lock();
            *ht_to_write = Some(ht_pages);

            // evict outside of the critical path.
            page_cache.evict();
        });
    }

    fn spawn_wal_writeout(wal_result_tx: Sender<anyhow::Result<()>>, bitbox: DB) {
        let bitbox = bitbox.clone();
        let tp = bitbox.shared.sync_tp.clone();
        tp.execute(move || {
            let wal_blob_builder = bitbox.shared.wal_blob_builder.lock();
            let wal_slice = wal_blob_builder.as_slice();
            let wal_result = writeout::write_wal(&bitbox.shared.wal_fd, wal_slice);
            let _ = wal_result_tx.send(wal_result);
        });
    }

    /// Wait for the pre-meta WAL file to be written out.
    ///
    /// Must be invoked by the sync thread. Blocking.
    pub fn wait_pre_meta(&self) -> anyhow::Result<()> {
        match self.wal_result_rx.recv() {
            Ok(wal_result) => wal_result,
            Err(_) => panic!("unexpected hungup"),
        }
    }

    /// Write out the HT pages and truncate the WAL file.
    ///
    /// Has to be called after the manifest is updated. Must be invoked by the sync
    /// thread. Blocking.
    pub fn post_meta(&self, io_handle: IoHandle) -> anyhow::Result<()> {
        let ht_pages = self.ht_to_write.lock().take().unwrap();
        writeout::write_ht(io_handle, &self.db.shared.ht_fd, ht_pages)?;
        writeout::truncate_wal(&self.db.shared.wal_fd)?;
        Ok(())
    }
}

/// Perform recovery by applying the WAL to the HT file.
fn recover(
    ht_fd: &File,
    mut wal_fd: &File,
    page_pool: &PagePool,
    ht_offsets: &HTOffsets,
    meta_map: &mut MetaMap,
    seed: [u8; 16],
) -> anyhow::Result<()> {
    use crate::bitbox::wal::WalBlobReader;
    use std::io::{Seek, SeekFrom};

    wal_fd.seek(SeekFrom::Start(0))?;

    // The indicies of pages (in the metabits page space) that were changed and require updates.
    // Note those are not ht page numbers yet and still require additional conversion.
    let mut changed_meta_page_ixs = HashSet::new();
    let mut wal_reader = WalBlobReader::new(page_pool, wal_fd)?;

    while let Some(entry) = wal_reader.read_entry()? {
        match entry {
            wal::WalEntry::Clear { bucket } => {
                meta_map.set_tombstone(bucket as usize);

                // Note that the meta page requires update.
                changed_meta_page_ixs.insert(meta_map.page_index(bucket as usize));
            }
            wal::WalEntry::Update {
                page_id,
                page_diff,
                changed_nodes,
                bucket,
            } => {
                let hash = hash_raw_page_id(page_id, &seed);
                let meta_map_changed = meta_map.hint_not_match(bucket as usize, hash);
                if meta_map_changed {
                    meta_map.set_full(bucket as usize, hash);
                    // Note that the meta page requires update.
                    changed_meta_page_ixs.insert(meta_map.page_index(bucket as usize));
                }

                // Apply the diff to the page in the ht file.
                //
                // The algorithm is:
                // - read the bucket page from the ht file.
                // - for each index of a bit in a diff that equals to 1, copy the changed node into
                //   the page.
                // - store the changed page.
                let pn = ht_offsets.data_page_index(bucket);

                let mut page = io::read_page(page_pool, ht_fd, pn)?;
                if page_diff.count() != changed_nodes.len() {
                    anyhow::bail!(
                        "mismatched number of changed nodes: {} != {}",
                        page_diff.count(),
                        changed_nodes.len()
                    );
                }
                page_diff.unpack_changed_nodes(&changed_nodes, &mut page);

                ht_fd.write_all_at(&page, pn * PAGE_SIZE as u64)?;
            }
        }
    }

    // Now that we have applied all the updates, we know precisely which meta pages have been
    // updated.
    //
    // We now write those pages out to the HT file.
    for changed_meta_page_ix in changed_meta_page_ixs {
        unsafe {
            let page = page_pool.alloc();
            // SAFETY: page is a fresh allocation from page pool and it's not aliased.
            let page_data = page.as_mut_slice();
            page_data[..].copy_from_slice(meta_map.page_slice(changed_meta_page_ix));

            let pn = ht_offsets.meta_bytes_index(changed_meta_page_ix as u64);
            ht_fd.write_all_at(page_data, pn * PAGE_SIZE as u64)?;

            page_pool.dealloc(page);
        }
    }

    // Finally, we collapse the WAL file.
    wal_fd.set_len(0)?;

    Ok(())
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
            probe_sequence: ProbeSequence::new(&page_id, &self.meta_map, &self.shared.seed),
            page_id,
            state: PageLoadState::Pending,
        }
    }

    /// Advance the state of the given page load, blocking the current thread.
    /// Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns `Ok(true)` if the page request has been submitted and a completion will be
    /// coming. `Ok(false)` means that the page is guaranteed to be fresh.
    pub fn advance(&self, load: &mut PageLoad, user_data: u64) -> anyhow::Result<bool> {
        let bucket = loop {
            match load.probe_sequence.next(&self.meta_map) {
                ProbeResult::Tombstone(_) => continue,
                ProbeResult::Empty(_) => return Ok(false),
                ProbeResult::PossibleHit(bucket) => break BucketIndex(bucket),
            }
        };

        let data_page_index = self.shared.store.data_page_index(bucket.0);

        let page = self.io_handle.page_pool().alloc_fat_page();
        let command = IoCommand {
            kind: IoKind::Read(self.shared.ht_fd.as_raw_fd(), data_page_index, page),
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

    /// Get the underlying I/O handle.
    pub fn io_handle(&self) -> &IoHandle {
        &self.io_handle
    }
}

/// Represents the completion of a page load.
pub struct PageLoadCompletion {
    page: FatPage,
    user_data: u64,
}

impl PageLoadCompletion {
    pub fn user_data(&self) -> u64 {
        self.user_data
    }

    pub fn apply_to(self, load: &mut PageLoad) -> Option<(FatPage, BucketIndex)> {
        assert!(load.needs_completion());
        if self.page[PAGE_SIZE - 32..] == load.page_id.encode() {
            Some((self.page, BucketIndex(load.probe_sequence.bucket())))
        } else {
            load.state = PageLoadState::Pending;
            None
        }
    }
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
}

#[derive(Clone, Copy, PartialEq)]
enum PageLoadState {
    Pending,
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
        let mut probe_seq = ProbeSequence::new(&page_id, &meta_map, &self.shared.seed);

        let mut i = 0;
        loop {
            i += 1;
            assert!(i < 10000, "hash-table full");
            match probe_seq.next(&meta_map) {
                ProbeResult::PossibleHit(_) => continue,
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

fn hash_page_id(page_id: &PageId, seed: &[u8; 16]) -> u64 {
    hash_raw_page_id(page_id.encode(), seed)
}

fn hash_raw_page_id(page_id: [u8; 32], seed: &[u8; 16]) -> u64 {
    let mut buf = [0u8; 8];
    let mut hasher = blake3::Hasher::new();
    hasher.update(&page_id);
    hasher.update(&seed[..]);
    buf.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    u64::from_le_bytes(buf)
}

#[derive(Clone, Copy)]
struct ProbeSequence {
    hash: u64,
    bucket: u64,
    step: u64,
}

enum ProbeResult {
    PossibleHit(u64),
    Empty(u64),
    Tombstone(u64),
}

impl ProbeSequence {
    fn new(page_id: &PageId, meta_map: &MetaMap, seed: &[u8; 16]) -> Self {
        let hash = hash_page_id(page_id, seed);
        Self {
            hash,
            bucket: hash % meta_map.len() as u64,
            step: 0,
        }
    }

    // probe until there is a possible hit or an empty bucket is found
    fn next(&mut self, meta_map: &MetaMap) -> ProbeResult {
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

    fn bucket(&self) -> u64 {
        self.bucket
    }
}
