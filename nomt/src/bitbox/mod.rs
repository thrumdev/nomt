use crossbeam::channel::{TryRecvError, TrySendError};
use nomt_core::page_id::PageId;
use parking_lot::{ArcRwLockReadGuard, RwLock};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    os::fd::RawFd,
    sync::Arc,
};

use crate::{
    io::{self, IoCommand, IoHandle, IoKind, Page, PAGE_SIZE},
    page_diff::PageDiff,
};

use self::{ht_file::HTOffsets, meta_map::MetaMap};

pub use self::ht_file::create;
pub use wal::WalBlobBuilder;

mod ht_file;
mod meta_map;
mod wal;

/// The index of a bucket within the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketIndex(u64);

pub struct DB {
    shared: Arc<Shared>,
}

pub struct Shared {
    store: HTOffsets,
    seed: [u8; 16],
    meta_map: Arc<RwLock<MetaMap>>,
}

impl DB {
    /// Opens an existing bitbox database.
    pub fn open(
        num_pages: u32,
        seed: [u8; 16],
        ht_fd: &File,
        wal_fd: &File,
    ) -> anyhow::Result<Self> {
        let (store, mut meta_map) = match ht_file::open(num_pages, ht_fd) {
            Ok(x) => x,
            Err(e) => {
                anyhow::bail!("encountered error in opening store: {e:?}");
            }
        };

        if wal_fd.metadata()?.len() > 0 {
            recover(ht_fd, wal_fd, &store, &mut meta_map, seed)?;
        }

        Ok(Self {
            shared: Arc::new(Shared {
                store,
                seed,
                meta_map: Arc::new(RwLock::new(meta_map)),
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

    pub fn prepare_sync(
        &self,
        changes: Vec<(PageId, BucketIndex, Option<(Box<Page>, PageDiff)>)>,
        wal_blob_builder: &mut WalBlobBuilder,
    ) -> anyhow::Result<WriteoutData> {
        let mut meta_map = self.shared.meta_map.write();

        let mut changed_meta_pages = HashSet::new();
        let mut ht_pages = Vec::new();

        for (page_id, BucketIndex(bucket), page_info) in changes {
            // let's extract its bucket
            match page_info {
                Some((mut page, page_diff)) => {
                    page[PAGE_SIZE - 32..].copy_from_slice(&page_id.encode());

                    // update meta map with new info
                    let hash = hash_page_id(&page_id, &self.shared.seed);
                    let meta_map_changed = meta_map.hint_not_match(bucket as usize, hash);
                    if meta_map_changed {
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
                    meta_map.set_tombstone(bucket as usize);
                    changed_meta_pages.insert(meta_map.page_index(bucket as usize));
                    wal_blob_builder.write_clear(bucket);
                }
            };
        }

        for changed_meta_page in changed_meta_pages {
            let mut buf = Box::new(Page::zeroed());
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

        Ok(WriteoutData { ht_pages })
    }
}

/// Perform recovery by applying the WAL to the HT file.
fn recover(
    mut ht_fd: &File,
    mut wal_fd: &File,
    ht_offsets: &HTOffsets,
    meta_map: &mut MetaMap,
    seed: [u8; 16],
) -> anyhow::Result<()> {
    use crate::bitbox::wal::WalBlobReader;
    use std::io::{Seek, SeekFrom, Write};

    wal_fd.seek(SeekFrom::Start(0))?;

    // The indicies of pages (in the metabits page space) that were changed and require updates.
    // Note those are not ht page numbers yet and still require additional conversion.
    let mut changed_meta_page_ixs = HashSet::new();
    let mut wal_reader = WalBlobReader::new(wal_fd)?;

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
                let mut page = io::read_page(ht_fd, pn)?;

                if page_diff.count() != changed_nodes.len() {
                    anyhow::bail!(
                        "mismatched number of changed nodes: {} != {}",
                        page_diff.count(),
                        changed_nodes.len()
                    );
                }
                page_diff.unpack_changed_nodes(&changed_nodes, &mut *page);

                ht_fd.seek(SeekFrom::Start(pn * PAGE_SIZE as u64))?;
                ht_fd.write_all(&page)?;
            }
        }
    }

    // Now that we have applied all the updates, we know precisely which meta pages have been
    // updated.
    //
    // We now write those pages out to the HT file.
    for changed_meta_page_ix in changed_meta_page_ixs {
        let mut page = Page::zeroed();
        page[..].copy_from_slice(meta_map.page_slice(changed_meta_page_ix));
        let pn = ht_offsets.meta_bytes_index(changed_meta_page_ix as u64);
        ht_fd.seek(SeekFrom::Start(pn * PAGE_SIZE as u64))?;
        ht_fd.write_all(&page)?;
    }

    // Finally, we collapse the WAL file.
    wal_fd.set_len(0)?;

    Ok(())
}

pub struct WriteoutData {
    /// The pages to write out to the ht file.
    pub ht_pages: Vec<(u64, Box<Page>)>,
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

    /// Get the underlying I/O handle.
    pub fn io_handle(&self) -> &IoHandle {
        &self.io_handle
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

    pub fn apply_to(self, load: &mut PageLoad) -> Option<(Box<Page>, BucketIndex)> {
        assert!(load.needs_completion());
        if self.page[PAGE_SIZE - 32..] == load.page_id.encode() {
            Some((self.page, BucketIndex(load.probe_sequence.bucket())))
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
