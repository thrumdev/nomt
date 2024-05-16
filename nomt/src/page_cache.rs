use crate::{
    page_region::PageRegion,
    rw_pass_cell::{ReadPass, Region, RegionContains, RwPassCell, RwPassDomain, WritePass},
    store::{Store, Transaction},
    Options,
};
use bitvec::prelude::*;
use fxhash::{FxBuildHasher, FxHashMap};
use lru::LruCache;
use nomt_core::{
    page::DEPTH,
    page_id::{ChildPageIndex, PageId, NUM_CHILDREN, ROOT_PAGE_ID},
    trie::{LeafData, Node},
    trie_pos::{ChildNodeIndices, TriePosition},
};
use parking_lot::{Condvar, Mutex, RwLock};
use std::{collections::hash_map::Entry, fmt, num::NonZeroUsize, sync::Arc};
use threadpool::ThreadPool;

#[cfg(test)]
use dashmap::DashMap;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

// Every 256 pages is 1MB.
const CACHE_PAGE_LIMIT: usize = 256 * 256;
const PAGE_LIMIT_PER_ROOT_CHILD: usize = CACHE_PAGE_LIMIT / 64;

struct PageData {
    data: RwPassCell<Option<Vec<u8>>, ShardIndex>,
}

impl PageData {
    /// Creates a page with the given data.
    fn pristine_with_data(domain: &RwPassDomain, shard_index: ShardIndex, data: Vec<u8>) -> Self {
        Self {
            data: domain.protect_with_id(Some(data), shard_index),
        }
    }

    /// Creates an empty page.
    fn pristine_empty(domain: &RwPassDomain, shard_index: ShardIndex) -> Self {
        Self {
            data: domain.protect_with_id(None, shard_index),
        }
    }

    fn node(&self, read_pass: &ReadPass<impl RegionContains<ShardIndex>>, index: usize) -> Node {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        let data = self.data.read(read_pass);
        if let Some(data) = &*data {
            let start = index * 32;
            let end = start + 32;
            let mut node = [0; 32];
            node.copy_from_slice(&data[start..end]);
            node
        } else {
            Node::default()
        }
    }

    fn set_node(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        index: usize,
        node: Node,
    ) {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);
        let start = index * 32;
        let end = start + 32;
        data[start..end].copy_from_slice(&node);
    }

    fn set_leaf_data(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        children: ChildNodeIndices,
        leaf_data: LeafData,
    ) {
        let left_index = children.left();
        assert!(left_index < NODES_PER_PAGE - 1, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);
        let start = left_index * 32;
        let end = start + 64;

        leaf_data.encode_into(&mut data[start..end]);
    }

    fn clear_leaf_data(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        children: ChildNodeIndices,
    ) {
        let left_index = children.left();
        assert!(left_index < NODES_PER_PAGE - 1, "index out of bounds");

        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);

        let start = left_index * 32;
        let l_end = start + 32;
        let r_end = l_end + 32;

        data[start..l_end].copy_from_slice(&[0u8; 32]);
        data[l_end..r_end].copy_from_slice(&[0u8; 32]);
    }
}

/// Checks whether a page is empty.
pub fn page_is_empty(page: &[u8]) -> bool {
    // 1. we assume the top layer of nodes are kept at index 0 and 1, respectively, and this
    //    is packed as the first two 32-byte slots.
    // 2. if both are empty, then the whole page is empty. this is because internal nodes
    //    with both children as terminals are not allowed to exist.
    &page[..64] == [0u8; 64].as_slice()
}

/// Tracks which nodes have changed within a page.
#[derive(Debug, Default, Clone)]
pub struct PageDiff {
    /// A bitfield indicating the number of updated slots
    updated_slots: BitArray<[u64; 2], Lsb0>,
}

impl PageDiff {
    /// Note that some 32-byte slot in the page data has changed.
    /// The acceptable range is 0..NODES_PER_PAGE
    pub fn set_changed(&mut self, slot_index: usize) {
        assert!(slot_index < NODES_PER_PAGE);
        self.updated_slots.set(slot_index, true);
    }
}

/// A handle to the page.
///
/// Can be cloned cheaply.
#[derive(Clone)]
pub struct Page {
    inner: Arc<PageData>,
}

impl Page {
    /// Read out the node at the given index.
    pub fn node(
        &self,
        read_pass: &ReadPass<impl RegionContains<ShardIndex>>,
        index: usize,
    ) -> Node {
        self.inner.node(read_pass, index)
    }

    /// Write the node at the given index.
    pub fn set_node(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        index: usize,
        node: Node,
    ) {
        self.inner.set_node(write_pass, index, node);
    }

    /// Write leaf data at two positions under a leaf node.
    pub fn set_leaf_data(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        children: ChildNodeIndices,
        leaf_data: LeafData,
    ) {
        self.inner.set_leaf_data(write_pass, children, leaf_data)
    }

    /// Clear leaf data at two child positions.
    pub fn clear_leaf_data(
        &self,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        children: ChildNodeIndices,
    ) {
        self.inner.clear_leaf_data(write_pass, children)
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page").finish()
    }
}

/// Given a trie position and a current page corresponding to that trie position (None at the root)
/// along with a function for synchronously loading a new page, get the page and indices where the
/// leaf data for a leaf at `trie_pos` should be stored.
pub fn locate_leaf_data(
    trie_pos: &TriePosition,
    current_page: Option<&(PageId, Page)>,
    load: impl Fn(PageId) -> Page,
) -> (Page, PageId, ChildNodeIndices) {
    match current_page {
        None => {
            assert!(trie_pos.is_root());
            let page = load(ROOT_PAGE_ID);
            (page, ROOT_PAGE_ID, ChildNodeIndices::from_left(0))
        }
        Some((ref page_id, ref page)) => {
            let depth_in_page = trie_pos.depth_in_page();
            if depth_in_page == DEPTH {
                let child_page_id = page_id.child_page_id(trie_pos.child_page_index()).unwrap();
                let child_page = load(child_page_id.clone());
                (child_page, child_page_id, ChildNodeIndices::from_left(0))
            } else {
                (page.clone(), page_id.clone(), trie_pos.child_node_indices())
            }
        }
    }
}

/// Represents a fetch that is currently in progress.
///
/// A fetch can be in one of the following states:
/// - Scheduled. The fetch is scheduled for execution but has not started yet.
/// - Started. The db request has been issued but still waiting for the response.
/// - Completed. The page has been fetched and the waiters are notified with the fetched page.
struct InflightFetch {
    page: Mutex<Option<Page>>,
    ready: Condvar,
}

impl InflightFetch {
    fn new() -> Self {
        Self {
            page: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    /// Notifies all the waiting parties that the page has been fetched and destroys this handle.
    fn complete_and_notify(&self, p: Page) {
        let mut page = self.page.lock();
        if page.is_some() {
            return;
        }
        *page = Some(p);
        self.ready.notify_all();
    }

    /// Waits until the page is fetched and returns it.
    fn wait(&self) -> Page {
        let mut page = self.page.lock();
        loop {
            if let Some(ref page) = &*page {
                return page.clone();
            }
            self.ready.wait(&mut page);
        }
    }
}

enum PageStore {
    Real(Store),
    #[cfg(test)]
    Mock(DashMap<PageId, Vec<u8>>),
}

impl PageStore {
    fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        match self {
            PageStore::Real(s) => s.load_page(page_id),
            #[cfg(test)]
            PageStore::Mock(map) => Ok(map.get(&page_id).map(|x| x.clone())),
        }
    }
}

// Each shard has its own domain and handles a sub-tree of the page tree, defined by a
// continuous set of children of the root page.
struct CacheShard {
    region: PageRegion,
    locked: Mutex<CacheShardLocked>,
    page_limit: NonZeroUsize,
}

struct CacheShardLocked {
    inflight: FxHashMap<PageId, Arc<InflightFetch>>,
    cached: LruCache<PageId, Arc<PageData>, FxBuildHasher>,
}

impl CacheShardLocked {
    fn evict(&mut self, limit: NonZeroUsize) {
        while self.cached.len() > limit.get() {
            let _ = self.cached.pop_lru();
        }
    }
}

struct Shared {
    shards: Vec<CacheShard>,
    root_page: RwLock<Arc<PageData>>,
    page_rw_pass_domain: RwPassDomain,
    store: PageStore,
    /// The thread pool used for fetching pages from the store.
    ///
    /// Used for limiting the number of concurrent page fetches.
    fetch_tp: ThreadPool,
}

fn shard_regions(num_shards: usize) -> Vec<(PageRegion, usize)> {
    // We apply a simple strategy that assumes keys are uniformly distributed, and give
    // each shard an approximately even number of root child pages. This scales well up to
    // 64 shards.
    // The first `remainder` shards get `part + 1` children and the rest get `part`.
    let part = NUM_CHILDREN / num_shards;
    let remainder = NUM_CHILDREN % num_shards;

    let mut regions = Vec::with_capacity(num_shards);
    for shard_index in 0..num_shards {
        let (start, count) = if shard_index >= remainder {
            (part * shard_index + remainder, part)
        } else {
            (part * shard_index + shard_index, part + 1)
        };

        // UNWRAP: start / start + count are both less than the number of children.
        let start_child = ChildPageIndex::new(start as u8).unwrap();
        let end_child = ChildPageIndex::new((start + count - 1) as u8).unwrap();
        let region = PageRegion::from_page_id_descendants(ROOT_PAGE_ID, start_child, end_child);
        regions.push((region, count));
    }

    regions
}

fn make_shards(num_shards: usize) -> Vec<CacheShard> {
    assert!(num_shards > 0);
    shard_regions(num_shards)
        .into_iter()
        .map(|(region, count)| CacheShard {
            region,
            locked: Mutex::new(CacheShardLocked {
                inflight: FxHashMap::default(),
                cached: LruCache::unbounded_with_hasher(FxBuildHasher::default()),
            }),
            // UNWRAP: both factors are non-zero
            page_limit: NonZeroUsize::new(PAGE_LIMIT_PER_ROOT_CHILD * count).unwrap(),
        })
        .collect()
}

/// The index of a shard of a page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardIndex {
    /// The "shard" containing nothing but the root page.
    Root,
    /// The shard with the given index.
    Shard(usize),
}

/// The page-cache provides an in-memory layer between the user and the underyling DB.
/// It stores full pages and can be shared between threads.
///
/// It has a sharded representation for efficient concurrent access.
#[derive(Clone)]
pub struct PageCache {
    shared: Arc<Shared>,
}

impl PageCache {
    /// Create a new `PageCache` atop the provided [`Store`].
    pub fn new(store: Store, o: &Options) -> anyhow::Result<Self> {
        let fetch_tp = threadpool::Builder::new()
            .num_threads(o.fetch_concurrency)
            .thread_name("nomt-page-fetch".to_string())
            .build();

        let domain = RwPassDomain::new();
        let root_page = store.load_page(ROOT_PAGE_ID)?.map_or_else(
            || PageData::pristine_empty(&domain, ShardIndex::Root),
            |data| PageData::pristine_with_data(&domain, ShardIndex::Root, data),
        );

        Ok(Self {
            shared: Arc::new(Shared {
                shards: make_shards(o.fetch_concurrency),
                root_page: RwLock::new(Arc::new(root_page)),
                page_rw_pass_domain: domain,
                store: PageStore::Real(store),
                fetch_tp,
            }),
        })
    }

    /// Create a new `PageCache` with a mocked store for testing.
    #[cfg(test)]
    pub fn new_mocked(o: &Options) -> Self {
        let fetch_tp = threadpool::Builder::new()
            .num_threads(o.fetch_concurrency)
            .thread_name("nomt-page-fetch".to_string())
            .build();

        let domain = RwPassDomain::new();
        let root_page = PageData::pristine_empty(&domain, ShardIndex::Root);

        Self {
            shared: Arc::new(Shared {
                shards: make_shards(o.fetch_concurrency),
                root_page: RwLock::new(Arc::new(root_page)),
                page_rw_pass_domain: domain,
                store: PageStore::Mock(DashMap::new()),
                fetch_tp,
            }),
        }
    }

    fn shard_index_for(&self, page_id: &PageId) -> Option<usize> {
        if page_id == &ROOT_PAGE_ID {
            None
        } else {
            let res = self
                .shared
                .shards
                .iter()
                .position(|s| s.region.contains_exclusive(&page_id));

            // note: this is ALWAYS `Some` because the whole page space is covered.
            assert!(res.is_some());
            res
        }
    }

    /// Initiates retrieval of the page data at the given [`PageId`] asynchronously.
    ///
    /// If the page is already in the cache, this method does nothing. Otherwise, it fetches the
    /// page from the underlying store and caches it.
    pub fn prepopulate(&self, page_id: PageId) {
        let shard_index = match self.shard_index_for(&page_id) {
            None => return, // root is always populated.
            Some(index) => index,
        };

        let mut shard = self.shard(shard_index).locked.lock();

        if !shard.cached.contains(&page_id) {
            let v = match shard.inflight.entry(page_id.clone()) {
                Entry::Vacant(v) => v,
                Entry::Occupied(_) => return, // fetch in-flight already.
            };

            // Nope, then we need to fetch the page from the store.
            let inflight = Arc::new(InflightFetch::new());
            v.insert(inflight.clone());
            let task = {
                let shared = self.shared.clone();
                move || {
                    // the page fetch has been pre-empted in the meantime. avoid querying.
                    if Arc::strong_count(&inflight) == 1 {
                        return;
                    }

                    let entry = shared
                        .store
                        .load_page(page_id.clone())
                        .expect("db load failed") // TODO: handle the error
                        .map_or_else(
                            || {
                                PageData::pristine_empty(
                                    &shared.page_rw_pass_domain,
                                    ShardIndex::Shard(shard_index),
                                )
                            },
                            |data| {
                                PageData::pristine_with_data(
                                    &shared.page_rw_pass_domain,
                                    ShardIndex::Shard(shard_index),
                                    data,
                                )
                            },
                        );
                    let entry = Arc::new(entry);

                    let mut shard = shared.shards[shard_index].locked.lock();
                    if shard.cached.contains(&page_id) {
                        // We race against pre-emption in the case other code pre-empts us by
                        // allocating an empty page.
                        return;
                    }

                    // UNWRAP: if we haven't been pre-empted, this can not have been
                    // removed.
                    let inflight = shard.inflight.remove(&page_id).unwrap();

                    shard.cached.push(page_id.clone(), entry.clone());
                    inflight.complete_and_notify(Page { inner: entry });
                }
            };
            self.shared.fetch_tp.execute(task);
        }
    }

    /// Pre-empt a previously submitted prepopulation request on a best-effort basis by returning
    /// an empty page to all waiters. This should only be called when it is known that the page
    /// will definitely not exist.
    ///
    /// This is not guaranteed to cancel the request if it is already being processed by a
    /// DB thread.
    pub fn cancel_prepopulate(&self, page_id: PageId) {
        let shard_index = match self.shard_index_for(&page_id) {
            None => return, // root always populated.
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        let page_data = Arc::new(PageData::pristine_empty(
            &self.shared.page_rw_pass_domain,
            ShardIndex::Shard(shard_index),
        ));

        let Some(inflight) = shard.inflight.remove(&page_id) else {
            return;
        };

        shard.cached.push(page_id.clone(), page_data.clone());
        inflight.complete_and_notify(Page { inner: page_data });
    }

    /// Retrieves the page data at the given [`PageId`] synchronously.
    ///
    /// If the page is in the cache, it is returned immediately. If the page is not in the cache, it
    /// is fetched from the underlying store and returned. If `hint_fresh` is true, this immediately
    /// returns a blank page.
    ///
    /// This method is blocking, but doesn't suffer from the channel overhead.
    pub fn retrieve_sync(&self, page_id: PageId, hint_fresh: bool) -> Page {
        let shard_index = match self.shard_index_for(&page_id) {
            None => {
                let page_data = self.shared.root_page.read().clone();
                return Page { inner: page_data };
            }
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        if let Some(page) = shard.cached.get(&page_id) {
            return Page {
                inner: page.clone(),
            };
        }

        let shard_mut = &mut *shard;
        let existing_inflight = if let Some(inflight) = shard_mut.inflight.get(&page_id) {
            if hint_fresh {
                // pre-empt stale fetch.

                let page_data = Arc::new(PageData::pristine_empty(
                    &self.shared.page_rw_pass_domain,
                    ShardIndex::Shard(shard_index),
                ));

                let fresh_page = Page {
                    inner: page_data.clone(),
                };

                shard_mut.cached.push(page_id.clone(), page_data.clone());
                inflight.complete_and_notify(fresh_page.clone());
                shard_mut.inflight.remove(&page_id);
                return fresh_page;
            } else {
                Some(inflight.clone())
            }
        } else if hint_fresh {
            let page_data = Arc::new(PageData::pristine_empty(
                &self.shared.page_rw_pass_domain,
                ShardIndex::Shard(shard_index),
            ));

            let fresh_page = Page {
                inner: page_data.clone(),
            };
            shard_mut.cached.push(page_id, page_data);
            return fresh_page;
        } else {
            None
        };

        // do not wait with shard lock held; deadlock
        drop(shard);

        if let Some(inflight) = existing_inflight {
            let page = inflight.wait();
            return page;
        }

        let entry = self
            .shared
            .store
            .load_page(page_id.clone())
            .expect("db load failed") // TODO: handle the error
            .map_or_else(
                || {
                    PageData::pristine_empty(
                        &self.shared.page_rw_pass_domain,
                        ShardIndex::Shard(shard_index),
                    )
                },
                |data| {
                    PageData::pristine_with_data(
                        &self.shared.page_rw_pass_domain,
                        ShardIndex::Shard(shard_index),
                        data,
                    )
                },
            );

        let entry = Arc::new(entry);

        // re-acquire lock after doing I/O
        let mut shard = self.shard(shard_index).locked.lock();

        if let Some(page) = shard.cached.peek(&page_id) {
            // pre-empted while lock wasn't held
            return Page {
                inner: page.clone(),
            };
        }

        shard.cached.push(page_id.clone(), entry.clone());
        if let Some(inflight) = shard.inflight.remove(&page_id) {
            inflight.complete_and_notify(Page {
                inner: entry.clone(),
            });
        }
        Page { inner: entry }
    }

    /// Acquire a read pass for all pages in the cache.
    pub fn new_read_pass(&self) -> ReadPass<ShardIndex> {
        self.shared
            .page_rw_pass_domain
            .new_read_pass()
            .with_region(ShardIndex::Root)
    }

    /// Acquire a write pass for all pages in the cache.
    pub fn new_write_pass(&self) -> WritePass<ShardIndex> {
        self.shared
            .page_rw_pass_domain
            .new_write_pass()
            .with_region(ShardIndex::Root)
    }

    /// Get the number of shards in this page region.
    pub fn shard_count(&self) -> usize {
        self.shared.shards.len()
    }

    /// Get the page-region associated with a shard.
    ///
    /// # Panics
    ///
    /// Panics if shard index is out of range.
    pub fn shard_region(&self, shard_index: usize) -> PageRegion {
        self.shard(shard_index).region.clone()
    }

    /// Flushes all the dirty pages into the underlying store.
    /// This takes a read pass.
    ///
    /// After the commit, all the dirty pages are cleared.
    pub fn commit(
        &self,
        page_diffs: impl IntoIterator<Item = (PageId, PageDiff)>,
        tx: &mut Transaction,
    ) {
        const FULL_PAGE_THRESHOLD: usize = 32;

        let read_pass = self.new_read_pass();
        let mut apply_page = |page_id, page_data: Option<&Vec<u8>>, page_diff: PageDiff| {
            if page_data.map_or(true, |p| page_is_empty(&p[..])) {
                tx.delete_page(page_id);
                return;
            }

            let Some(page_data) = page_data else {
                return;
            };

            let updated_count = page_diff.updated_slots.count_ones();
            if updated_count >= FULL_PAGE_THRESHOLD {
                tx.write_page(page_id, page_data);
                return;
            }

            let mut tagged_nodes = Vec::with_capacity(33 * updated_count);
            for slot_index in page_diff.updated_slots.iter_ones() {
                tagged_nodes.push(slot_index as u8);

                tagged_nodes.extend(&page_data[slot_index * 32..][..32]);
            }
            tx.write_page_nodes(page_id, tagged_nodes);
        };

        // helper for exploiting locality effects in the diffs to avoid searching through
        // shards constantly.
        let mut last_shard_index = None;
        let shard_guards = self
            .shared
            .shards
            .iter()
            .map(|s| s.locked.lock())
            .collect::<Vec<_>>();

        for (page_id, page_diff) in page_diffs {
            if page_id == ROOT_PAGE_ID {
                let page = self.shared.root_page.read();
                let page_data = page.data.read(&read_pass);
                apply_page(page_id, page_data.as_ref(), page_diff);
                continue;
            }

            // UNWRAP: all pages which are not the root page are in a shard.
            let shard_index = match last_shard_index {
                Some(i) if self.shard(i).region.contains(&page_id) => i,
                _ => self.shard_index_for(&page_id).unwrap(),
            };
            last_shard_index = Some(shard_index);

            if let Some(ref page) = shard_guards[shard_index].cached.peek(&page_id) {
                let page_data = page.data.read(&read_pass);
                apply_page(page_id, page_data.as_ref(), page_diff);
            } else {
                panic!("dirty page {:?} is missing", page_id);
            }
        }

        // purge stale pages.
        for (shard, mut guard) in self.shared.shards.iter().zip(shard_guards) {
            guard.evict(shard.page_limit);
        }
    }

    fn shard(&self, index: usize) -> &CacheShard {
        &self.shared.shards[index]
    }
}

impl Region for ShardIndex {
    fn encompasses(&self, other: &Self) -> bool {
        match (self, other) {
            (ShardIndex::Root, _) => true,
            (ShardIndex::Shard(a), ShardIndex::Shard(b)) => a == b,
            _ => false,
        }
    }

    fn excludes_unique(&self, other: &Self) -> bool {
        match (self, other) {
            (ShardIndex::Shard(a), ShardIndex::Shard(b)) => a != b,
            _ => false,
        }
    }
}

unsafe impl RegionContains<ShardIndex> for ShardIndex {
    fn contains(&self, page_shard: &ShardIndex) -> bool {
        match (self, page_shard) {
            (ShardIndex::Shard(a), ShardIndex::Shard(b)) => a == b,
            (ShardIndex::Root, _) => true,
            (ShardIndex::Shard(_), ShardIndex::Root) => {
                // safe to read the root page from any shard pass.
                true
            }
        }
    }

    fn contains_exclusive(&self, page_shard: &ShardIndex) -> bool {
        match (self, page_shard) {
            (ShardIndex::Shard(a), ShardIndex::Shard(b)) => a == b,
            (ShardIndex::Root, _) => true,
            _ => false, // exclusive access only to pages in this shard.
        }
    }
}
