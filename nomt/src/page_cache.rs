use crate::{
    bitbox::BucketIndex,
    io::{page_pool::FatPage, PagePool},
    metrics::{Metric, Metrics},
    page_diff::PageDiff,
    page_region::PageRegion,
    rw_pass_cell::{ReadPass, Region, RegionContains, RwPassCell, RwPassDomain, WritePass},
    store::MerkleTransaction,
    Options,
};
use fxhash::FxBuildHasher;
use lru::LruCache;
use nomt_core::{
    page::DEPTH,
    page_id::{ChildPageIndex, PageId, NUM_CHILDREN, ROOT_PAGE_ID},
    trie::Node,
};
use parking_lot::{Mutex, RwLock};
use std::{fmt, num::NonZeroUsize, sync::Arc};

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

// Every 256 pages is 1MB.
const CACHE_PAGE_LIMIT: usize = 256 * 256;
const PAGE_LIMIT_PER_ROOT_CHILD: usize = CACHE_PAGE_LIMIT / 64;

struct PageData {
    data: RwPassCell<Option<FatPage>, ShardIndex>,
}

impl PageData {
    /// Creates a page with the given data.
    fn pristine_with_data(domain: &RwPassDomain, shard_index: ShardIndex, data: FatPage) -> Self {
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
        page_pool: &PagePool,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        index: usize,
        node: Node,
    ) {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| page_pool.alloc_fat_page());
        let start = index * 32;
        let end = start + 32;
        data[start..end].copy_from_slice(&node);
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
        page_pool: &PagePool,
        write_pass: &mut WritePass<impl RegionContains<ShardIndex>>,
        index: usize,
        node: Node,
    ) {
        self.inner.set_node(page_pool, write_pass, index, node);
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page").finish()
    }
}

struct CacheEntry {
    page_data: Arc<PageData>,
    // the bucket index where this page is stored. `None` if it's a fresh page.
    bucket_index: Option<BucketIndex>,
}

impl CacheEntry {
    fn init(
        domain: &RwPassDomain,
        shard_index: ShardIndex,
        maybe_page: Option<(FatPage, BucketIndex)>,
    ) -> Self {
        match maybe_page {
            Some((data, bucket_index)) => CacheEntry {
                page_data: Arc::new(PageData::pristine_with_data(domain, shard_index, data)),
                bucket_index: Some(bucket_index),
            },
            None => CacheEntry {
                page_data: Arc::new(PageData::pristine_empty(domain, shard_index)),
                bucket_index: None,
            },
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
    cached: LruCache<PageId, CacheEntry, FxBuildHasher>,
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
    root_page: RwLock<CacheEntry>,
    page_rw_pass_domain: RwPassDomain,
    metrics: Metrics,
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

// returns the shard index, according to the allocation in `shard_regions`.
fn shard_index_for(num_shards: usize, first_ancestor: usize) -> usize {
    let part = NUM_CHILDREN / num_shards;
    let remainder = NUM_CHILDREN % num_shards;

    if (part + 1) * remainder > first_ancestor {
        // in a 'remainder' shard
        first_ancestor / (part + 1)
    } else {
        // in a non-remainder shard - take all the remainder shards out, then divide by part,
        // and add back the remainder to get the final index
        ((first_ancestor - (part + 1) * remainder) / part) + remainder
    }
}

fn make_shards(num_shards: usize) -> Vec<CacheShard> {
    assert!(num_shards > 0);
    shard_regions(num_shards)
        .into_iter()
        .map(|(region, count)| CacheShard {
            region,
            locked: Mutex::new(CacheShardLocked {
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

/// The page-cache stores full pages and can be shared between threads.
///
/// It has a sharded representation for efficient concurrent access.
#[derive(Clone)]
pub struct PageCache {
    shared: Arc<Shared>,
}

impl PageCache {
    /// Create a new `PageCache`.
    pub fn new(
        root_page_data: Option<(FatPage, BucketIndex)>,
        o: &Options,
        metrics: impl Into<Option<Metrics>>,
    ) -> Self {
        let domain = RwPassDomain::new();
        Self {
            shared: Arc::new(Shared {
                shards: make_shards(o.commit_concurrency),
                root_page: RwLock::new(CacheEntry::init(&domain, ShardIndex::Root, root_page_data)),
                page_rw_pass_domain: domain,
                metrics: metrics.into().unwrap_or(Metrics::new(false)),
            }),
        }
    }

    fn shard_index_for(&self, page_id: &PageId) -> Option<usize> {
        if page_id == &ROOT_PAGE_ID {
            None
        } else {
            let first_ancestor = page_id.child_index_at_level(0).to_u8() as usize;
            let shard_index = shard_index_for(self.shared.shards.len(), first_ancestor);
            debug_assert!(self.shared.shards[shard_index]
                .region
                .contains_exclusive(page_id));
            Some(shard_index)
        }
    }

    /// Query the cache for the page data at the given [`PageId`].
    ///
    /// Returns `None` if not in the cache.
    pub fn get(&self, page_id: PageId) -> Option<Page> {
        self.shared.metrics.count(Metric::PageRequests);
        let shard_index = match self.shard_index_for(&page_id) {
            None => {
                let page_data = self.shared.root_page.read().page_data.clone();
                return Some(Page { inner: page_data });
            }
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        match shard.cached.get(&page_id) {
            Some(page) => Some(Page {
                inner: page.page_data.clone(),
            }),
            None => {
                self.shared.metrics.count(Metric::PageCacheMisses);
                None
            }
        }
    }

    /// Insert a page into the cache by its data. If `Some`, provide the bucket index where the
    /// page is stored.
    ///
    /// This ignores the inputs if the page was already present, and returns that.
    pub fn insert(&self, page_id: PageId, page: Option<(FatPage, BucketIndex)>) -> Page {
        let domain = &self.shared.page_rw_pass_domain;
        let shard_index = match self.shard_index_for(&page_id) {
            None => {
                let page_data = self.shared.root_page.read().page_data.clone();
                return Page { inner: page_data };
            }
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        let cache_entry = shard.cached.get_or_insert(page_id, || {
            CacheEntry::init(domain, ShardIndex::Shard(shard_index), page)
        });

        Page {
            inner: cache_entry.page_data.clone(),
        }
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

    /// Get the shard with the given index. This can quickly answer queries directed at a particular
    /// shard of the page cache, avoiding the overhead of determining which shard to use.
    pub fn get_shard(&self, shard_index: usize) -> PageCacheShard {
        PageCacheShard {
            shared: self.shared.clone(),
            shard_index,
        }
    }

    /// Prepares a transaction of altered pages, according to the provided page diffs.
    /// This takes a read pass.
    pub fn prepare_transaction(
        &self,
        page_diffs: impl IntoIterator<Item = (PageId, PageDiff)>,
        tx: &mut MerkleTransaction,
    ) {
        let read_pass = self.new_read_pass();
        let mut apply_page = |page_id,
                              bucket: &mut Option<BucketIndex>,
                              page_data: Option<&FatPage>,
                              page_diff: PageDiff| {
            match (page_data, *bucket) {
                (None, Some(known_bucket)) => {
                    tx.delete_page(page_id, known_bucket);
                    *bucket = None;
                }
                (Some(_), Some(known_bucket)) if page_diff.cleared() => {
                    tx.delete_page(page_id, known_bucket);
                    *bucket = None;
                }
                (Some(page), maybe_bucket) if !page_diff.cleared() => {
                    let new_bucket = tx.write_page(page_id, maybe_bucket, page, page_diff);
                    *bucket = Some(new_bucket);
                }
                _ => {} // empty pages which had no known bucket. don't write or delete.
            }
        };

        // helper for exploiting locality effects in the diffs to avoid searching through
        // shards constantly.
        let mut shard_guards = self
            .shared
            .shards
            .iter()
            .map(|s| s.locked.lock())
            .collect::<Vec<_>>();

        for (page_id, page_diff) in page_diffs {
            if page_id == ROOT_PAGE_ID {
                let mut root_page = self.shared.root_page.write();
                let root_page = &mut *root_page;
                let page_data = root_page.page_data.data.read(&read_pass);
                let bucket = &mut root_page.bucket_index;
                apply_page(page_id, bucket, page_data.as_ref(), page_diff);
                continue;
            }

            // UNWRAP: all pages which are not the root page are in a shard.
            let shard_index = self.shard_index_for(&page_id).unwrap();

            if let Some(ref mut entry) = shard_guards[shard_index].cached.peek_mut(&page_id) {
                let page_data = entry.page_data.data.read(&read_pass);
                let bucket = &mut entry.bucket_index;
                apply_page(page_id, bucket, page_data.as_ref(), page_diff);
            } else {
                panic!("dirty page {:?} is missing", page_id);
            }
        }
    }

    /// Evict stale pages for the cache. This should only be used after all dirty pages have been
    /// prepared for writeout with `prepare_transaction`.
    pub fn evict(&self) {
        let shard_guards = self
            .shared
            .shards
            .iter()
            .map(|s| s.locked.lock())
            .collect::<Vec<_>>();

        for (shard, mut guard) in self.shared.shards.iter().zip(shard_guards) {
            guard.evict(shard.page_limit);
        }
    }

    fn shard(&self, index: usize) -> &CacheShard {
        &self.shared.shards[index]
    }
}

/// A shard of the page cache. This should only be used for pages which fall within
/// that shard, with the exception of the root page, which is accessible via this shard.
pub struct PageCacheShard {
    shared: Arc<Shared>,
    shard_index: usize,
}

impl PageCacheShard {
    /// Query the cache for the page data at the given [`PageId`].
    ///
    /// Returns `None` if not in the cache.
    pub fn get(&self, page_id: PageId) -> Option<Page> {
        self.shared.metrics.count(Metric::PageRequests);
        if page_id == ROOT_PAGE_ID {
            let page_data = self.shared.root_page.read().page_data.clone();
            return Some(Page { inner: page_data });
        }

        debug_assert!(self.shared.shards[self.shard_index]
            .region
            .contains_exclusive(&page_id));

        let mut shard = self.shared.shards[self.shard_index].locked.lock();
        match shard.cached.get(&page_id) {
            Some(page) => Some(Page {
                inner: page.page_data.clone(),
            }),
            None => {
                self.shared.metrics.count(Metric::PageCacheMisses);
                None
            }
        }
    }

    /// Insert a page into the cache by its data. If `Some`, provide the bucket index where the
    /// page is stored.
    ///
    /// This ignores the inputs if the page was already present, and returns that.
    pub fn insert(&self, page_id: PageId, page: Option<(FatPage, BucketIndex)>) -> Page {
        let domain = &self.shared.page_rw_pass_domain;
        if page_id == ROOT_PAGE_ID {
            let page_data = self.shared.root_page.read().page_data.clone();
            return Page { inner: page_data };
        }

        debug_assert!(self.shared.shards[self.shard_index]
            .region
            .contains_exclusive(&page_id));

        let mut shard = self.shared.shards[self.shard_index].locked.lock();
        let cache_entry = shard.cached.get_or_insert(page_id, || {
            CacheEntry::init(domain, ShardIndex::Shard(self.shard_index), page)
        });

        Page {
            inner: cache_entry.page_data.clone(),
        }
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
