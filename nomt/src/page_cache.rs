use crate::{
    bitbox::BucketIndex,
    io::{page_pool::FatPage, PagePool, PAGE_SIZE},
    merkle::UpdatedPage,
    metrics::{Metric, Metrics},
    page_diff::PageDiff,
    page_region::PageRegion,
    rw_pass_cell::{Region, RegionContains, RwPassDomain, WritePass},
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

fn read_node(data: &FatPage, index: usize) -> Node {
    assert!(index < NODES_PER_PAGE, "index out of bounds");
    let start = index * 32;
    let end = start + 32;
    let mut node = [0; 32];
    node.copy_from_slice(&data[start..end]);
    node
}

fn set_node(data: &mut FatPage, index: usize, node: Node) {
    assert!(index < NODES_PER_PAGE, "index out of bounds");
    let start = index * 32;
    let end = start + 32;
    data[start..end].copy_from_slice(&node);
}

/// A mutable page.
pub struct PageMut {
    inner: Option<FatPage>,
}

impl PageMut {
    /// Freeze the page.
    pub fn freeze(self) -> Page {
        Page {
            inner: self.inner.map(Arc::new),
        }
    }

    /// Create a pristine (i.e. blank) `PageMut`.
    pub fn pristine_empty() -> PageMut {
        PageMut { inner: None }
    }

    /// Create a mutable page from raw page data.
    pub fn pristine_with_data(data: FatPage) -> PageMut {
        PageMut { inner: Some(data) }
    }

    /// Read out the node at the given index.
    pub fn node(&self, index: usize) -> Node {
        self.inner
            .as_ref()
            .map(|d| read_node(d, index))
            .unwrap_or_default()
    }

    /// Write the node at the given index.
    pub fn set_node(&mut self, page_pool: &PagePool, index: usize, node: Node) {
        let data = self.inner.get_or_insert_with(|| page_pool.alloc_fat_page());

        set_node(data, index, node);
    }

    /// Set the page ID metadata within the raw buffer, if there is one. No-op if this is a pristine
    /// empty page.
    fn set_page_id(&mut self, page_id: &PageId) {
        if let Some(ref mut data) = self.inner {
            data[PAGE_SIZE - 32..].copy_from_slice(&page_id.encode());
        }
    }
}

impl From<FatPage> for PageMut {
    fn from(data: FatPage) -> PageMut {
        Self::pristine_with_data(data)
    }
}

/// A handle to the page.
///
/// Can be cloned cheaply.
#[derive(Clone)]
pub struct Page {
    inner: Option<Arc<FatPage>>,
}

impl Page {
    /// Read out the node at the given index.
    pub fn node(&self, index: usize) -> Node {
        self.inner
            .as_ref()
            .map(|d| read_node(d, index))
            .unwrap_or_default()
    }

    /// Create a mutable deep copy of this page.
    pub fn deep_copy(&self) -> PageMut {
        PageMut {
            inner: self.inner.as_ref().map(|x| FatPage::clone(x)),
        }
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page").finish()
    }
}

struct CacheEntry {
    page_data: Option<Arc<FatPage>>,
    // the bucket index where this page is stored. `None` if it's a fresh page.
    bucket_index: Option<BucketIndex>,
}

impl CacheEntry {
    fn init(page_data: Option<FatPage>, bucket_index: Option<BucketIndex>) -> Self {
        CacheEntry {
            page_data: page_data.map(Arc::new),
            bucket_index,
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

fn make_shards(num_shards: usize, page_cache_size: usize) -> Vec<CacheShard> {
    // page_cache_size is measured in MiB
    let cache_page_limit = (page_cache_size * 1024 * 1024) / PAGE_SIZE;
    let page_limit_per_root_child = cache_page_limit / 64;

    assert!(num_shards > 0);
    shard_regions(num_shards)
        .into_iter()
        .map(|(region, count)| CacheShard {
            region,
            locked: Mutex::new(CacheShardLocked {
                cached: LruCache::unbounded_with_hasher(FxBuildHasher::default()),
            }),
            // UNWRAP: both factors are non-zero
            page_limit: NonZeroUsize::new(page_limit_per_root_child * count).unwrap(),
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

        let (root_page, root_page_bucket) = if let Some((page, bucket)) = root_page_data {
            (Some(page), Some(bucket))
        } else {
            (None, None)
        };

        Self {
            shared: Arc::new(Shared {
                shards: make_shards(o.commit_concurrency, o.page_cache_size),
                root_page: RwLock::new(CacheEntry::init(root_page, root_page_bucket)),
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

    /// Insert a page into the cache by its data. Provide the bucket index where the
    /// page is stored if this was loaded from the disk.
    ///
    /// This ignores the inputs if the page was already present, and returns that.
    pub fn insert(
        &self,
        page_id: PageId,
        page: PageMut,
        bucket_index: Option<BucketIndex>,
    ) -> Page {
        let shard_index = match self.shard_index_for(&page_id) {
            None => {
                let page_data = self.shared.root_page.read().page_data.clone();
                return Page { inner: page_data };
            }
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        let cache_entry = shard
            .cached
            .get_or_insert(page_id, || CacheEntry::init(page.inner, bucket_index));

        Page {
            inner: cache_entry.page_data.clone(),
        }
    }

    /// Insert a page into the cache by its data. Provide the bucket index where the page is
    /// stored if this was loaded from the disk.
    ///
    /// This overwrites any page that was already present in the cache.
    #[cfg(test)]
    pub fn insert_overwriting(
        &self,
        page_id: PageId,
        page: PageMut,
        bucket_index: Option<BucketIndex>,
    ) {
        let shard_index = match self.shard_index_for(&page_id) {
            None => {
                self.shared.root_page.write().page_data = page.freeze().inner;
                return;
            }
            Some(i) => i,
        };

        let mut shard = self.shard(shard_index).locked.lock();
        shard
            .cached
            .put(page_id, CacheEntry::init(page.inner, bucket_index));
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

    /// Absorb a set of altered pages into the cache and populates the given disk transaction.
    ///
    /// This returns a set of outdated pages which can be dropped outside of the critical path.
    pub fn absorb_and_populate_transaction(
        &self,
        updated_pages: impl IntoIterator<Item = UpdatedPage>,
        tx: &mut MerkleTransaction,
    ) -> Vec<Option<Arc<FatPage>>> {
        let mut apply_page = |page_id,
                              bucket: &mut Option<BucketIndex>,
                              page_data: Option<&Arc<FatPage>>,
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
                    let new_bucket = tx.write_page(page_id, maybe_bucket, page.clone(), page_diff);
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

        let mut deferred_drop = Vec::new();

        for mut updated_page in updated_pages {
            // Pages must store their ID before being written out. Set it here.
            updated_page.page.set_page_id(&updated_page.page_id);

            if updated_page.page_id == ROOT_PAGE_ID {
                let mut root_page = self.shared.root_page.write();
                let root_page = &mut *root_page;
                // update the cached page data.
                let old =
                    std::mem::replace(&mut root_page.page_data, updated_page.page.freeze().inner);
                deferred_drop.push(old);

                let page_data = &root_page.page_data;
                let bucket = &mut root_page.bucket_index;
                apply_page(
                    updated_page.page_id,
                    bucket,
                    page_data.as_ref(),
                    updated_page.diff,
                );
                continue;
            }

            // UNWRAP: all pages which are not the root page are in a shard.
            let shard_index = self.shard_index_for(&updated_page.page_id).unwrap();

            let cache_entry = shard_guards[shard_index]
                .cached
                .get_or_insert_mut(updated_page.page_id.clone(), || {
                    CacheEntry::init(None, None)
                });

            let old =
                std::mem::replace(&mut cache_entry.page_data, updated_page.page.freeze().inner);
            deferred_drop.push(old);

            let bucket = &mut cache_entry.bucket_index;
            apply_page(
                updated_page.page_id,
                bucket,
                cache_entry.page_data.as_ref(),
                updated_page.diff,
            );
        }

        deferred_drop
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
    #[cfg(test)]
    pub fn insert(
        &self,
        page_id: PageId,
        page: PageMut,
        bucket_index: Option<BucketIndex>,
    ) -> Page {
        if page_id == ROOT_PAGE_ID {
            let page_data = self.shared.root_page.read().page_data.clone();
            return Page { inner: page_data };
        }

        debug_assert!(self.shared.shards[self.shard_index]
            .region
            .contains_exclusive(&page_id));

        let mut shard = self.shared.shards[self.shard_index].locked.lock();
        let cache_entry = shard
            .cached
            .get_or_insert(page_id, || CacheEntry::init(page.inner, bucket_index));

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
