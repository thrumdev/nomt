use crate::{
    cursor::PageCacheCursor,
    rw_pass_cell::{ReadPass, RwPassCell, RwPassDomain, WritePass},
    store::{Store, Transaction},
    Options,
};
use bitvec::prelude::*;
use dashmap::{mapref::entry::Entry, DashMap};
use fxhash::FxBuildHasher;
use nomt_core::{
    page::DEPTH,
    page_id::PageId,
    trie::{LeafData, Node},
};
use parking_lot::{Condvar, Mutex, RwLock};
use std::{fmt, mem, sync::Arc};
use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;
// Within the page, we also store a bitfield indicating whether leaf data is stored at a particular
// location. This bitfield has '1' bits set for leaf data and '0' bits set for nodes.
const LEAF_DATA_BITFIELD_OFF: usize = NODES_PER_PAGE * 32;

pub struct PageData {
    data: RwPassCell<Option<Vec<u8>>>,
}

impl PageData {
    /// Creates a page with the given data.
    fn pristine_with_data(domain: &RwPassDomain, data: Vec<u8>) -> Self {
        Self {
            data: domain.protect(Some(data)),
        }
    }

    /// Creates an empty page.
    fn pristine_empty(domain: &RwPassDomain) -> Self {
        Self {
            data: domain.protect(None),
        }
    }

    /// Read out the node at the given index.
    fn node(&self, read_pass: &ReadPass, index: usize) -> Node {
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

    /// Write the node at the given index.
    fn set_node(&self, write_pass: &mut WritePass, index: usize, node: Node) {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);
        let start = index * 32;
        let end = start + 32;
        data[start..end].copy_from_slice(&node);

        // clobbering the leaf data bit here means that if a user is building a tree
        // upwards (most algorithms do), they can overwrite the leaf children and then overwrite
        // the leaf without worrying about deleting the node they've just written.
        leaf_data_bits(data).set(index, false);
    }

    fn set_leaf_data(&self, write_pass: &mut WritePass, left_index: usize, leaf_data: LeafData) {
        assert!(left_index < NODES_PER_PAGE - 1, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);
        {
            let leaf_meta = leaf_data_bits(data);
            leaf_meta.set(left_index, true);
            leaf_meta.set(left_index + 1, true);
        }
        let start = left_index * 32;
        let end = start + 64;

        leaf_data.encode_into(&mut data[start..end]);
    }

    fn clear_leaf_data(&self, write_pass: &mut WritePass, left_index: usize) {
        assert!(left_index < NODES_PER_PAGE - 1, "index out of bounds");
        let mut data = self.data.write(write_pass);
        let data = data.get_or_insert_with(|| vec![0; 4096]);
        let (overwrite_l, overwrite_r) = {
            let leaf_meta = leaf_data_bits(data);
            (
                leaf_meta.replace(left_index, false),
                leaf_meta.replace(left_index + 1, false),
            )
        };

        let start = left_index * 32;
        let l_end = start + 32;
        let r_end = l_end + 32;

        if overwrite_l {
            data[start..l_end].copy_from_slice(&[0u8; 32]);
        }
        if overwrite_r {
            data[l_end..r_end].copy_from_slice(&[0u8; 32]);
        }
    }
}

fn leaf_data_bits(page: &mut [u8]) -> &mut BitSlice<u8, Msb0> {
    page[LEAF_DATA_BITFIELD_OFF..][..32].view_bits_mut::<Msb0>()
}

fn page_is_empty(page: &[u8]) -> bool {
    // 1. we assume the top layer of nodes are kept at index 0 and 1, respectively, and this
    //    is packed as the first two 32-byte slots.
    // 2. if both are empty, then the whole page is empty. this is because internal nodes
    //    with both children as terminals are not allowed to exist.
    &page[..64] == [0u8; 64].as_slice()
}

enum PageState {
    Inflight(Arc<InflightFetch>),
    Cached(Arc<PageData>),
}

/// A handle to the page.
///
/// Can be cloned cheaply.
#[derive(Clone)]
pub struct Page {
    pub(crate) inner: Arc<PageData>,
}

impl Page {
    /// Read out the node at the given index.
    pub fn node(&self, read_pass: &ReadPass, index: usize) -> Node {
        self.inner.node(read_pass, index)
    }

    /// Write the node at the given index.
    pub fn set_node(&self, write_pass: &mut WritePass, index: usize, node: Node) {
        self.inner.set_node(write_pass, index, node)
    }

    /// Write leaf data at two positions under a leaf node, `left_index` and `left_index + 1`.
    pub fn set_leaf_data(
        &self,
        write_pass: &mut WritePass,
        left_index: usize,
        leaf_data: LeafData,
    ) {
        self.inner.set_leaf_data(write_pass, left_index, leaf_data);
    }

    /// Clear leaf data at two positions under a leaf node, `left_index` and `left_index + 1`.
    pub fn clear_leaf_data(&self, write_pass: &mut WritePass, left_index: usize) {
        self.inner.clear_leaf_data(write_pass, left_index);
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page").finish()
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

slotmap::new_key_type! {
    struct CacheSlotIndex;
}

struct CacheSlot {
    inner: Arc<Mutex<PageState>>,
    // next: CacheSlotIndex,
}

/// The page-cache provides an in-memory layer between the user and the underlying DB.
/// It stores full pages and can be shared between threads.
#[derive(Clone)]
pub struct PageCache {
    shared: Arc<Shared>,
}

struct Shared {
    page_rw_pass_domain: RwPassDomain,
    store: Store,
    /// The thread pool used for fetching pages from the store.
    ///
    /// Used for limiting the number of concurrent page fetches.
    fetch_tp: ThreadPool,

    slot_registry: DashMap<PageId, CacheSlotIndex, FxBuildHasher>,
    /// The pages loaded from the store, possibly dirty.
    cache_slots: RwLock<slotmap::SlotMap<CacheSlotIndex, CacheSlot>>,
    max_sz: usize,
}

impl PageCache {
    /// Create a new `PageCache` atop the provided [`Store`].
    pub fn new(store: Store, o: &Options) -> Self {
        let fetch_tp = threadpool::Builder::new()
            .num_threads(o.fetch_concurrency)
            .thread_name("nomt-page-fetch".to_string())
            .build();
        Self {
            shared: Arc::new(Shared {
                page_rw_pass_domain: RwPassDomain::new(),
                slot_registry: DashMap::with_hasher(FxBuildHasher::default()),
                cache_slots: RwLock::new(slotmap::SlotMap::with_capacity_and_key(
                    o.page_cache_size,
                )),
                store,
                fetch_tp,
                max_sz: o.page_cache_size,
            }),
        }
    }

    /// Initiates retrieval of the page data at the given [`PageId`] asynchronously.
    ///
    /// If the page is already in the cache, this method does nothing. Otherwise, it fetches the
    /// page from the underlying store and caches it.
    pub fn prepopulate(&self, page_id: PageId) {
        let _ = self.begin_load_page(page_id, /* sync */ false);
    }

    /// Retrieves the page data at the given [`PageId`] synchronously.
    ///
    /// If the page is in the cache, it is returned immediately. If the page is not in the cache, it
    /// is fetched from the underlying store and returned.
    ///
    /// This method is blocking, but doesn't suffer from the channel overhead.
    pub fn retrieve_sync(&self, page_id: PageId) -> Page {
        self.begin_load_page(page_id, /* sync */ true).unwrap()
    }

    /// TODO:
    /// If `block` is true, then `Page` is guaranteed to be `Some`.`
    fn begin_load_page(&self, page_id: PageId, block: bool) -> Option<Page> {
        // On a high-level, we need to do the following:
        //
        // 1. Insert a new `InflightFetch` into the `cache_slot[slot_ix]`, where
        //    `slot_ix = slot_registry[page_id]`, if not already present. If it is present, then
        //    wait on it and return the page if `block` is `true` or `None` if `block` is `false`.
        //
        // 1. Call `load_page_sync` to load the page from the store and cache it. Depending on the
        //    value of `block`, do it either synchronously or submit a task to the thread pool.
        //
        // 2. If `block` is `true`, return the page. Otherwise, return `None`.
        //
        // This is complicated due to the fact that slot_registry is separate from cache_slots, so
        // we need to do some gymnastics to ensure that we don't lock excessively and deadlock.

        let (slot_ix_guard, known_page_id) = {
            let this = self.shared.slot_registry.entry(page_id.clone());
            match this {
                Entry::Occupied(entry) => (entry.into_ref(), true),
                Entry::Vacant(entry) => {
                    let slot_ix = self.shared.cache_slots.write().insert(CacheSlot {
                        inner: Arc::new(Mutex::new(PageState::Inflight(Arc::new(
                            InflightFetch::new(),
                        )))),
                        // TODO: next = back, back = prev
                        // next: todo!(),
                    });
                    (entry.insert(slot_ix), false)
                }
            }
        };

        if known_page_id {
            let slot_ix = *slot_ix_guard;
            // Unwrap is justified because slot_ix is ensured to be present above.
            let cache_slots = self.shared.cache_slots.read();
            let slot = cache_slots.get(slot_ix).unwrap();

            let page_state = slot.inner.lock();
            return match &*page_state {
                PageState::Inflight(inflight) => {
                    if block {
                        // Take out the inflight, drop the locks and only then wait.
                        let inflight = inflight.clone();
                        drop(page_state);
                        drop(cache_slots);
                        drop(slot_ix_guard);
                        Some(inflight.wait())
                    } else {
                        None
                    }
                }
                PageState::Cached(page_data) => Some(Page {
                    inner: page_data.clone(),
                }),
            };
        }

        // The page is not known, so we need to load it from the store.
        let slot_ix = *slot_ix_guard;
        drop(slot_ix_guard);

        if block {
            let page_data = Self::load_page_sync(&self.shared, page_id, slot_ix);
            Some(Page { inner: page_data })
        } else {
            let shared = self.shared.clone();
            let page_id = page_id.clone();
            self.shared.fetch_tp.execute(move || {
                let _ = Self::load_page_sync(&shared, page_id, slot_ix);
            });
            None
        }
    }

    // Must not be called while holding a mutable lock to `shared.cached`.
    fn load_page_sync(shared: &Shared, page_id: PageId, slot_ix: CacheSlotIndex) -> Arc<PageData> {
        let entry = shared
            .store
            .load_page(page_id.clone())
            .expect("db load failed") // TODO: handle the error
            .map_or_else(
                || PageData::pristine_empty(&shared.page_rw_pass_domain),
                |data| PageData::pristine_with_data(&shared.page_rw_pass_domain, data),
            );
        let entry = Arc::new(entry);

        let mut cache_slots = shared.cache_slots.write();
        if cache_slots.len() >= shared.max_sz {
            // TODO: evict a page
        }

        // TODO: next = back, back = prev
        let slot = cache_slots.get_mut(slot_ix).unwrap();
        let new = CacheSlot {
            inner: Arc::new(Mutex::new(PageState::Cached(entry.clone()))),
            // next: slot.next,
        };
        let prev = mem::replace(slot, new);
        drop(cache_slots);

        match &*prev.inner.lock() {
            PageState::Inflight(inflight) => {
                inflight.complete_and_notify(Page {
                    inner: entry.clone(),
                });
            }
            PageState::Cached(_) => {
                panic!("page was already cached");
            }
        }
        entry
    }

    pub fn new_read_cursor(&self, root: Node) -> PageCacheCursor {
        let read_pass = self.shared.page_rw_pass_domain.new_read_pass();
        PageCacheCursor::new_read(root, self.clone(), read_pass)
    }

    pub fn new_write_cursor(&self, root: Node) -> PageCacheCursor {
        let write_pass = self.shared.page_rw_pass_domain.new_write_pass();
        PageCacheCursor::new_write(root, self.clone(), write_pass)
    }

    /// Flushes all the dirty pages into the underlying store.
    ///
    /// After the commit, all the dirty pages are cleared.
    pub fn commit(&self, cursor: PageCacheCursor, tx: &mut Transaction) {
        let (dirty_pages, mut write_pass) = cursor.finish_write();
        for (page, page_id) in dirty_pages.dirtied {
            let page_data = page.0.inner.data.read(write_pass.downgrade());
            let page_data = page_data.as_ref().map(|v| &v[..]);
            tx.write_page(page_id, page_data.filter(|p| !page_is_empty(p)));
        }
    }
}
