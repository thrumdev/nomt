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
    page_id::{PageId, PageIdsIterator, ROOT_PAGE_ID},
    trie::{self, KeyPath, LeafData, Node},
    trie_pos::{ChildNodeIndices, TriePosition},
};
use parking_lot::{Condvar, Mutex};
use std::{fmt, mem, sync::Arc};
use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;
// Within the page, we also store a bitfield indicating whether leaf data is stored at a particular
// location. This bitfield has '1' bits set for leaf data and '0' bits set for nodes.
const LEAF_DATA_BITFIELD_OFF: usize = NODES_PER_PAGE * 32;

struct PageData {
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

    fn set_leaf_data(
        &self,
        write_pass: &mut WritePass,
        children: ChildNodeIndices,
        leaf_data: LeafData,
    ) {
        let left_index = children.left();
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

    fn clear_leaf_data(&self, write_pass: &mut WritePass, children: ChildNodeIndices) {
        let left_index = children.left();
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
    inner: Arc<PageData>,
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

    /// Write leaf data at two positions under a leaf node.
    pub fn set_leaf_data(
        &self,
        write_pass: &mut WritePass,
        children: ChildNodeIndices,
        leaf_data: LeafData,
    ) {
        self.inner.set_leaf_data(write_pass, children, leaf_data);
    }

    /// Clear leaf data at two child positions.
    pub fn clear_leaf_data(&self, write_pass: &mut WritePass, children: ChildNodeIndices) {
        self.inner.clear_leaf_data(write_pass, children);
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
    /// The pages loaded from the store, possibly dirty.
    cached: DashMap<PageId, PageState, FxBuildHasher>,
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
                cached: DashMap::with_hasher(FxBuildHasher::default()),
                store,
                fetch_tp,
            }),
        }
    }

    /// Initiates retrieval of the page data at the given [`PageId`] asynchronously.
    ///
    /// If the page is already in the cache, this method does nothing. Otherwise, it fetches the
    /// page from the underlying store and caches it.
    fn prepopulate(&self, page_ids: Vec<PageId>) {
        let mut fetches = Vec::with_capacity(page_ids.len());

        for page_id in page_ids {
            if let Entry::Vacant(v) = self.shared.cached.entry(page_id.clone()) {
                // Nope, then we need to fetch the page from the store.
                let inflight = Arc::new(InflightFetch::new());
                v.insert(PageState::Inflight(inflight));
                fetches.push(page_id);
            }
        }

        if fetches.is_empty() {
            return;
        }

        let task = {
            let shared = self.shared.clone();
            move || {
                let pages = shared
                    .store
                    .load_pages_dependent(fetches.clone())
                    .expect("db load failed"); // TODO: handle the error.

                for (page_id, page) in fetches.into_iter().zip(pages) {
                    let entry = page.map_or_else(
                        || PageData::pristine_empty(&shared.page_rw_pass_domain),
                        |data| PageData::pristine_with_data(&shared.page_rw_pass_domain, data),
                    );
                    let entry = Arc::new(entry);

                    // Unwrap: the operation was inserted above. It is scheduled for execution only
                    // once. It may removed only in the line below. Therefore, `None` is impossible.
                    let mut page_state_guard = shared.cached.get_mut(&page_id).unwrap();
                    let page_state = page_state_guard.value_mut();
                    let PageState::Inflight(inflight) =
                        mem::replace(page_state, PageState::Cached(entry.clone()))
                    else {
                        panic!("page was not inflight");
                    };
                    inflight.complete_and_notify(Page { inner: entry });
                }
            }
        };

        self.shared.fetch_tp.execute(task);
    }

    /// Retrieves the page data at the given [`PageId`] synchronously.
    ///
    /// If the page is in the cache, it is returned immediately. If the page is not in the cache, it
    /// is fetched from the underlying store and returned.
    ///
    /// This method is blocking, but doesn't suffer from the channel overhead.
    pub fn retrieve_sync(&self, page_id: PageId) -> Page {
        let maybe_inflight = match self.shared.cached.entry(page_id.clone()) {
            Entry::Occupied(o) => {
                let page = o.get();
                match page {
                    PageState::Inflight(inflight) => Some(inflight.clone()),
                    PageState::Cached(page) => {
                        return Page {
                            inner: page.clone(),
                        };
                    }
                }
            }
            Entry::Vacant(v) => {
                v.insert(PageState::Inflight(Arc::new(InflightFetch::new())));
                None
            }
        };

        // do not wait with dashmap lock held; deadlock
        if let Some(existing_inflight) = maybe_inflight {
            return existing_inflight.wait();
        }

        let entry = self
            .shared
            .store
            .load_page(page_id.clone())
            .expect("db load failed") // TODO: handle the error
            .map_or_else(
                || PageData::pristine_empty(&self.shared.page_rw_pass_domain),
                |data| PageData::pristine_with_data(&self.shared.page_rw_pass_domain, data),
            );
        let entry = Arc::new(entry);
        let prev = self
            .shared
            .cached
            .insert(page_id.clone(), PageState::Cached(entry.clone()));
        match prev {
            None => {}
            Some(PageState::Inflight(inflight)) => {
                inflight.complete_and_notify(Page {
                    inner: entry.clone(),
                });
            }
            Some(PageState::Cached(_)) => {
                panic!("page was already cached");
            }
        }
        return Page { inner: entry };
    }

    pub fn new_write_cursor(&self, root: Node) -> PageCacheCursor {
        let write_pass = self.shared.page_rw_pass_domain.new_write_pass();
        PageCacheCursor::new_write(root, self.clone(), write_pass)
    }

    pub fn new_seeker(&self, root: Node) -> Seeker {
        let read_pass = self.shared.page_rw_pass_domain.new_read_pass();
        Seeker::new(root, self.clone(), read_pass)
    }

    /// Flushes all the dirty pages into the underlying store.
    ///
    /// After the commit, all the dirty pages are cleared.
    pub fn commit(&self, cursor: PageCacheCursor, tx: &mut Transaction) {
        let (dirty_pages, mut write_pass) = cursor.finish_write();
        for page_id in dirty_pages {
            // Unwrap: the invariant is that all items from `dirty` are present in the `cached` and
            // thus cannot be `None`.
            let page_state = self
                .shared
                .cached
                .get_mut(&page_id)
                .expect("a dirty page is not in the cache");
            let page_state = page_state.value();
            if let PageState::Cached(ref page) = *page_state {
                let page_data = page.data.read(write_pass.downgrade());
                let page_data = page_data.as_ref().map(|v| &v[..]);
                tx.write_page(page_id, page_data.filter(|p| !page_is_empty(p)));
            } else {
                panic!("dirty page is inflight");
            }
        }
    }
}

/// Modes for seeking to a key path.
#[derive(Debug, Clone, Copy)]
pub enum SeekMode {
    /// Retrieve the pages with the child location of any sibling nodes which are also leaves.
    ///
    /// This should be used when preparing to delete a key, which can cause leaf nodes to be
    /// relocated.
    RetrieveSiblingLeafChildren,
    /// Retrieve the pages along the path to the key path's corresponding terminal node only.
    PathOnly,
}

/// The results of a seek operation.
#[derive(Debug, Clone)]
pub struct Seek {
    /// The siblings encountered along the path, in ascending order by depth.
    ///
    /// The number of siblings is equal to the depth of the sought key.
    pub siblings: Vec<Node>,
    /// The terminal node encountered.
    pub terminal: Option<trie::LeafData>,
}

impl Seek {
    /// Get the depth of the terminal node.
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }
}

/// A [`Seeker`] can be used to seek for keys in the trie..
pub struct Seeker {
    cache: PageCache,
    read_pass: ReadPass,
    root: Node,
}

impl Seeker {
    /// Create a new Seeker, given the cache, page read pass, and a root node.
    pub fn new(root: Node, cache: PageCache, read_pass: ReadPass) -> Self {
        Seeker {
            cache,
            read_pass,
            root,
        }
    }

    fn read_leaf_children(
        &self,
        trie_pos: &TriePosition,
        current_page: Option<&(PageId, Page)>,
    ) -> trie::LeafData {
        let (page, _, children) = locate_leaf_data(trie_pos, current_page, |page_id| {
            self.cache.retrieve_sync(page_id)
        });
        trie::LeafData {
            key_path: page.node(&self.read_pass, children.left()),
            value_hash: page.node(&self.read_pass, children.right()),
        }
    }

    fn down(
        &self,
        bit: bool,
        pos: &mut TriePosition,
        cur_page: &mut Option<(PageId, Page)>,
    ) -> (Node, Node) {
        if pos.depth() as usize % DEPTH == 0 {
            // attempt to load next page if we are at the end of our previous page or the root.
            // UNWRAP: page index is valid, nodes never fall beyond the 42nd page.
            let page_id = match cur_page {
                None => ROOT_PAGE_ID,
                Some((ref id, _)) => id
                    .child_page_id(pos.child_page_index())
                    .expect("Pages do not go deeper than the maximum layer, 42"),
            };

            *cur_page = Some((page_id.clone(), self.cache.retrieve_sync(page_id)));
        }
        pos.down(bit);

        // UNWRAP: safe, was just set if at root
        let page = &cur_page.as_ref().unwrap().1;

        (
            page.node(&self.read_pass, pos.node_index()),
            page.node(&self.read_pass, pos.sibling_index()),
        )
    }

    /// Seek to the given [`KeyPath`], loading the terminal node, all siblings on the path, and caching
    /// all pages.
    ///
    /// This returns a [`Seek`] object encapsulating the results of the seek.
    pub fn seek(&self, dest: KeyPath, mode: SeekMode) -> Seek {
        /// The breadth of the prefetch request.
        ///
        /// The number of items we want to request in a single batch.
        const PREFETCH_N: usize = 7;

        let mut result = Seek {
            siblings: Vec::with_capacity(32),
            terminal: None,
        };

        let mut trie_pos = TriePosition::new();
        let mut page: Option<(PageId, Page)> = None;

        if !trie::is_internal(&self.root) {
            // fast path: don't pre-fetch when trie is just a root.
            if trie::is_leaf(&self.root) {
                result.terminal = Some(self.read_leaf_children(&trie_pos, None));
            };

            return result;
        }

        let mut ppf = PageIdsIterator::new(dest);

        let mut sibling = trie::TERMINATOR;
        let mut cur_node = self.root;

        for bit in dest.view_bits::<Msb0>().iter().by_vals() {
            if !trie::is_internal(&cur_node) {
                if trie::is_leaf(&cur_node) {
                    let leaf_data = self.read_leaf_children(&trie_pos, page.as_ref());
                    assert!(leaf_data
                        .key_path
                        .view_bits::<Msb0>()
                        .starts_with(&dest.view_bits::<Msb0>()[..trie_pos.depth() as usize]));

                    result.terminal = Some(leaf_data);
                };

                return result;
            }
            if trie_pos.depth() as usize % DEPTH == 0 {
                if trie_pos.depth() as usize % PREFETCH_N == 0 {
                    let page_ids = (&mut ppf).take(PREFETCH_N).collect();
                    self.cache.prepopulate(page_ids);
                }

                if let (&Some((ref id, _)), SeekMode::RetrieveSiblingLeafChildren, true) =
                    (&page, mode, trie::is_leaf(&sibling))
                {
                    // sibling is a leaf and at the end of the (non-root) page.
                    // initiate a load of the sibling's page.
                    let child_page_id = id
                        .child_page_id(trie_pos.sibling_child_page_index())
                        .expect("Pages do not go deeper than the maximum layer, 42");
                    // async; just warm up.
                    let _ = self.cache.prepopulate(vec![child_page_id]);
                }
            }

            let (new_node, new_sibling) = self.down(bit, &mut trie_pos, &mut page);
            cur_node = new_node;
            sibling = new_sibling;
            result.siblings.push(new_sibling);
        }

        panic!("no terminal along path {}", dest.view_bits::<Msb0>());
    }
}
