use crate::{
    cursor::PageCacheCursor,
    rw_pass_cell::{ReadPass, RwPassCell, RwPassDomain, WritePass},
    store::{Store, Transaction},
    Options,
};
use dashmap::{mapref::entry::Entry, DashMap};
use fxhash::FxBuildHasher;
use nomt_core::{page::DEPTH, page_id::PageId, trie::Node};
use parking_lot::{Condvar, Mutex};
use std::{fmt, mem, sync::Arc};
use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

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
        if data.is_none() {
            *data = Some(vec![0; 4096]);
        }
        let start = index * 32;
        let end = start + 32;
        // Unwrap: self.data is Some, we just ensured it above.
        data.as_mut().unwrap()[start..end].copy_from_slice(&node);
    }
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
    pub fn prepopulate(&self, page_id: PageId) {
        if let Entry::Vacant(v) = self.shared.cached.entry(page_id) {
            // Nope, then we need to fetch the page from the store.
            let inflight = Arc::new(InflightFetch::new());
            v.insert(PageState::Inflight(inflight));
            let task = {
                let shared = self.shared.clone();
                move || {
                    let entry = shared
                        .store
                        .load_page(page_id.clone())
                        .expect("db load failed") // TODO: handle the error
                        .map_or_else(
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
            };
            self.shared.fetch_tp.execute(task);
        }
    }

    /// Retrieves the page data at the given [`PageId`] synchronously.
    ///
    /// If the page is in the cache, it is returned immediately. If the page is not in the cache, it
    /// is fetched from the underlying store and returned.
    ///
    /// This method is blocking, but doesn't suffer from the channel overhead.
    pub fn retrieve_sync(&self, page_id: PageId) -> Page {
        let maybe_inflight = match self.shared.cached.entry(page_id) {
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
