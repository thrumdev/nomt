use crate::{
    store::{Store, Transaction},
    Options,
};
use nomt_core::{page::DEPTH, page_id::PageId, trie::Node};
use parking_lot::Mutex;
use std::{
    collections::{HashMap, HashSet},
    fmt, mem,
    sync::Arc,
};
use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

struct PageData {
    data: Option<Vec<u8>>,
    dirty: bool,
}

impl PageData {
    /// Creates a page with the given data.
    fn pristine_with_data(data: Vec<u8>) -> Self {
        Self {
            data: Some(data),
            dirty: false,
        }
    }

    /// Creates an empty page.
    fn pristine_empty() -> Self {
        Self {
            data: None,
            dirty: false,
        }
    }

    /// Read out the node at the given index.
    fn node(&self, index: usize) -> Node {
        assert!(index < NODES_PER_PAGE, "index out of bounds");

        if let Some(data) = &self.data {
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
    fn set_node(&mut self, index: usize, node: Node) {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        if self.data.is_none() {
            self.data = Some(vec![0; 4096]);
        }
        let start = index * 32;
        let end = start + 32;
        // Unwrap: self.data is Some, we just ensured it above.
        self.data.as_mut().unwrap()[start..end].copy_from_slice(&node);
        self.dirty = true;
    }
}

/// A handle to the page.
///
/// Can be cloned cheaply.
#[derive(Clone)]
pub struct Page {
    inner: Arc<Mutex<PageData>>,
}

impl Page {
    /// Read out the node at the given index.
    pub fn node(&self, index: usize) -> Node {
        self.inner.lock().node(index)
    }

    /// Write the node at the given index.
    pub fn set_node(&self, index: usize, node: Node) {
        self.inner.lock().set_node(index, node);
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Page").finish()
    }
}

pub enum PagePromise {
    Now(Page),
    Later(crossbeam_channel::Receiver<Page>),
}

impl PagePromise {
    /// Waits for the page to be fetched.
    pub fn wait(self) -> Page {
        match self {
            PagePromise::Now(page) => page,
            PagePromise::Later(rx) => {
                // TODO: this unwrap must be removed, it's edge-case. Specifically, during the
                //       shutdown it may happen that the sender is dropped before the receiver.
                rx.recv().unwrap()
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
    tx: Option<crossbeam_channel::Sender<Page>>,
    rx: crossbeam_channel::Receiver<Page>,
}

impl InflightFetch {
    fn new() -> Self {
        let (tx, rx) = crossbeam_channel::bounded(1);
        Self { tx: Some(tx), rx }
    }

    /// Notifies all the waiting parties that the page has been fetched and destroys this handle.
    fn complete_and_notify(mut self, page: Page) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(page);
        }
    }

    /// Returns the promise that resolves when the page is fetched.
    fn promise(&self) -> PagePromise {
        PagePromise::Later(self.rx.clone())
    }
}

/// The page-cache provides an in-memory layer between the user and the underlying DB.
/// It stores full pages and can be shared between threads.
#[derive(Clone)]
pub struct PageCache {
    shared: Arc<Mutex<Shared>>,
    store: Store,
    /// The thread pool used for fetching pages from the store.
    ///
    /// Used for limiting the number of concurrent page fetches.
    fetch_tp: ThreadPool,
}

struct Shared {
    /// The pages loaded from the store, possibly dirty.
    cached: HashMap<PageId, Page>,
    /// The pages that were modified and need to be flushed to the store.
    ///
    /// The invariant is that the pages in the dirty set are always present in the `cached` map and
    /// are marked as dirty there.
    dirty: HashSet<PageId>,
    /// Pages that are currently being fetched from the store.
    inflight: HashMap<PageId, InflightFetch>,
}

impl PageCache {
    /// Create a new `PageCache` atop the provided [`Store`].
    pub fn new(store: Store, o: &Options) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            cached: HashMap::new(),
            dirty: HashSet::new(),
            inflight: HashMap::new(),
        }));
        let fetch_tp = threadpool::Builder::new()
            .num_threads(o.fetch_concurrency)
            .thread_name("nomt-page-fetch".to_string())
            .build();
        Self {
            shared,
            store,
            fetch_tp,
        }
    }

    /// Retrieves the page data at the given [`PageId`] asynchronously.
    ///
    /// If the page is in the cache, it is returned immediately. If the page is not in the cache, it
    /// is fetched from the underlying store and returned.
    pub fn retrieve(&self, page_id: PageId) -> PagePromise {
        let mut shared = self.shared.lock();

        // If the page is not present there, we check in the pristine cache.
        if let Some(page) = shared.cached.get(&page_id) {
            return PagePromise::Now(page.clone());
        }
        // In case none of those are present, we check if there is an inflight fetch for the page.
        if let Some(inflight) = shared.inflight.get(&page_id) {
            return inflight.promise();
        }
        // Nope, then we need to fetch the page from the store.
        let inflight = InflightFetch::new();
        let promise = inflight.promise();
        shared.inflight.insert(page_id.clone(), inflight);
        let task = {
            let store = self.store.clone();
            let shared = self.shared.clone();
            move || {
                let entry = store
                    .load_page(page_id.clone())
                    .expect("db load failed") // TODO: handle the error
                    .map_or_else(PageData::pristine_empty, PageData::pristine_with_data);
                let entry = Page {
                    inner: Arc::new(Mutex::new(entry)),
                };

                let mut shared = shared.lock();
                // Unwrap: the operation was inserted above. It is scheduled for execution only
                // once. It may removed only in the line below. Therefore, `None` is impossible.
                let inflight = shared.inflight.remove(&page_id).unwrap();
                shared.cached.insert(page_id, entry.clone());
                inflight.complete_and_notify(entry);
                drop(shared);
            }
        };
        self.fetch_tp.execute(task);
        promise
    }

    /// Expected to be called when the page was modified.
    pub fn mark_dirty(&self, page_id: PageId) {
        self.shared.lock().dirty.insert(page_id);
    }

    /// Flushes all the dirty pages into the underlying store.
    ///
    /// After the commit, all the dirty pages are cleared.
    pub fn commit(&self, tx: &mut Transaction) {
        let mut shared = self.shared.lock();
        for page_id in mem::take(&mut shared.dirty) {
            // Unwrap: the invariant is that all items from `dirty` are present in the `cached` and
            // thus cannot be `None`.
            let page = shared
                .cached
                .get_mut(&page_id)
                .expect("a dirty page is not in the cache");
            let mut page_data = page.inner.lock();
            assert!(page_data.dirty, "dirty page is not marked as dirty");
            page_data.dirty = false;
            tx.write_page(page_id, page_data.data.as_ref());
        }
    }
}
