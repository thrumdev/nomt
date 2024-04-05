use crate::{
    store::{Store, Transaction},
    Options,
};
use nomt_core::{page::DEPTH, page_id::PageId, trie::Node};
use parking_lot::{Condvar, Mutex};
use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::{Arc, Weak},
};

use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

#[derive(Clone)]
pub struct PageData {
    data: Option<Vec<u8>>,
}

impl PageData {
    /// Creates a page with the given data.
    fn pristine_with_data(data: Vec<u8>) -> Self {
        Self { data: Some(data) }
    }

    /// Creates an empty page.
    fn pristine_empty() -> Self {
        Self { data: None }
    }

    /// Read out the node at the given index.
    pub fn node(&self, index: usize) -> Node {
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
    pub fn set_node(&mut self, index: usize, node: Node) {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        if self.data.is_none() {
            self.data = Some(vec![0; 4096]);
        }
        let start = index * 32;
        let end = start + 32;
        // Unwrap: self.data is Some, we just ensured it above.
        self.data.as_mut().unwrap()[start..end].copy_from_slice(&node);
    }
}

/// A reference to a page within the cache.
///
/// These references are not intended to be long-lived and may become invalid.
///
/// Keeping long-lived references to page data can impact performance, by forcing later
/// modifications to copy-on-write rather than modifying in place.
pub struct PageRef {
    page_id: PageId,
    inner: Weak<PageData>,
}

impl PageRef {
    pub fn id(&self) -> PageId {
        self.page_id.clone()
    }
}

pub enum PagePromise {
    Now(PageRef),
    Later(crossbeam_channel::Receiver<PageRef>),
}

impl PagePromise {
    /// Waits for the page to be fetched.
    pub fn wait(self) -> PageRef {
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
    tx: Option<crossbeam_channel::Sender<PageRef>>,
    rx: crossbeam_channel::Receiver<PageRef>,
}

impl InflightFetch {
    fn new() -> Self {
        let (tx, rx) = crossbeam_channel::bounded(1);
        Self { tx: Some(tx), rx }
    }

    /// Notifies all the waiting parties that the page has been fetched and destroys this handle.
    fn complete_and_notify(mut self, page: PageRef) {
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
    inflight_completed: Arc<Condvar>,
    /// The thread pool used for fetching pages from the store.
    ///
    /// Used for limiting the number of concurrent page fetches.
    fetch_tp: ThreadPool,
}

struct Shared {
    /// The pages loaded from the store, possibly dirty.
    cached: HashMap<PageId, Arc<PageData>>,
    /// The pages that were modified and need to be flushed to the store.
    ///
    /// The invariant is that the pages in the dirty set are always present in the `cached` map and
    /// are marked as dirty there.
    dirty: HashSet<PageId>,
    /// Pages that are currently being fetched from the store.
    inflight: HashMap<PageId, InflightFetch>,
}

impl Shared {
    fn start_retrieve(
        &mut self,
        fetch_tp: &ThreadPool,
        page_id: PageId,
        store: Store,
        shared: Arc<Mutex<Shared>>,
        inflight_completed: Arc<Condvar>,
        strong_tx: Option<crossbeam_channel::Sender<Arc<PageData>>>,
    ) -> PagePromise {
        let inflight = InflightFetch::new();
        let promise = inflight.promise();
        self.inflight.insert(page_id.clone(), inflight);

        let task = move || {
            let entry = store
                .load_page(page_id.clone())
                .expect("db load failed") // TODO: handle the error
                .map_or_else(PageData::pristine_empty, PageData::pristine_with_data);
            let entry = Arc::new(entry);
            let page_ref = PageRef {
                page_id: page_id.clone(),
                inner: Arc::downgrade(&entry),
            };

            let mut shared = shared.lock();
            // Unwrap: the operation was inserted above. It is scheduled for execution only
            // once. It may removed only in the line below. Therefore, `None` is impossible.
            let inflight = shared.inflight.remove(&page_id).unwrap();
            shared.cached.insert(page_id, entry.clone());
            inflight.complete_and_notify(page_ref);
            if let Some(strong_tx) = strong_tx {
                let _ = strong_tx.send(entry);
            }
            inflight_completed.notify_all();
        };

        fetch_tp.execute(task);
        promise
    }
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
            inflight_completed: Arc::new(Condvar::new()),
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
            return PagePromise::Now(PageRef {
                page_id,
                inner: Arc::downgrade(&page),
            });
        }
        // In case none of those are present, we check if there is an inflight fetch for the page.
        if let Some(inflight) = shared.inflight.get(&page_id) {
            return inflight.promise();
        }
        // Nope, then we need to fetch the page from the store.
        shared.start_retrieve(
            &self.fetch_tp,
            page_id,
            self.store.clone(),
            self.shared.clone(),
            self.inflight_completed.clone(),
            None,
        )
    }

    /// Get a node based on a reference to the given page.
    pub fn node(&self, page_ref: &PageRef, node_index: usize) -> Node {
        if let Some(strong) = page_ref.inner.upgrade() {
            return strong.node(node_index);
        }

        let strong_or_wait = {
            let mut shared = self.shared.lock();
            // ref invalid but cache not
            if let Some(page) = shared.cached.get(&page_ref.page_id) {
                Ok(page.clone())
            } else {
                // cache invalidated.
                // need to fetch.
                let (tx, rx) = crossbeam_channel::bounded(1);
                shared.start_retrieve(
                    &self.fetch_tp,
                    page_ref.page_id.clone(),
                    self.store.clone(),
                    self.shared.clone(),
                    self.inflight_completed.clone(),
                    Some(tx),
                );

                Err(rx)
            }
        };

        // re-entrance safety: only call closure without locks held.
        // deadlock safety: only wait on recv without locks held.

        // TODO: this unwrap must be removed, it's edge-case. Specifically, during the
        //       shutdown it may happen that the sender is dropped before the receiver.
        let strong = strong_or_wait.unwrap_or_else(|rx| rx.recv().unwrap());
        strong.node(node_index)
    }

    /// Acquire a writer for the page cache.
    pub fn acquire_writer(&self) -> PageCacheWriter {
        // we require all in-flight fetches to have concluded.
        let mut shared = self.shared.lock();
        self.inflight_completed
            .wait_while(&mut shared, |s| !s.inflight.is_empty());

        PageCacheWriter {
            shared,
            store: self.store.clone(),
        }
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
            tx.write_page(page_id, page.data.as_ref());
        }
    }
}

pub struct PageCacheWriter<'a> {
    shared: parking_lot::MutexGuard<'a, Shared>,
    store: Store,
}

impl<'a> PageCacheWriter<'a> {
    pub fn retrieve_blocking(&mut self, page_id: PageId) -> PageRef {
        if let Some(page) = self.shared.cached.get(&page_id) {
            return PageRef {
                page_id,
                inner: Arc::downgrade(&page),
            };
        }

        let entry = self
            .store
            .load_page(page_id.clone())
            .expect("db load failed") // TODO: handle the error
            .map_or_else(PageData::pristine_empty, PageData::pristine_with_data);
        let entry = Arc::new(entry);
        let page_ref = PageRef {
            page_id: page_id.clone(),
            inner: Arc::downgrade(&entry),
        };
        self.shared.cached.insert(page_id, entry);
        page_ref
    }

    pub fn node(&self, page_ref: &PageRef, node_index: usize) -> Node {
        if let Some(strong) = page_ref.inner.upgrade() {
            return strong.node(node_index);
        }

        // ref invalid but cache not
        if let Some(page) = self.shared.cached.get(&page_ref.page_id) {
            page.node(node_index)
        } else {
            // cache invalidated
            println!("cache invalidated during write?");
            // TODO: actually fetch
            nomt_core::trie::TERMINATOR
        }
    }

    pub fn set_node(&mut self, page_ref: &mut PageRef, node_index: usize, node: Node) {
        page_ref.inner = Weak::new();
        if let Some(page) = self.shared.cached.get_mut(&page_ref.page_id) {
            // note: this should always have strong count 1 except in cases where another page
            // ref has been squirreled away or there are multiple page caches, or other such
            // degenerate cases.
            {
                if Arc::strong_count(page) != 1 {
                    println!("cloning page unnecessarily");
                }
                let page = Arc::make_mut(page);
                page.set_node(node_index, node);
            }
            page_ref.inner = Arc::downgrade(&page);
        } else {
            // cold: wait on I/O
            let mut entry = self
                .store
                .load_page(page_ref.page_id.clone())
                .expect("db load failed") // TODO: handle the error
                .map_or_else(PageData::pristine_empty, PageData::pristine_with_data);

            entry.set_node(node_index, node);
            let entry = Arc::new(entry);
            page_ref.inner = Arc::downgrade(&entry);
            self.shared.cached.insert(page_ref.page_id, entry);
        };

        self.shared.dirty.insert(page_ref.page_id.clone());
    }
}
