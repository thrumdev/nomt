use crate::{
    store::{Store, Transaction},
    Options,
};
use nomt_core::{page::DEPTH, page_id::PageId, trie::Node};
use parking_lot::Mutex;
use std::{collections::HashMap, fmt, mem, sync::Arc};
use threadpool::ThreadPool;

// Total number of nodes stored in one Page. It depends on the `DEPTH`
// of the rootless sub-binary tree stored in a page following this formula:
// (2^(DEPTH + 1)) - 2
pub const NODES_PER_PAGE: usize = (1 << DEPTH + 1) - 2;

/// A handle to the page.
///
/// Can be cloned cheaply.
#[derive(Clone)]
pub enum Page {
    /// Represents a page that does not exist in the underlying store.
    Nil,
    /// Represents a page that exists in the underlying store.
    Exists(Arc<Vec<u8>>),
}

impl Page {
    /// Read out the node at the given index.
    pub fn node(&self, index: usize) -> Node {
        assert!(index < NODES_PER_PAGE, "index out of bounds");
        match self {
            Page::Nil => [0; 32],
            Page::Exists(data) => {
                let start = index * 32;
                let end = start + 32;
                let mut node = [0; 32];
                node.copy_from_slice(&data[start..end]);
                node
            }
        }
    }
}

impl fmt::Debug for Page {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Page::Nil => write!(f, "Page::Nil"),
            Page::Exists(_) => write!(f, "Page::Exists"),
        }
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
/// - Supplanted. The page has been updated (supplanted) before the fetch has completed. The waiters
///   are notified with the dirty page.
/// - Completed. The page has been fetched and the waiters are notified with the fetched page.
struct InflightFetch {
    tx: Option<crossbeam_channel::Sender<Page>>,
    rx: crossbeam_channel::Receiver<Page>,
    supplanted: Option<Page>,
}

impl InflightFetch {
    fn new() -> Self {
        let (tx, rx) = crossbeam_channel::bounded(1);
        Self {
            tx: Some(tx),
            rx,
            supplanted: None,
        }
    }

    /// Called when the page is updated before the fetch is completed.
    ///
    /// Can be called once. Panics otherwise.
    fn supplant_and_notify(&mut self, page: Page) {
        assert!(self.supplanted.is_none());
        self.supplanted = Some(page.clone());
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(page);
        }
    }

    /// Notifies all the waiting parties that the page has been fetched and destroys this handle.
    fn complete_and_notify(mut self, page: Page) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(page);
        }
    }

    /// Returns the promise that resolves when the page is fetched.
    fn promise(&self) -> PagePromise {
        match &self.supplanted {
            Some(page) => PagePromise::Now(page.clone()),
            None => PagePromise::Later(self.rx.clone()),
        }
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
    /// The unmodified pages loaded from the store.
    ///
    /// This cache must be always coherent with the underlying store. That implies that the updates
    /// performed to the store must be reflected in the cache.
    pristine: HashMap<PageId, Page>,
    /// The pages that were modified, but not yet committed.
    dirty: HashMap<PageId, Page>,
    /// Pages that are currently being fetched from the store.
    inflight: HashMap<PageId, InflightFetch>,
}

impl PageCache {
    /// Create a new `PageCache` atop the provided [`Store`].
    pub fn new(store: Store, o: &Options) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            pristine: HashMap::new(),
            dirty: HashMap::new(),
            inflight: HashMap::new(),
        }));
        let fetch_tp = ThreadPool::new(o.fetch_concurrency);
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

        // We first check in the dirty pages, since that's where we would find the most recent
        // version of the page. If the page is not present there, we check in the pristine cache.
        if let Some(page) = shared.dirty.get(&page_id) {
            return PagePromise::Now(page.clone());
        }
        if let Some(page) = shared.pristine.get(&page_id) {
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
                    .map(Arc::new)
                    .map_or(Page::Nil, Page::Exists);
                let mut shared = shared.lock();
                // Unwrap: the operation was inserted above. It is scheduled for execution only
                // once. It may removed only in the line below. Therefore, `None` is impossible.
                let inflight = shared.inflight.remove(&page_id).unwrap();
                shared.pristine.insert(page_id, entry.clone());
                inflight.complete_and_notify(entry);
                drop(shared);
            }
        };
        self.fetch_tp.execute(task);
        promise
    }

    /// Flushes all the dirty pages into the underlying store.
    ///
    /// After the commit, all the dirty pages are cleared.
    pub fn commit(&self, tx: &mut Transaction) {
        let mut shared = self.shared.lock();
        for (page_id, page) in mem::take(&mut shared.dirty) {
            shared.pristine.insert(page_id.clone(), page.clone());
            let page_data = match page {
                Page::Nil => None,
                Page::Exists(data) => Some(data),
            };
            tx.write_page(page_id, page_data.as_deref());
            // It doesn't seem that the page can be in the inflight fetches at this point.
            //
            // If the page ended up in the dirty set, then it must have been supplanted in case
            // there was an inflight fetch for that page. After it was supplanted, the `retrieve`
            // function would not initiate another fetch for the same page.
            //
            // Therefore, we don't need to divert the inflight fetches here.
        }
    }

    /// Replaces the page in the cache with the given page.
    ///
    /// The inflight fetch for the given page if any is resolved with the given page.
    pub fn supplant(&self, page_id: PageId, page: Page) {
        let mut shared = self.shared.lock();
        shared.dirty.insert(page_id, page.clone());
        if let Some(inflight) = shared.inflight.get_mut(&page_id) {
            inflight.supplant_and_notify(page);
        }
    }
}
