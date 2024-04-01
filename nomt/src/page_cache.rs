use crate::{
    store::{Store, Transaction},
    Options,
};
use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node},
};
use std::{
    collections::HashMap,
    mem,
    sync::{Arc, Mutex},
    time::Instant,
};
use threadpool::ThreadPool;

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

enum PagePromise {
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
    prestine: HashMap<PageId, Page>,
    /// The pages that were modified, but not yet committed.
    dirty: HashMap<PageId, Page>,
    /// Pages that are currently being fetched from the store.
    inflight: HashMap<PageId, InflightFetch>,
}

impl PageCache {
    pub fn new(store: Store, o: &Options) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            prestine: HashMap::new(),
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
    fn retrieve(&self, page_id: PageId) -> PagePromise {
        let mut shared = self.shared.lock().unwrap();

        // We first check in the dirty pages, since that's where we would find the most recent
        // version of the page. If the page is not present there, we check in the prestine cache.
        if let Some(page) = shared.dirty.get(&page_id) {
            return PagePromise::Now(page.clone());
        }
        if let Some(page) = shared.prestine.get(&page_id) {
            return PagePromise::Now(page.clone());
        }
        // In case none of those are present, we check if there is an inflight fetch for the page.
        if let Some(inflight) = shared.inflight.get(&page_id) {
            return inflight.promise();
        }
        // Nope, then we need to fetch the page from the store.
        let inflight = InflightFetch::new();
        let promise = inflight.promise();
        shared.inflight.insert(page_id, inflight);
        let task = {
            let store = self.store.clone();
            let shared = self.shared.clone();
            move || {
                let entry = store
                    .load_page(page_id)
                    .map(Arc::new)
                    .map_or(Page::Nil, Page::Exists);
                let mut shared = shared.lock().unwrap();
                // Unwrap: the operation was inserted above. It is scheduled for execution only
                // once. It may removed only in the line below. Therefore, `None` is impossible.
                let inflight = shared.inflight.remove(&page_id).unwrap();
                shared.prestine.insert(page_id, entry.clone());
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
        let mut shared = self.shared.lock().unwrap();
        for (page_id, page) in mem::take(&mut shared.dirty) {
            shared.prestine.insert(page_id, page.clone());
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

    /// Creates a new cursor assuming the given root node.
    pub fn create_cursor(&self, root: Node) -> PageCacheCursor {
        PageCacheCursor {
            root,
            page_cache: self.clone(),
        }
    }

    /// Replaces the page in the cache with the given page.
    ///
    /// The inflight fetch for the given page if any is resolved with the given page.
    pub fn supplant(&self, page_id: PageId, page: Page) {
        let mut shared = self.shared.lock().unwrap();
        if let Some(inflight) = shared.inflight.get_mut(&page_id) {
            inflight.supplant_and_notify(page);
        }
    }
}

pub struct PageCacheCursor {
    root: Node,
    page_cache: PageCache,
    // TODO: position
}

impl PageCacheCursor {
    /// Moves the cursor to the given [`KeyPath`].
    ///
    /// Moving the cursor using this function would be more efficient than using the navigation
    /// functions such as [`Self::down_left`] and [`Self::down_right`] due to leveraging warmup
    /// hints.
    ///
    /// After returning of this function, the cursor is positioned either at the given key path or
    /// at the closest key path that is on the way to the given key path.
    pub fn seek(&mut self, dest: KeyPath) {
        const PREFETCH_N: usize = 7;
        // TODO: destruct the key path, extract the first `PREFETCH_N` page IDs and initiate
        // the fetches. Then, traverse only the first `PREFETCH_N` pages and if we haven't
        // encountered the terminal, fetch another batch of `PREFETCH_N` pages and repeat.

        // TODO: self.page_cache.retrieve(page_id)
        todo!()
    }

    // TODO: actually, I am ok with the existing terminology of `traverse`.
    pub fn up(&self, d: u8) {}

    pub fn down_left(&self) {}

    pub fn down_right(&self) {}
}
