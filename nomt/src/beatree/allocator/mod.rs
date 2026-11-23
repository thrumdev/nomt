use crate::io::{self, page_pool::FatPage, IoCommand, IoKind, PagePool, PAGE_SIZE};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::{ArcMutexGuard, Mutex};
use std::{
    collections::BTreeSet,
    fs::File,
    os::{
        fd::{AsRawFd, RawFd},
        unix::fs::MetadataExt,
    },
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
};

use free_list::FreeList;

mod free_list;

/// The number of a page
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageNumber(pub u32);

impl PageNumber {
    pub fn is_nil(&self) -> bool {
        self.0 == 0
    }
}

impl From<u32> for PageNumber {
    fn from(x: u32) -> Self {
        PageNumber(x)
    }
}

/// 0 is used to indicate that the free-list is empty.
pub const FREELIST_EMPTY: PageNumber = PageNumber(0);

/// A store is a file keeping beatree data pages.
///
/// The store is shadow-paged and makes use of an embedded free-list to track free pages.
#[derive(Clone)]
pub struct Store {
    file: Arc<File>,
    sync: Arc<Mutex<StoreSync>>,
}

impl Store {
    /// Create a new `Store` over an existing file.
    pub fn open(
        page_pool: &PagePool,
        file: Arc<File>,
        bump: PageNumber,
        free_list_head: Option<PageNumber>,
    ) -> anyhow::Result<Self> {
        let file_size = file.metadata()?.size() as usize;

        let sync = StoreSync {
            free_list: FreeList::read(page_pool, &file, free_list_head)?,
            bump,
            max_bump: PageNumber((file_size / PAGE_SIZE) as u32),
        };

        Ok(Store {
            file,
            sync: Arc::new(Mutex::new(sync)),
        })
    }

    /// Reads the page with the specified page number. Blocks the current thread.
    pub fn query(&self, page_pool: &PagePool, pn: PageNumber) -> FatPage {
        io::read_page(page_pool, &self.file, pn.0 as u64).unwrap()
    }

    /// Create an I/O command for querying a page by number.
    pub fn io_command(&self, page_pool: &PagePool, pn: PageNumber, user_data: u64) -> IoCommand {
        let page = page_pool.alloc_fat_page();
        IoCommand {
            kind: IoKind::Read(self.file.as_raw_fd(), pn.0 as u64, page),
            user_data,
        }
    }

    /// Get all pages tracked by the free-list, including the pages storing the free-list.
    ///
    /// Deadlocks if sync is ongoing.
    pub fn all_tracked_freelist_pages(&self) -> BTreeSet<PageNumber> {
        self.sync.lock().free_list.all_tracked_pages()
    }

    /// Start synchronization. This produces two handles,
    /// a [`SyncAllocator`] and a [`SyncFinisher`].
    ///
    /// The `SyncAllocator` may be cloned and shared between threads to allocate pages. The finisher
    /// should be used to update the embedded free-list and prepare the writes for that purpose.
    ///
    /// This will block if another sync is in progress.
    pub fn start_sync(&self) -> (SyncAllocator, SyncFinisher) {
        let sync = Mutex::lock_arc(&self.sync);
        // Reuse free pages only once there are enough of them that the free pages at the cursor
        // form long runs. Until then, append.
        let free = sync.free_list.as_clean().len();
        let live = (sync.bump.0 as usize).saturating_sub(free);
        let free_len = if free >= FREE_SLACK * live { free } else { 0 };
        let (sync_tx, sync_rx) = crossbeam_channel::bounded(1);

        let finisher = SyncFinisher {
            file: self.file.clone(),
            sync_finish: sync_rx,
        };

        let allocator = SyncAllocator {
            file: self.file.clone(),
            inner: Arc::new(SyncAllocatorInner {
                max_bump: AtomicU32::new(sync.max_bump.0),
                set_len_lock: Mutex::new(sync.max_bump),
                sync: Some(sync),
                send_to_finish: sync_tx,
                allocations: AtomicUsize::new(0),
                free_len,
            }),
        };

        (allocator, finisher)
    }

    /// Get the raw FD of the store.
    pub fn store_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// A convenience wrapper around a [`Store`]. This wraps the page pool, along with
/// the store.
#[derive(Clone)]
pub struct StoreReader {
    store: Store,
    page_pool: PagePool,
}

impl StoreReader {
    /// Create a new [`StoreReader`].
    pub fn new(store: Store, page_pool: PagePool) -> Self {
        StoreReader { store, page_pool }
    }

    /// Get a reference to the page pool.
    pub fn page_pool(&self) -> &PagePool {
        &self.page_pool
    }

    /// Reads the page with the specified page number. Blocks the current thread.
    pub fn query(&self, pn: PageNumber) -> FatPage {
        self.store.query(&self.page_pool, pn)
    }

    /// Create an I/O command for querying a page by number.
    pub fn io_command(&self, pn: PageNumber, user_data: u64) -> IoCommand {
        self.store.io_command(&self.page_pool, pn, user_data)
    }
}

struct StoreSync {
    /// the next page number of the store.
    bump: PageNumber,
    /// the maximum capacity of the store. beyond this size, the store must have its
    /// length extended.
    max_bump: PageNumber,
    /// the free-list of pages.
    free_list: FreeList,
}

type StoreSyncGuard = ArcMutexGuard<parking_lot::RawMutex, StoreSync>;

/// Free pages are reused once they number this many times the live pages.
const FREE_SLACK: usize = 2;

// Grow the store by 32MB at a time.
const GROW_STORE_BY_PAGES: u32 = 8192;

/// The sync allocator can be used by multiple threads to prospectively allocate pages in the store.
///
/// Pages which are allocated may be safely and immediately written as soon as
/// `SyncFinisher::prepare_write` is called.
#[derive(Clone)]
pub struct SyncAllocator {
    file: Arc<File>,
    inner: Arc<SyncAllocatorInner>,
}

impl SyncAllocator {
    /// Get the next page number.
    ///
    /// Most of the time, this requires no synchronization. Occasionally,
    /// it will extend the length of the file and block other threads for a short period.
    ///
    /// It is acceptable for writes to be out-of-order relative to the order in which they were
    /// allocated.
    ///
    /// This returns an error when setting the length of the file fails.
    pub fn allocate(&self) -> std::io::Result<PageNumber> {
        let allocation_index = self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        self.page_at(allocation_index)
    }

    /// Get the next `n` page numbers, in allocation order.
    ///
    /// Consecutive allocations come from consecutive free-list entries or from the bump, so a
    /// caller writing them in order issues writes that are as close together as the store allows.
    /// Page numbers that end up unused must be returned as freed pages.
    pub fn allocate_many(&self, n: usize) -> std::io::Result<Vec<PageNumber>> {
        let first = self.inner.allocations.fetch_add(n, Ordering::Relaxed);
        (first..first + n).map(|i| self.page_at(i)).collect()
    }

    fn page_at(&self, allocation_index: usize) -> std::io::Result<PageNumber> {
        let sync = self.inner.sync();

        let free_list = sync.free_list.as_clean();
        let free_len = self.inner.free_len;
        if allocation_index >= free_len {
            let pn = PageNumber(sync.bump.0 + (allocation_index - free_len) as u32);

            // fast path: no contention
            if pn.0 < self.inner.max_bump.load(Ordering::Relaxed) {
                return Ok(pn);
            }

            // slow path: take the lock and check.
            let mut set_len_guard = self.inner.set_len_lock.lock();
            if pn.0 < set_len_guard.0 {
                // lost race with another thread, but they already set the length.
                return Ok(pn);
            }

            *set_len_guard = grow(&self.file, pn)?;

            // note that we only write the atomic while the mutex guard is live.
            self.inner
                .max_bump
                .store(set_len_guard.0, Ordering::Relaxed);

            Ok(pn)
        } else {
            Ok(free_list.get_nth_pop(allocation_index))
        }
    }

    /// Get the raw FD of the store.
    pub fn store_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

struct SyncAllocatorInner {
    sync: Option<StoreSyncGuard>,
    // The number of free-list entries this sync allocates from before bumping.
    free_len: usize,
    send_to_finish: Sender<Finish>,
    allocations: AtomicUsize,
    max_bump: AtomicU32,
    set_len_lock: Mutex<PageNumber>,
}

impl SyncAllocatorInner {
    fn sync(&self) -> &StoreSyncGuard {
        // UNWRAP: `sync` is initialized to `Some` and only taken when dropped.
        self.sync.as_ref().unwrap()
    }
}

impl Drop for SyncAllocatorInner {
    fn drop(&mut self) {
        // UNWRAP: `sync` is initialized to `Some` and only taken when dropped.
        let _ = self.send_to_finish.send(Finish {
            sync: self.sync.take().unwrap(),
            allocations: *self.allocations.get_mut(),
            free_len: self.free_len,
            max_bump: *self.set_len_lock.get_mut(),
        });
    }
}

/// Grow the file to include the given page number.
///
/// The file is extended to the next chunk boundary defined by
/// GROW_STORE_BY_PAGES.
///
/// If `page` is already aligned, an extra chunk is added.
///
/// Returns the new boundary (the last accessible page) or an I/O error.
fn grow(file: &File, page: PageNumber) -> std::io::Result<PageNumber> {
    let next_bump = (page.0 + GROW_STORE_BY_PAGES - 1).next_multiple_of(GROW_STORE_BY_PAGES);
    let len = next_bump as u64 * PAGE_SIZE as u64;
    // Allocate the blocks up front: direct writes into holes allocate them one write at a time.
    // SAFETY: plain syscall on a valid fd.
    if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len as libc::off_t) } != 0 {
        file.set_len(len)?;
    }
    Ok(PageNumber(next_bump))
}

struct Finish {
    sync: StoreSyncGuard,
    allocations: usize,
    free_len: usize,
    max_bump: PageNumber,
}

/// The sync finisher is used to process and finalize the writes to the store.
///
/// This does not actually perform any writes, except to alter the length of the store file.
pub struct SyncFinisher {
    file: Arc<File>,
    sync_finish: Receiver<Finish>,
}

impl SyncFinisher {
    /// Finish the sync, updating the store metadata.
    ///
    /// Provide a vector of all freed pages.
    /// This produces a final set of pages to write to update the embedded free-list in the store.
    ///
    /// This returns an error only if the file could not be extended.
    pub fn finish(
        self,
        page_pool: &PagePool,
        freed: Vec<PageNumber>,
    ) -> std::io::Result<(Vec<(PageNumber, FatPage)>, StoreMeta)> {
        // Block on `sync_finish`.
        // UNWRAP: `SyncAllocator` sends the guard when dropped. We assume it is not leaked.
        let Finish {
            mut sync,
            allocations,
            free_len,
            mut max_bump,
        } = self.sync_finish.recv().unwrap();

        let used = allocations.min(free_len);

        // remaining allocations all logically incremented bump.
        let mut next_bump = PageNumber(sync.bump.0 + (allocations - used) as u32);
        let freelist_pages = sync.free_list.rewrite(page_pool, used, freed, &mut next_bump);

        // writing the free-list pages might require more bumps, which may require growing the file
        // further.
        if next_bump.0 > max_bump.0 {
            max_bump = grow(&self.file, next_bump)?;
        }

        sync.bump = next_bump;
        sync.max_bump = max_bump;

        let meta = StoreMeta {
            freelist_pn: sync.free_list.head_pn().unwrap_or(FREELIST_EMPTY).0,
            bump: next_bump.0,
        };
        Ok((freelist_pages, meta))
    }
}

/// New store metadata following a sync.
pub struct StoreMeta {
    /// The page-number indicating the head of the free-list.
    pub freelist_pn: u32,
    /// The next free page number.
    pub bump: u32,
}
