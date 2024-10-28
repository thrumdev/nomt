use crate::io::{self, page_pool::FatPage, IoCommand, IoHandle, IoKind, PagePool, PAGE_SIZE};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::{ArcMutexGuard, Mutex};
use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
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

/// The AllocatorReader enables fetching pages from the store.
///
/// HACK: clones will be able to receive completions from each other due to the nature of cloning
/// an IoHandle.
#[derive(Clone)]
pub struct AllocatorReader {
    // HACK: might be better to not have this struct and instead just expose raw functions over a
    // file and an Io handle. Not much benefit to owning the Io handle here.
    store_file: Arc<File>,
    io_handle: IoHandle,
}

/// The AllocatorWriter enables dynamic allocation and release of Pages.
/// Upon calling commit, it returns a list of encoded pages that must be written
/// to storage to reflect the store's state at that moment
pub struct AllocatorWriter {
    // Monotonic page number, used when the free list is empty
    bump: PageNumber,
    // The store is an array of pages, with indices as PageNumbers,
    // file_max_bump can be considered as either the size of the array
    // or a page number one greater than the maximum bump value that can be used to
    // safely write a page to storage without necessitating a file growth operation
    file_max_bump: PageNumber,
    free_list: FreeList,
    // Used for storing transitional data between commits
    released: Vec<PageNumber>,
}

impl AllocatorReader {
    /// creates an AllocatorReader over a possibly already existing File.
    pub fn new(fd: File, io_handle: IoHandle) -> Self {
        AllocatorReader {
            store_file: Arc::new(fd),
            io_handle,
        }
    }

    /// Returns the page with the specified page number. Blocks the current thread.
    pub fn query(&self, pn: PageNumber) -> FatPage {
        io::read_page(self.io_handle.page_pool(), &self.store_file, pn.0 as u64).unwrap()
    }

    /// Get a reference to the I/O handle.
    pub fn io_handle(&self) -> &IoHandle {
        &self.io_handle
    }

    /// Create an I/O command for querying a page by number.
    pub fn io_command(&self, pn: PageNumber, user_data: u64) -> IoCommand {
        let page = self.io_handle.page_pool().alloc_fat_page();
        IoCommand {
            kind: IoKind::Read(self.store_file.as_raw_fd(), pn.0 as u64, page),
            user_data,
        }
    }
}

impl AllocatorWriter {
    /// creates an AllocatorWriter over an already existing File.
    pub fn open(
        page_pool: &PagePool,
        fd: File,
        free_list_head: Option<PageNumber>,
        bump: PageNumber,
    ) -> Self {
        let file_size = fd
            .metadata()
            .expect("Error extracting metadata from file")
            .size() as usize;

        AllocatorWriter {
            free_list: FreeList::read(page_pool, &fd, free_list_head).unwrap(),
            bump,
            file_max_bump: PageNumber((file_size / PAGE_SIZE) as u32),
            released: vec![],
        }
    }

    pub fn free_list(&self) -> &FreeList {
        &self.free_list
    }

    pub fn allocate(&mut self) -> PageNumber {
        let pn = match self.free_list.pop() {
            Some(pn) => pn,
            None => {
                let pn = self.bump;
                self.bump.0 += 1;
                pn
            }
        };

        pn
    }

    pub fn release(&mut self, id: PageNumber) {
        self.released.push(id);
    }

    // Commits the changes creating a set of pages that needs to be written into the store.
    //
    // The output will include not only the list of pages that need to be written but also
    // the new free_list head, the current bump page number, and the new required file size
    pub fn commit(&mut self, page_pool: &PagePool) -> AllocatorCommitOutput {
        let released = std::mem::take(&mut self.released);

        let free_list_pages = self.free_list.commit(page_pool, released, &mut self.bump);

        // The store is expected to grow in increments of 1 MiB blocks,
        // equivalent to chunks of 256 4KiB pages.
        //
        // If the self.bump exceeds the file_max_bump,
        // the file will not be resized to store only the extra pages,
        // but rather resized to store at least the new pages and possibly
        // leaving some empty pages at the end.
        let next_max_bump = self.bump.0.next_multiple_of(256);
        let extend_file_sz = if self.file_max_bump.0 < next_max_bump {
            self.file_max_bump = PageNumber(next_max_bump);
            Some(self.file_max_bump.0 as u64 * PAGE_SIZE as u64)
        } else {
            None
        };

        AllocatorCommitOutput {
            free_list_pages,
            bump: self.bump,
            extend_file_sz,
            freelist_head: self.free_list.head_pn().unwrap_or(FREELIST_EMPTY),
        }
    }
}

pub struct AllocatorCommitOutput {
    pub free_list_pages: Vec<(PageNumber, FatPage)>,
    pub bump: PageNumber,
    pub extend_file_sz: Option<u64>,
    pub freelist_head: PageNumber,
}

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
        file: File,
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
            file: Arc::new(file),
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

    /// Start synchronization. This produces two handles,
    /// a [`SyncAllocator`] and a [`SyncFinisher`].
    ///
    /// The `SyncAllocator` may be cloned and shared between threads to allocate pages. The finisher
    /// should be used to update the embedded free-list and prepare the writes for that purpose.
    ///
    /// This will block if another sync is in progress.
    pub fn start_sync(&self) -> (SyncAllocator, SyncFinisher) {
        let sync = Mutex::lock_arc(&self.sync);
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
            }),
        };

        (allocator, finisher)
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

// Grow the store by 1MB at a time.
const GROW_STORE_BY_PAGES: u32 = 256;

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
    pub fn allocate(&mut self) -> anyhow::Result<PageNumber> {
        let allocation_index = self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        let sync = self.inner.sync();

        let free_list = sync.free_list.as_clean();
        if allocation_index >= free_list.len() {
            let pn = PageNumber(sync.bump.0 + (allocation_index - free_list.len()) as u32);

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
}

struct SyncAllocatorInner {
    sync: Option<StoreSyncGuard>,
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
            max_bump: *self.set_len_lock.get_mut(),
        });
    }
}

// grow the file to accommodate writes to the page with the given number. returns the new boundary
// page of the file.
//
// returns an error when growing the file fails.
fn grow(file: &File, page: PageNumber) -> anyhow::Result<PageNumber> {
    let next_bump = page.0.next_multiple_of(GROW_STORE_BY_PAGES);
    file.set_len(next_bump as u64 * PAGE_SIZE as u64)?;
    Ok(PageNumber(next_bump))
}

struct Finish {
    sync: StoreSyncGuard,
    allocations: usize,
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
        mut self,
        page_pool: &PagePool,
        freed: Vec<PageNumber>,
    ) -> anyhow::Result<Vec<(PageNumber, FatPage)>> {
        // Block on `sync_finish`.
        // UNWRAP: `SyncAllocator` sends the guard when dropped. We assume it is not leaked.
        let Finish {
            mut sync,
            allocations: mut n_allocations,
            mut max_bump,
        } = self.sync_finish.recv().unwrap();

        while n_allocations > 0 {
            if sync.free_list.pop().is_some() {
                n_allocations -= 1;
            } else {
                break;
            }
        }

        // remaining allocations all logically incremented bump.
        let mut next_bump = PageNumber(sync.bump.0 + n_allocations as u32);
        let freelist_pages = sync.free_list.commit(page_pool, freed, &mut next_bump);

        // writing the free-list pages might require more bumps, which may require growing the file
        // further.
        if next_bump.0 > max_bump.0 {
            max_bump = grow(&self.file, next_bump)?;
        }

        sync.bump = next_bump;
        sync.max_bump = max_bump;

        Ok(freelist_pages)
    }
}
