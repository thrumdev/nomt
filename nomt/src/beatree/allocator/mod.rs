use crate::io::{
    self,
    page_pool::{FatPage, Page},
    IoCommand, IoHandle, IoKind, PagePool, PAGE_SIZE,
};

use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    sync::Arc,
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

    /// Reads the page into the given buffer. Blocks the current thread.
    pub fn query_into(&self, buf: &mut [u8], pn: PageNumber) {
        io::read_page_into(buf, &self.store_file, pn.0 as u64).unwrap()
    }

    /// Get a reference to the I/O handle.
    pub fn io_handle(&self) -> &IoHandle {
        &self.io_handle
    }

    /// Get a reference to the page pool.
    pub fn page_pool(&self) -> &PagePool {
        self.io_handle.page_pool()
    }

    /// Create an I/O command for querying a page by number.
    ///
    /// Safety: the page must be live and unaliased until the command is dropped or completed.
    pub unsafe fn io_command(&self, pn: PageNumber, user_data: u64, page: &Page) -> IoCommand {
        IoCommand {
            kind: IoKind::ReadRaw(
                self.store_file.as_raw_fd(),
                pn.0 as u64,
                page.as_mut_ptr(),
                PAGE_SIZE,
            ),
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
            free_list: FreeList::read(page_pool, &fd, free_list_head),
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
