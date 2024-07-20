use crate::{
    beatree::allocator::free_list::FreeList,
    io::{CompleteIo, IoCommand, IoKind},
    store::{Page, PAGE_SIZE},
};
use crossbeam_channel::{Receiver, Sender, TrySendError};

use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
};

mod free_list;

/// The number of a page
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// The AllocatorReader enables fetching pages from the store.
pub struct AllocatorReader {
    store_file: File,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
}

/// The AllocatorWriter enables dynamic allocation and release of Pages.
/// Upon calling commit, it returns a list of encoded pages that must be written
/// to storage to reflect the store's state at that moment
pub struct AllocatorWriter {
    store_file: File,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
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

/// creates a pair of AllocatorReader and AllocatorWriter over a possibly already existing File.
pub fn create(
    fd: File,
    free_list_head: Option<PageNumber>,
    bump: PageNumber,
    wr_io_handle_index: usize,
    wr_io_sender: Sender<IoCommand>,
    wr_io_receiver: Receiver<CompleteIo>,
    rd_io_handle_index: usize,
    rd_io_sender: Sender<IoCommand>,
    rd_io_receiver: Receiver<CompleteIo>,
) -> (AllocatorReader, AllocatorWriter) {
    let file_size = fd
        .metadata()
        .expect("Error extracting metadata from file")
        .size() as usize;

    let reader_fd = fd.try_clone().expect("failed to clone file");
    let writer_fd = fd;

    let writer = AllocatorWriter {
        free_list: FreeList::read(
            &writer_fd,
            &wr_io_sender,
            wr_io_handle_index,
            &wr_io_receiver,
            free_list_head,
        ),
        store_file: writer_fd,
        io_handle_index: wr_io_handle_index,
        io_sender: wr_io_sender,
        io_receiver: wr_io_receiver,
        bump,
        file_max_bump: PageNumber((file_size / PAGE_SIZE) as u32),
        released: vec![],
    };

    let reader = AllocatorReader {
        store_file: reader_fd,
        io_handle_index: rd_io_handle_index,
        io_sender: rd_io_sender,
        io_receiver: rd_io_receiver,
    };

    (reader, writer)
}

impl AllocatorReader {
    /// Returns the page with the specified page number.
    pub fn query(&self, pn: PageNumber) -> Box<Page> {
        let page = Box::new(Page::zeroed());

        let mut command = Some(IoCommand {
            kind: IoKind::Read(self.store_file.as_raw_fd(), pn.0 as u64, page),
            handle: self.io_handle_index,
            user_data: 0,
        });

        while let Some(c) = command.take() {
            match self.io_sender.try_send(c) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => panic!("I/O store worker dropped"),
                Err(TrySendError::Full(c)) => {
                    command = Some(c);
                }
            }
        }

        // wait for completion
        let completion = self.io_receiver.recv().expect("I/O store worker dropped");
        assert!(completion.result.is_ok());
        let page = completion.command.kind.unwrap_buf();

        page
    }
}

impl AllocatorWriter {
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
    pub fn commit(&mut self) -> AllocatorCommitOutput {
        let released = std::mem::take(&mut self.released);

        let free_list_pages = self.free_list.commit(released, &mut self.bump);

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
            Some(self.file_max_bump.0 as usize * PAGE_SIZE)
        } else {
            None
        };

        AllocatorCommitOutput {
            free_list_pages,
            bump: self.bump,
            extend_file_sz,
            // after appending to the free list, the head will always be present
            freelist_head: self.free_list.head_pn().unwrap(),
        }
    }
}

pub struct AllocatorCommitOutput {
    pub free_list_pages: Vec<(PageNumber, Box<Page>)>,
    pub bump: PageNumber,
    pub extend_file_sz: Option<usize>,
    pub freelist_head: PageNumber,
}
