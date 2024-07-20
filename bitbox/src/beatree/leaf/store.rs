use crate::beatree::leaf::free_list::FreeList;
use crate::beatree::leaf::{node::LeafNode, PageNumber};
use crate::io::{CompleteIo, IoCommand, IoKind};
use crate::store::{Page, PAGE_SIZE};
use crossbeam_channel::{Receiver, Sender, TrySendError};

use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;

pub struct LeafStoreReader {
    store_file: File,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
}

/// The LeafStoreWriter enables dynamic allocation and release of Leaf Pages.
/// Upon calling commit, it returns a list of encoded pages that must be written
/// to storage to reflect the LeafStore's state at that moment
pub struct LeafStoreWriter {
    store_file: File,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
    // Monotonic page number, used when the free list is empty
    bump: PageNumber,
    // The leaf store is an array of pages, with indices as PageNumbers,
    // file_max_bump can be considered as either the size of the array
    // or a page number one greater than the maximum bump value that can be used to
    // safely write a page to storage without necessitating a file growth operation
    file_max_bump: PageNumber,
    free_list: FreeList,
    // Used for storing transitional data between commits
    released: Vec<PageNumber>,
    pending: Vec<(PageNumber, LeafNode)>,
}

/// creates a pair of LeafStoreReader and LeafStoreWriter over a possibly already existing File.
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
) -> (LeafStoreReader, LeafStoreWriter) {
    let file_size = fd
        .metadata()
        .expect("Error extracting metadata from LeafStore file")
        .size() as usize;

    let reader_fd = fd.try_clone().expect("failed to clone LeafStore file");
    let writer_fd = fd;

    let writer = LeafStoreWriter {
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
        pending: vec![],
        released: vec![],
    };

    let reader = LeafStoreReader {
        store_file: reader_fd,
        io_handle_index: rd_io_handle_index,
        io_sender: rd_io_sender,
        io_receiver: rd_io_receiver,
    };

    (reader, writer)
}

impl LeafStoreReader {
    /// Returns the leaf page with the specified page number.
    pub fn query(&self, pn: PageNumber) -> LeafNode {
        let page = Box::new(Page::zeroed());

        let mut command = Some(IoCommand {
            kind: IoKind::Read(self.store_file.as_raw_fd(), pn.0 as u64, page),
            handle: self.io_handle_index,
            user_data: 0,
        });

        while let Some(c) = command.take() {
            match self.io_sender.try_send(c) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => panic!("I/O leaf store worker dropped"),
                Err(TrySendError::Full(c)) => {
                    command = Some(c);
                }
            }
        }

        // wait for completion
        let completion = self
            .io_receiver
            .recv()
            .expect("I/O leaf store worker dropped");
        assert!(completion.result.is_ok());
        let page = completion.command.kind.unwrap_buf();

        LeafNode { inner: page }
    }
}

impl LeafStoreWriter {
    pub fn allocate(&mut self, leaf_page: LeafNode) -> PageNumber {
        let leaf_pn = match self.free_list.pop() {
            Some(pn) => pn,
            None => {
                let pn = self.bump;
                self.bump.0 += 1;
                pn
            }
        };

        self.pending.push((leaf_pn, leaf_page));

        leaf_pn
    }

    pub fn release(&mut self, id: PageNumber) {
        self.released.push(id);
    }

    // Commits the changes creating a set of pages that needs to be written into the LeafStore.
    //
    // The output will include not only the list of pages that need to be written but also
    // the new free_list head, the current bump page number, and the new file size if extending is needed
    pub fn commit(&mut self) -> LeafStoreCommitOutput {
        let released = std::mem::take(&mut self.released);
        let pending = std::mem::take(&mut self.pending);

        let free_list_pages = self.free_list.commit(released, &mut self.bump);

        let pages = pending
            .into_iter()
            .map(|(pn, leaf_page)| (pn, leaf_page.inner))
            .chain(
                free_list_pages
                    .into_iter()
                    .map(|(pn, free_list_page)| (pn, free_list_page.inner)),
            )
            .collect::<Vec<_>>();

        // The LeafStore is expected to grow in increments of 1 MiB blocks,
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

        LeafStoreCommitOutput {
            pages,
            bump: self.bump,
            extend_file_sz,
            // after appending to the free list, the head will always be present
            freelist_head_pn: self.free_list.head_pn().unwrap(),
        }
    }
}

pub struct LeafStoreCommitOutput {
    pub pages: Vec<(PageNumber, Box<Page>)>,
    pub bump: PageNumber,
    pub extend_file_sz: Option<usize>,
    pub freelist_head_pn: PageNumber,
}
