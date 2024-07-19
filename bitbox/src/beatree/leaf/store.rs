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

pub struct LeafStoreWriter {
    store_file: File,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
    // Monotonic page number, used when the free list is empty
    bump: PageNumber,
    // This is the max supported page number + 1.
    // + 1 to easily handle the case where the file is empty and each page,
    // starting from page_number 0, will exceed
    max_bump: PageNumber,
    free_list: FreeList,
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
        max_bump: PageNumber((file_size / PAGE_SIZE) as u32),
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
    // create a LeafStoreTx able to append and release leaves from the LeafStore
    pub fn start_tx<'a>(&'a mut self) -> LeafStoreTx<'a> {
        LeafStoreTx {
            bump: self.bump,
            free_list: &mut self.free_list,
            max_bump: self.max_bump,
            to_allocate: vec![],
            released: vec![],
            exceeded: vec![],
        }
    }
}

pub struct LeafStoreTx<'a> {
    bump: PageNumber,
    free_list: &'a mut FreeList,
    max_bump: PageNumber,
    released: Vec<PageNumber>,
    to_allocate: Vec<(PageNumber, LeafNode)>,
    exceeded: Vec<(PageNumber, LeafNode)>,
}

pub struct LeafStoreTxOutput {
    // contains pages that can already be written in the LeafStore
    pub to_allocate: Vec<(PageNumber, Box<Page>)>,
    // contains pages that require the LeafStore to grow
    pub exceeded: Vec<(PageNumber, Box<Page>)>,
    pub new_free_list_head: PageNumber,
}

impl<'a> LeafStoreTx<'a> {
    pub fn allocate(&mut self, leaf_page: LeafNode) -> PageNumber {
        let leaf_pn = match self.free_list.pop() {
            Some(pn) => pn,
            None => {
                let pn = self.bump;
                self.bump.0 += 1;
                pn
            }
        };

        if leaf_pn.0 < self.max_bump.0 {
            self.to_allocate.push((leaf_pn, leaf_page));
        } else {
            self.exceeded.push((leaf_pn, leaf_page));
        }
        leaf_pn
    }

    pub fn release(&mut self, id: PageNumber) {
        self.released.push(id);
    }

    // commits the changes creating a set of pages that needs to be written into the LeafStore
    pub fn commit(mut self) -> LeafStoreTxOutput {
        let free_list_output = self
            .free_list
            .commit(self.released, &mut self.bump, self.max_bump);

        let to_allocate = self
            .to_allocate
            .into_iter()
            .map(|(pn, leaf_page)| (pn, leaf_page.inner))
            .chain(
                free_list_output
                    .to_allocate
                    .into_iter()
                    .map(|(pn, free_list_page)| (pn, free_list_page.inner)),
            )
            .collect::<Vec<_>>();

        let exceeded = self
            .exceeded
            .into_iter()
            .map(|(pn, leaf_page)| (pn, leaf_page.inner))
            .chain(
                free_list_output
                    .exceeded
                    .into_iter()
                    .map(|(pn, free_list_page)| (pn, free_list_page.inner)),
            )
            .collect::<Vec<_>>();

        LeafStoreTxOutput {
            to_allocate,
            exceeded,
            // after appending to the free list, the head will always be present
            new_free_list_head: self.free_list.head_pn().unwrap(),
        }
    }
}
