use crate::{
    beatree::{
        allocator::{self, AllocatorCommitOutput, AllocatorReader, AllocatorWriter, PageNumber},
        leaf::node::LeafNode,
    },
    io::{CompleteIo, IoCommand},
    store::Page,
};
use crossbeam_channel::{Receiver, Sender};

use std::fs::File;

pub struct LeafStoreReader {
    allocator_reader: AllocatorReader,
}

/// The LeafStoreWriter enables dynamic allocation and release of Leaf Pages.
/// Upon calling commit, it returns a list of encoded pages that must be written
/// to storage to reflect the LeafStore's state at that moment
pub struct LeafStoreWriter {
    allocator_writer: AllocatorWriter,
    pending: Vec<(PageNumber, Box<Page>)>,
}

/// creates a pair of LeafStoreReader and LeafStoreWriter over a possibly already existing File.
pub fn create(
    fd: File,
    free_list_head: Option<PageNumber>,
    bump: PageNumber,
    rd_io_handle_index_shared: usize,
    rd_io_sender_shared: Sender<IoCommand>,
    rd_io_receiver_shared: Receiver<CompleteIo>,
    rd_io_handle_index_sync: usize,
    rd_io_sender_sync: Sender<IoCommand>,
    rd_io_receiver_sync: Receiver<CompleteIo>,
    wr_io_handle_index: usize,
    wr_io_sender: Sender<IoCommand>,
    wr_io_receiver: Receiver<CompleteIo>,
) -> (LeafStoreReader, LeafStoreReader, LeafStoreWriter) {
    let allocator_reader_shared = AllocatorReader::new(
        fd.try_clone().expect("failed to clone file"),
        rd_io_handle_index_shared,
        rd_io_sender_shared,
        rd_io_receiver_shared,
    );

    let allocator_reader_sync = AllocatorReader::new(
        fd.try_clone().expect("failed to clone file"),
        rd_io_handle_index_sync,
        rd_io_sender_sync,
        rd_io_receiver_sync,
    );

    let allocator_writer = AllocatorWriter::new(
        fd,
        free_list_head,
        bump,
        wr_io_handle_index,
        wr_io_sender,
        wr_io_receiver,
    );

    (
        LeafStoreReader {
            allocator_reader: allocator_reader_shared,
        },
        LeafStoreReader {
            allocator_reader: allocator_reader_sync,
        },
        LeafStoreWriter {
            allocator_writer,
            pending: vec![],
        },
    )
}

impl LeafStoreReader {
    /// Returns the leaf page with the specified page number.
    pub fn query(&self, pn: PageNumber) -> LeafNode {
        let page = self.allocator_reader.query(pn);

        LeafNode { inner: page }
    }
}

impl LeafStoreWriter {
    pub fn allocate(&mut self, leaf_page: LeafNode) -> PageNumber {
        let pn = self.allocator_writer.allocate();
        self.pending.push((pn, leaf_page.inner));
        pn
    }

    pub fn release(&mut self, id: PageNumber) {
        self.allocator_writer.release(id)
    }

    // Commits the changes creating a set of pages that needs to be written into the LeafStore.
    //
    // The output will include not only the list of pages that need to be written but also
    // the new free_list head, the current bump page number, and the new file size if extending is needed
    pub fn commit(&mut self) -> LeafStoreCommitOutput {
        let pending = std::mem::take(&mut self.pending);

        let AllocatorCommitOutput {
            free_list_pages,
            bump,
            extend_file_sz,
            freelist_head,
        } = self.allocator_writer.commit();

        LeafStoreCommitOutput {
            pages: pending.into_iter().chain(free_list_pages).collect(),
            bump,
            extend_file_sz,
            freelist_head,
        }
    }
}

pub struct LeafStoreCommitOutput {
    pub pages: Vec<(PageNumber, Box<Page>)>,
    pub bump: PageNumber,
    pub extend_file_sz: Option<u64>,
    pub freelist_head: PageNumber,
}
