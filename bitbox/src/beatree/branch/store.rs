use crate::{
    beatree::{
        allocator::{self, AllocatorCommitOutput, AllocatorReader, AllocatorWriter, PageNumber},
        branch::BranchNode,
    },
    io::{CompleteIo, IoCommand},
    store::Page,
};
use crossbeam_channel::{Receiver, Sender};

use std::fs::File;

pub struct BbnStoreReader {
    allocator_reader: AllocatorReader,
}

/// The BbnStoreWriter enables dynamic allocation and release of BBNs.
/// Upon calling commit, it returns a list of encoded pages and BranchNodes that must be written
/// to storage to reflect the BbnStore's state at that moment
pub struct BbnStoreWriter {
    allocator_writer: AllocatorWriter,
    pending: Vec<BranchNode>,
}

/// creates a pair of BbnStoreReader and BbnStoreWriter over a possibly already existing File.
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
) -> (BbnStoreReader, BbnStoreWriter) {
    let (allocator_reader, allocator_writer) = allocator::create(
        fd,
        free_list_head,
        bump,
        wr_io_handle_index,
        wr_io_sender,
        wr_io_receiver,
        rd_io_handle_index,
        rd_io_sender,
        rd_io_receiver,
    );

    (
        BbnStoreReader { allocator_reader },
        BbnStoreWriter {
            allocator_writer,
            pending: vec![],
        },
    )
}

impl BbnStoreReader {
    pub fn recover_bbns(self) -> Vec<Box<Page>> {
        let mut valid_pages = vec![];

        // read a free list from store
        let invalid_pages = self.allocator_reader.free_list().into_set();

        // iterate over all pages
        for pn in 0..self.allocator_reader.bump().0 {
            // filter invalid ones
            if !invalid_pages.contains(&pn.into()) {
                let page = self.allocator_reader.query(pn.into());
                valid_pages.push(page);
            }
        }
        valid_pages
    }
}

impl BbnStoreWriter {
    pub fn allocate(&mut self, branch_node: BranchNode) -> PageNumber {
        let pn = self.allocator_writer.allocate();
        self.pending.push(branch_node);
        pn
    }

    pub fn release(&mut self, id: PageNumber) {
        self.allocator_writer.release(id)
    }

    // Commits the changes creating a set of pages that needs to be written into the BbnStore.
    //
    // The output will include not only the list of pages that need to be written but also
    // the new free_list head, the current bump page number, and the new file size if extending is needed
    pub fn commit(&mut self) -> BbnStoreCommitOutput {
        let pending = std::mem::take(&mut self.pending);

        let AllocatorCommitOutput {
            free_list_pages,
            bump,
            extend_file_sz,
            freelist_head,
        } = self.allocator_writer.commit();

        BbnStoreCommitOutput {
            bbn: pending,
            free_list_pages,
            bump,
            extend_file_sz,
            freelist_head,
        }
    }
}

pub struct BbnStoreCommitOutput {
    pub bbn: Vec<BranchNode>,
    pub free_list_pages: Vec<(PageNumber, Box<Page>)>,
    pub bump: PageNumber,
    pub extend_file_sz: Option<usize>,
    pub freelist_head: PageNumber,
}
