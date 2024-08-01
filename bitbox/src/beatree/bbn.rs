use crate::{
    beatree::{
        allocator::{self, AllocatorCommitOutput, AllocatorReader, AllocatorWriter, PageNumber},
        branch::BranchNode,
    },
    io::{CompleteIo, IoCommand},
    store::Page,
};
use crossbeam_channel::{Receiver, Sender};

use std::{collections::BTreeSet, fs::File};

/// The BbnStoreWriter enables dynamic allocation and release of BBNs.
/// Upon calling commit, it returns a list of encoded pages and BranchNodes that must be written
/// to storage to reflect the BbnStore's state at that moment
pub struct BbnStoreWriter {
    allocator_writer: AllocatorWriter,
    pending: Vec<BranchNode>,
}

/// Initializes a BbnStoreWriter over a possibly already existing File and returns the pages from
/// the freelist to inform the reconstruction process.
pub fn create(
    fd: File,
    free_list_head: Option<PageNumber>,
    bump: PageNumber,
    io_handle_index: usize,
    io_sender: Sender<IoCommand>,
    io_receiver: Receiver<CompleteIo>,
) -> (BbnStoreWriter, BTreeSet<PageNumber>) {
    let allocator_writer = AllocatorWriter::new(
        fd,
        free_list_head,
        bump,
        io_handle_index,
        io_sender,
        io_receiver,
    );
    let freelist = allocator_writer.free_list().get_set();

    (
        BbnStoreWriter {
            allocator_writer,
            pending: vec![],
        },
        freelist,
    )
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
    pub extend_file_sz: Option<u64>,
    pub freelist_head: PageNumber,
}
