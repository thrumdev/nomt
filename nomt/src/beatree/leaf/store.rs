use crate::{
    beatree::{
        allocator::{AllocatorCommitOutput, AllocatorReader, AllocatorWriter, PageNumber},
        leaf::node::LeafNode,
    },
    io::{IoCommand, IoHandle, IoPool, Page},
};

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

/// creates a pair of LeafStoreReader and LeafStoreWriter over an already existing File.
pub fn open(
    fd: File,
    free_list_head: Option<PageNumber>,
    bump: PageNumber,
    io_pool: &IoPool,
) -> (LeafStoreReader, LeafStoreReader, LeafStoreWriter) {
    let allocator_reader_shared = AllocatorReader::new(
        fd.try_clone().expect("failed to clone file"),
        io_pool.make_handle(),
    );

    let allocator_reader_sync = AllocatorReader::new(
        fd.try_clone().expect("failed to clone file"),
        io_pool.make_handle(),
    );

    let allocator_writer = AllocatorWriter::open(fd, free_list_head, bump);

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
    pub fn query(&self, pn: PageNumber) -> Box<Page> {
        self.allocator_reader.query(pn)
    }

    pub fn io_handle(&self) -> &IoHandle {
        self.allocator_reader.io_handle()
    }

    /// Create an I/O command for querying a page by number.
    pub fn io_command(&self, pn: PageNumber, user_data: u64) -> IoCommand {
        self.allocator_reader.io_command(pn, user_data)
    }
}

impl LeafStoreWriter {
    /// Preallocate a page number.
    pub fn preallocate(&mut self) -> PageNumber {
        self.allocator_writer.allocate()
    }

    /// Write a leaf node, allocating a page number.
    pub fn write(&mut self, leaf_page: LeafNode) -> PageNumber {
        let pn = self.allocator_writer.allocate();
        self.write_preallocated(pn, leaf_page.inner);
        pn
    }

    /// Write a page under a preallocated page number.
    pub fn write_preallocated(&mut self, pn: PageNumber, page: Box<Page>) {
        self.pending.push((pn, page));
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
