// The `LeafStore` struct manages leaves. It's responsible for management (allocation and
// deallocation) and querying the LNs by their LNID.
//
// It maintains an in-memory copy of the freelist to facilitate the page management. The allocation
// is performed in LIFO order. The allocations are performed in batches to amortize the IO for the
// freelist and metadata updates (growing the file in case freelist is empty).
//
// The leaf store doesn't perform caching. When querying the leaf store returns a handle to a page.
// As soon as the handle is dropped, the data becomes inaccessible and another disk roundtrip would
// be required to access the data again.

use std::fs::File;

/// The page number of a leaf page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafPn(u32);

impl LeafPn {
    pub fn is_nil(&self) -> bool {
        self.0 == 0
    }
}

pub struct LeafPage {}

/// Handle. Cheap to clone.
#[derive(Clone)]
pub struct LeafStore {}

impl LeafStore {
    pub fn create(fd: File, freelist_pn: LeafPn) -> LeafStore {
        todo!()
    }

    /// Returns the leaf page with the specified page number.
    pub fn query(&self, pn: LeafPn) -> LeafPage {
        todo!()
    }
    
    pub fn start_tx(&self) -> LeafStoreTx {
        todo!()
    }
}

pub struct LeafStoreTx {
}

impl LeafStoreTx {
    pub fn allocate(&mut self, page: LeafPage) -> LeafPn {
        let _ = page;
        todo!()
    }

    pub fn release(&mut self, id: LeafPn) {
        let _ = id;
        todo!()
    }

    pub fn query(&self, id: LeafPn) -> LeafPage {
        let _ = id;
        todo!()
    }
}
