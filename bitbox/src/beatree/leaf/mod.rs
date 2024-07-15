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

use crate::store::Page;

mod free_list;
pub mod store;

pub struct LeafPage {
    pub inner: Box<Page>,
}

pub struct FreeListPage {
    pub inner: Box<Page>,
}

/// The number of a page, either a Leaf or a Free List page, in the LeafStore
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageNumber(u32);

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
