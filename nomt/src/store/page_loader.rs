use crate::{bitbox, io::IoHandle};
use nomt_core::page_id::PageId;

pub use bitbox::PageLoad;

pub struct PageLoader {
    pub(super) inner: bitbox::PageLoader,
}

impl PageLoader {
    /// Create a new page load.
    pub fn start_load(&self, page_id: PageId) -> PageLoad {
        self.inner.start_load(page_id)
    }

    /// Advance the state of the given page load, blocking the current thread.
    /// Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns `Ok(true)` if the page request has been submitted and a completion will be
    /// coming. `Ok(false)` means that the page is guaranteed to be fresh.
    pub fn probe(
        &self,
        load: &mut PageLoad,
        io_handle: &IoHandle,
        user_data: u64,
    ) -> anyhow::Result<bool> {
        self.inner.probe(load, io_handle, user_data)
    }
}
