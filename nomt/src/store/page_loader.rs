use crate::bitbox;
use nomt_core::page_id::PageId;

use std::{os::fd::AsRawFd, sync::Arc};

use super::Shared;

pub use bitbox::{PageLoad, PageLoadAdvance, PageLoadCompletion};

pub struct PageLoader {
    pub(super) shared: Arc<Shared>,
    pub(super) inner: bitbox::PageLoader,
}

impl PageLoader {
    /// Create a new page load.
    pub fn start_load(&self, page_id: PageId) -> PageLoad {
        self.inner.start_load(page_id)
    }

    /// Try to advance the state of the given page load. Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns a value indicating the state of the page load.
    /// The user_data is only relevant if `Submitted` is returned, in which case a completion will
    /// arrive with the same user data at some point.
    pub fn try_advance(
        &self,
        load: &mut PageLoad,
        user_data: u64,
    ) -> anyhow::Result<PageLoadAdvance> {
        self.inner
            .try_advance(self.shared.ht_fd.as_raw_fd(), load, user_data)
    }

    /// Advance the state of the given page load, blocking the current thread.
    /// Fails if the I/O pool is down.
    ///
    /// Panics if the page load needs a completion.
    ///
    /// This returns `Ok(true)` if the page request has been submitted and a completion will be
    /// coming. `Ok(false)` means that the page is guaranteed to be fresh.
    pub fn advance(&self, load: &mut PageLoad, user_data: u64) -> anyhow::Result<bool> {
        self.inner
            .advance(self.shared.ht_fd.as_raw_fd(), load, user_data)
    }

    /// Try to receive the next completion, without blocking the current thread.
    ///
    /// Fails if the I/O pool is down or a request caused an I/O error.
    pub fn try_complete(&self) -> anyhow::Result<Option<PageLoadCompletion>> {
        self.inner.try_complete()
    }

    /// Receive the next completion, blocking the current thread.
    ///
    /// Fails if the I/O pool is down or a request caused an I/O error.
    pub fn complete(&self) -> anyhow::Result<PageLoadCompletion> {
        self.inner.complete()
    }

    /// Get the underlying I/O handle.
    pub fn io_handle(&self) -> &crate::io::IoHandle {
        self.inner.io_handle()
    }
}
