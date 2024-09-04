
use super::PAGE_SIZE;
use parking_lot::RwLock;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

// Region is 256 MiB. The choice is mostly arbitrary, but:
//
// 1. it's big enough so that we don't have to allocate too often.
// 2. it's a multiple of 2 MiB which is the size of huge-page on x86-64 and aarch64.
// 3. it fits 65k 4 KiB pages which requires 2 bytes to address making it nice round number.
//
const REGION_SLOT_BITS: u32 = 16;
const REGION_SLOT_MASK: u32 = (1 << REGION_SLOT_BITS) - 1;
const REGION_SIZE: usize = (1 << REGION_SLOT_BITS) * PAGE_SIZE;

#[derive(Clone, Copy)]
struct PageIndex(u32);

impl PageIndex {
    /// Extracts the region index from the reference.
    fn region(&self) -> usize {
        (self.0 >> 16) as usize
    }

    /// Extracts the slot index within the region.
    fn slot_index(&self) -> usize {
        (self.0 & REGION_SLOT_MASK) as usize
    }
}

/// A page reference to the pool.
#[derive(Clone)]
pub struct Page(PageIndex);

impl Page {
    /// Returns a pointer to the page.
    pub fn as_ptr(&self, pool: &PagePool) -> *const u8 {
        pool.data_ptr(self.0) as *const u8
    }

    /// Returns a mutable pointer to the page.
    pub fn as_mut_ptr(&self, pool: &PagePool) -> *mut u8 {
        pool.data_ptr(self.0)
    }

    /// This is a convenience function that uses [`std::slice::from_raw_parts_mut`] to create a
    /// mutable slice.
    ///
    /// # Safety
    ///
    /// The caller is responsible for making sure:
    ///
    /// 1. that the page is not freed,
    /// 2. that the [`PagePool`] is the same that was used to allocate the page.
    /// 3. that the [`PagePool`] is not dropped while the slice is used.
    /// 4. that there is only a single mutable slice into the page at any given time.
    pub unsafe fn as_mut_slice(&self, pool: &PagePool) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(pool), PAGE_SIZE)
    }
}

/// Provides a managed version of a [`Page`] by wrapping it and it's [`PagePool`].
///
/// Unlike [`Page`], this type handles deallocation for you upon dropping. It also provides a safe
/// way to access the contents of the page. However, the price for this convenience is that it is
/// heavier than the bare page type and it doesn't allow cloning.
pub struct FatPage {
    page_pool: PagePool,
    page: Page,
}

impl FatPage {
    /// See [`Page::as_ptr`].
    pub fn as_ptr(&self, pool: &PagePool) -> *const u8 {
        self.page.as_ptr(pool)
    }

    /// See [`Page::as_mut_ptr`].
    pub fn as_mut_ptr(&self, pool: &PagePool) -> *mut u8 {
        self.page.as_mut_ptr(pool)
    }
}

impl Deref for FatPage {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { self.page.as_mut_slice(&self.page_pool) }
    }
}

impl DerefMut for FatPage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.page.as_mut_slice(&self.page_pool) }
    }
}

impl Drop for FatPage {
    fn drop(&mut self) {
        self.page_pool.dealloc(self.page.clone());
    }
}

/// [`PagePool`] is an efficient allocator for pages used in IO operations.
///
/// It allows for efficient allocation and deallocation of pages.
#[derive(Clone)]
pub struct PagePool {
    inner: Arc<Inner>,
}

struct Inner {
    regions: RwLock<Vec<*mut libc::c_void>>,
    freelist: RwLock<Vec<Page>>,
}

impl PagePool {
    /// Creates a new empty page pool.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                regions: RwLock::new(vec![]),
                freelist: RwLock::new(vec![]),
            }),
        }
    }

    /// Allocates a new [`FatPage`].
    pub fn alloc_fat_page(&self) -> FatPage {
        let page = self.alloc_zeroed();
        FatPage {
            page_pool: self.clone(),
            page,
        }
    }

    /// Allocates a new [`Page`] and fills it with zeroes.
    pub fn alloc_zeroed(&self) -> Page {
        let page = {
            let mut freelist = self.inner.freelist.write();
            if freelist.is_empty() {
                self.grow(&mut *freelist)
            } else {
                freelist.pop().unwrap()
            }
        };
        unsafe {
            page.as_mut_slice(self).fill(0);
        }
        page
    }

    /// Deallocates a [`Page`].
    pub fn dealloc(&self, page: Page) {
        self.inner.freelist.write().push(page);
    }

    fn grow(&self, freelist: &mut Vec<Page>) -> Page {
        assert!(freelist.is_empty());
        let region_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                REGION_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                /* fd */ -1,
                /* offset */ 0,
            )
        };
        if region_ptr == libc::MAP_FAILED {
            panic!("Failed to allocate memory");
        }
        let mut regions = self.inner.regions.write();
        regions.push(region_ptr);

        let region: u32 = ((regions.len() - 1) as u32) << 16;
        for slot in 0..=REGION_SLOT_MASK {
            freelist.push(Page(PageIndex(slot | region)));
        }
        // unwrap: we know that the freelist is not empty
        freelist.pop().unwrap()
    }

    fn data_ptr(&self, page_index: PageIndex) -> *mut u8 {
        let region_ptr = self.inner.regions.read()[page_index.region()];
        unsafe { region_ptr.add(page_index.slot_index() * PAGE_SIZE) as *mut u8 }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let mut regions = self.regions.write();
        for region_ptr in regions.drain(..) {
            unsafe {
                libc::munmap(region_ptr as *mut libc::c_void, REGION_SIZE);
            }
        }
    }
}

unsafe impl Send for PagePool {}
unsafe impl Sync for PagePool {}
