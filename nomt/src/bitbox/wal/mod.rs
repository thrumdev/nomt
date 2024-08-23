use crate::io::PAGE_SIZE;
use std::sync::Arc;

struct Mmap {
    ptr: *mut u8,
    size: usize,
}

impl Mmap {
    fn new(size: usize) -> Self {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        } as *mut u8;
        if ptr == libc::MAP_FAILED as *mut u8 {
            panic!("mmap failed");
        }
        Self { ptr, size }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
    }
}

pub struct WalBlobBuilder {
    mmap: Arc<Mmap>,
    cur: usize,
}

impl WalBlobBuilder {
    pub fn new() -> Self {
        // 128 GiB / 4 KiB = 33554432.
        //
        // 128 GiB is the maximum size of a single commit in WAL after which we panic. This seems
        // to be enough for now. We should explore making this elastic in the future.
        let mmap = Mmap::new(33554432);
        Self {
            mmap: Arc::new(mmap),
            cur: 0,
        }
    }

    pub fn write_clear(&mut self, bucket_index: u64) {
        unsafe {
            self.write_byte(0);
            self.write(&bucket_index.to_le_bytes());
        }
    }

    pub fn write_update(
        &mut self,
        page_id: [u8; 32],
        page_diff: [u8; 16],
        changed: Vec<[u8; 32]>,
        bucket_index: u64,
    ) {
        unsafe {
            // SAFETY: Those do not overlap with the mmap. Potentially, the `changed` slices could
            // overlap with the mmap, but I would argue expecting that (as well as doing that) would
            // be schizo.
            self.write_byte(1);
            self.write(&page_id);
            self.write(&page_diff);
            for changed in changed {
                self.write(&changed);
            }
            self.write(&bucket_index.to_le_bytes());
        }
    }

    fn write_byte(&mut self, byte: u8) {
        unsafe {
            self.write(&[byte]);
        }
    }

    /// # Safety
    ///
    /// The `bytes` mut not overlap with the mmap.
    unsafe fn write(&mut self, bytes: &[u8]) {
        let pos = self.cur;
        let new_cur = self.cur.checked_add(bytes.len()).unwrap();
        if new_cur > self.mmap.size {
            panic!("WAL blob too large");
        }
        self.cur = new_cur;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mmap.ptr.add(pos), bytes.len());
        }
    }

    pub fn reset(&mut self) {
        // Note we don't madvise(DONTNEED) or any other tricks for now. If we did, then each time
        // we write a page it would need to be mounted and thus zero filled.
        //
        // The hope is that should there be memory  pressure, the memory would be swapped out as
        // needed.
        self.cur = 0;
    }

    /// Returns a pointer to the start of the blob.
    ///
    /// The caller must ensure that the blob is not dropped before the pointer is no longer
    /// used.
    ///
    /// It's possible to overwrite the data in the blob after calling this function so don't keep
    /// the pointer around for too long.
    ///
    /// The pointer is aligned to the page size.
    pub fn ptr(&self) -> *mut u8 {
        self.mmap.ptr
    }

    /// The length of the blob. The size is a multiple of the page size.
    pub fn len(&self) -> usize {
        // round up to the next page size.
        (self.cur + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE
    }
}

unsafe impl Send for WalBlobBuilder {}
