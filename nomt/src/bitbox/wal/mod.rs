use anyhow::bail;

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

const WAL_ENTRY_TAG_CLEAR: u8 = 1;
const WAL_ENTRY_TAG_UPDATE: u8 = 2;

/// A builder for a WAL blob.
pub struct WalBlobBuilder {
    mmap: Arc<Mmap>,
    /// The position at which the next byte will be written. Never reaches `mmap.size`.
    cur: usize,
}

impl WalBlobBuilder {
    pub fn new() -> Self {
        // 128 GiB = 17179869184 bytes.
        //
        // 128 GiB is the maximum size of a single commit in WAL after which we panic. This seems
        // to be enough for now. We should explore making this elastic in the future.
        //
        // Note that here we allocate virtual memory unbacked by physical pages. Those pages will
        // become backed by physical pages on first write to each page.
        let mmap = Mmap::new(17179869184);
        Self {
            mmap: Arc::new(mmap),
            cur: 0,
        }
    }

    pub fn write_clear(&mut self, bucket_index: u64) {
        unsafe {
            self.write_byte(WAL_ENTRY_TAG_CLEAR);
            self.write(&bucket_index.to_le_bytes());
        }
    }

    pub fn write_update(
        &mut self,
        page_id: [u8; 32],
        page_diff: [u8; 16],
        changed: impl Iterator<Item = [u8; 32]>,
        bucket_index: u64,
    ) {
        unsafe {
            // SAFETY: Those do not overlap with the mmap.
            self.write_byte(WAL_ENTRY_TAG_UPDATE);
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
            // SAFETY: This slice trivially does not overlap with the mmap.
            self.write(&[byte]);
        }
    }

    /// # Safety
    ///
    /// The `bytes` mut not overlap with the mmap.
    unsafe fn write(&mut self, bytes: &[u8]) {
        let pos = self.cur;
        let new_cur = self.cur.checked_add(bytes.len()).unwrap();
        if new_cur >= self.mmap.size {
            panic!("WAL blob too large");
        }
        self.cur = new_cur;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mmap.ptr.add(pos), bytes.len());
        }
    }

    /// Finalizes the builder and returns the pointer to the start of the blob and its length.
    ///
    /// This also resets the builder preparing it for a new batch of writes.
    ///
    /// The caller must ensure that the blob is not dropped before the pointer is no longer
    /// used.
    ///
    /// It's possible to overwrite the data in the blob after calling this function so don't keep
    /// the pointer around for too long.
    ///
    /// The pointer is aligned to the page size.
    pub fn finalize(&mut self) -> (*mut u8, usize) {
        let ptr = self.mmap.ptr;
        // round up to the nearest page size.
        let len = (self.cur + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;

        // Note we don't madvise(DONTNEED) or any other tricks for now. If we did, then each time
        // we write a page it would need to be mounted and thus zero filled.
        //
        // The hope is that should there be memory  pressure, the memory would be swapped out as
        // needed.
        let cur = self.cur;
        unsafe {
            // Zero memory from `cur` to the end of the blob (which is `len`).
            let dst = ptr.add(cur);
            let count = len - cur;
            if count > 0 {
                // SAFETY:
                // - `dst` is never null.
                // - `dst` is always naturally aligned because it's a byte.
                // - `cur` is always less than the size of the mmap. Thus `dst + count` never lands
                //   past the end of the mmap.
                // - `0` is a valid value for `u8`.
                std::ptr::write_bytes(dst, 0, count);
            }
        }

        self.cur = 0;
        (ptr, len)
    }
}

unsafe impl Send for WalBlobBuilder {}

#[derive(Debug)]
pub enum WalEntry {
    Update {
        /// The unique identifier of the page being updated.
        page_id: [u8; 32],
        /// A bitmap where each bit indicates whether the node at the corresponding index was
        /// changed by this entry.
        page_diff: [u8; 16],
        /// The length of this array is the same as the count of ones in `page_diff`.
        changed_nodes: Vec<[u8; 32]>,
        /// The bucket index which is being updated.
        bucket: u64,
    },
    Clear {
        /// The bucket index which is being cleared.
        bucket: u64,
    },
}

pub struct WalBlobReader<'a> {
    wal_fd: &'a std::fs::File,
}
impl<'a> WalBlobReader<'a> {
    /// Creates a new WAL blob reader.
    ///
    /// The `wal_fd` is expected to be positioned at the start of the WAL file. The file must be
    /// a multiple of the page size.
    pub(crate) fn new(wal_fd: &'a std::fs::File) -> anyhow::Result<Self> {
        let stat = wal_fd.metadata()?;
        if stat.len() % PAGE_SIZE as u64 != 0 {
            anyhow::bail!("WAL file size is not a multiple of the page size");
        }
        Ok(Self { wal_fd })
    }

    /// Reads the next entry from the WAL file.
    ///
    /// Returns `None` if the end of the file is reached.
    pub fn read_entry(&mut self) -> anyhow::Result<Option<WalEntry>> {
        use crate::io::Page;
        use std::io::Read;

        let mut buffer = Box::new(Page::zeroed());
        self.wal_fd.read_exact(&mut buffer[..])?;

        let entry_tag = buffer[0];
        match entry_tag {
            WAL_ENTRY_TAG_CLEAR => {
                let bucket = u64::from_le_bytes(buffer[1..9].try_into().unwrap());
                Ok(Some(WalEntry::Clear { bucket }))
            }
            WAL_ENTRY_TAG_UPDATE => {
                let page_id: [u8; 32] = buffer[1..33].try_into().unwrap();
                let page_diff: [u8; 16] = buffer[33..49].try_into().unwrap();

                // Calculate the number of changed nodes based on the page_diff
                let changed_count =
                    page_diff.iter().map(|&byte| byte.count_ones()).sum::<u32>() as usize;

                let mut changed_nodes = Vec::with_capacity(changed_count);
                let mut offset = 49;
                for _ in 0..changed_count {
                    changed_nodes.push(buffer[offset..offset + 32].try_into().unwrap());
                    offset += 32;
                }

                let bucket = u64::from_le_bytes(buffer[offset..offset + 8].try_into().unwrap());

                Ok(Some(WalEntry::Update {
                    page_id,
                    page_diff,
                    changed_nodes,
                    bucket,
                }))
            }
            _ => bail!("unknown WAL entry tag: {entry_tag}"),
        }
    }
}
