use anyhow::bail;

use crate::{
    io::{self, PAGE_SIZE},
    page_diff::PageDiff,
};
use std::{fs::File, io::Seek, sync::Arc};

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

const WAL_ENTRY_TAG_END: u8 = 0;
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
        page_diff: &PageDiff,
        changed: impl Iterator<Item = [u8; 32]>,
        bucket_index: u64,
    ) {
        unsafe {
            // SAFETY: Those do not overlap with the mmap.
            self.write_byte(WAL_ENTRY_TAG_UPDATE);
            self.write(&page_id);
            self.write(&page_diff.as_bytes());
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
        self.write_byte(WAL_ENTRY_TAG_END);

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

#[derive(Debug, PartialEq, Eq)]
pub enum WalEntry {
    Update {
        /// The unique identifier of the page being updated.
        page_id: [u8; 32],
        /// A bitmap where each bit indicates whether the node at the corresponding index was
        /// changed by this update.
        page_diff: PageDiff,
        /// Nodes that were changed by this update. The length of this array must be consistent with
        /// the number of ones in `page_diff`.
        changed_nodes: Vec<[u8; 32]>,
        /// The bucket index which is being updated.
        bucket: u64,
    },
    Clear {
        /// The bucket index which is being cleared.
        bucket: u64,
    },
}

pub struct WalBlobReader {
    wal: Vec<u8>,
    offset: usize,
}
impl WalBlobReader {
    /// Creates a new WAL blob reader.
    ///
    /// The `wal_fd` is expected to be positioned at the start of the WAL file. The file must be
    /// a multiple of the page size.
    pub(crate) fn new(mut wal_fd: &File) -> anyhow::Result<Self> {
        let stat = wal_fd.metadata()?;
        let file_size = stat.len() as usize;
        if file_size % PAGE_SIZE != 0 {
            anyhow::bail!("WAL file size is not a multiple of the page size");
        }

        wal_fd.seek(std::io::SeekFrom::Start(0))?;

        // Read the entire WAL file into memory. We do it page-by-page because WAL fd is opened
        // with O_DIRECT flag, and that means we need to provide aligned buffers.
        let mut wal = Vec::with_capacity(file_size);
        let mut pn = 0;
        loop {
            let page = match io::read_page(wal_fd, pn) {
                Ok(page) => page,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };
            pn += 1;
            wal.extend_from_slice(&*page);
        }

        Ok(Self { wal, offset: 0 })
    }

    /// Reads the next entry from the WAL file.
    ///
    /// Returns `None` if the end of the file is reached.
    pub fn read_entry(&mut self) -> anyhow::Result<Option<WalEntry>> {
        let entry_tag = self.read_byte()?;
        match entry_tag {
            WAL_ENTRY_TAG_END => Ok(None),
            WAL_ENTRY_TAG_CLEAR => {
                let bucket = self.read_u64()?;
                Ok(Some(WalEntry::Clear { bucket }))
            }
            WAL_ENTRY_TAG_UPDATE => {
                let page_id: [u8; 32] = self.read_buf()?;
                let page_diff: [u8; 16] = self.read_buf()?;
                let page_diff = PageDiff::from_bytes(page_diff)
                    .ok_or_else(|| anyhow::anyhow!("Invalid page diff"))?;

                let changed_count = page_diff.count();
                let mut changed_nodes = Vec::with_capacity(changed_count);
                for _ in 0..changed_count {
                    let node = self.read_buf::<32>()?;
                    changed_nodes.push(node);
                }

                let bucket = self.read_u64()?;

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

    /// Reads a single byte from the WAL file.
    fn read_byte(&mut self) -> anyhow::Result<u8> {
        if self.offset >= self.wal.len() {
            bail!("Unexpected end of WAL file");
        }
        let byte = self.wal[self.offset];
        self.offset += 1;
        Ok(byte)
    }

    /// Reads a [u8; N] array from the WAL file.
    fn read_buf<const N: usize>(&mut self) -> anyhow::Result<[u8; N]> {
        if self.offset + N > self.wal.len() {
            bail!("Unexpected end of WAL file");
        }
        let array = self.wal[self.offset..self.offset + N]
            .try_into()
            .map_err(|_| anyhow::anyhow!("Failed to read [u8; {N}] from WAL file"))?;
        self.offset += N;
        Ok(array)
    }

    /// Reads a u64 from the WAL file in little-endian format.
    fn read_u64(&mut self) -> anyhow::Result<u64> {
        let buf = self.read_buf::<8>()?;
        Ok(u64::from_le_bytes(buf))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::OpenOptions, io::Write as _, os::unix::fs::OpenOptionsExt as _};

    use super::*;

    #[test]
    fn test_write_read() {
        let tempdir = tempfile::tempdir().unwrap();
        let wal_filename = tempdir.path().join("wal");
        std::fs::create_dir_all(tempdir.path()).unwrap();
        let mut wal_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&wal_filename).unwrap()
        };

        let mut builder = WalBlobBuilder::new();
        builder.write_clear(0);
        builder.write_update(
            [0; 32],
            &PageDiff::from_bytes(hex_literal::hex!(
                "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
            ))
            .unwrap(),
            vec![].into_iter(),
            0,
        );
        builder.write_clear(1);
        builder.write_update(
            [1; 32],
            &PageDiff::from_bytes(hex_literal::hex!(
                "01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
            ))
            .unwrap(),
            vec![[1; 32]].into_iter(),
            1,
        );
        builder.write_update(
            [2; 32],
            &{
                let mut diff = PageDiff::default();
                for i in 0..126 {
                    diff.set_changed(i);
                }
                diff
            },
            (0..126).map(|x| [x; 32]),
            2,
        );
        let (ptr, len) = builder.finalize();
        wal_fd
            .write_all(unsafe { std::slice::from_raw_parts(ptr, len) })
            .unwrap();
        wal_fd.sync_data().unwrap();

        let mut reader = WalBlobReader::new(&wal_fd).unwrap();
        assert_eq!(
            reader.read_entry().unwrap(),
            Some(WalEntry::Clear { bucket: 0 })
        );
        assert_eq!(
            reader.read_entry().unwrap(),
            Some(WalEntry::Update {
                page_id: [0; 32],
                page_diff: PageDiff::default(),
                changed_nodes: vec![],
                bucket: 0,
            })
        );
        assert_eq!(
            reader.read_entry().unwrap(),
            Some(WalEntry::Clear { bucket: 1 })
        );
        assert_eq!(
            reader.read_entry().unwrap(),
            Some(WalEntry::Update {
                page_id: [1; 32],
                page_diff: {
                    let mut diff = PageDiff::default();
                    diff.set_changed(0);
                    diff
                },
                changed_nodes: vec![[1; 32]],
                bucket: 1,
            })
        );
        assert_eq!(
            reader.read_entry().unwrap(),
            Some(WalEntry::Update {
                page_id: [2; 32],
                page_diff: {
                    let mut diff = PageDiff::default();
                    for i in 0..126 {
                        diff.set_changed(i);
                    }
                    diff
                },
                changed_nodes: (0..126).map(|x| [x; 32]).collect(),
                bucket: 2,
            })
        );
        assert_eq!(reader.read_entry().unwrap(), None);
    }
}
