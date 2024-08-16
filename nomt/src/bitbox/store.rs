use super::meta_map::MetaMap;
use crate::io::{Page, PAGE_SIZE};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, RawFd},
        unix::fs::OpenOptionsExt,
    },
    path::PathBuf,
    sync::Arc,
};

/// The Store is an on disk array of [`Page`]
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    store_file: File,
    // the number of pages to add to a page number to find its real location in the file,
    // taking account of the meta page and meta byte pages.
    data_page_offset: u64,
}

impl Store {
    /// Create a new Store given the StoreOptions. Returns a handle to the store file plus the
    /// loaded meta-bits.
    pub fn open(num_pages: u32, path: PathBuf) -> anyhow::Result<(Self, MetaMap)> {
        let store_path = path.join("ht");
        let mut store_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(store_path)?;

        if store_file.metadata()?.len() != expected_file_len(num_pages) {
            anyhow::bail!("Store corrupted; unexpected file length");
        }

        // Read the extra meta pages. Note that due to O_DIRECT we are only allowed to read into
        // aligned buffers. You cannot really conjure a Vec from raw parts because the Vec doesn't
        // store alignment but deducts it from T before deallocation and the allocator might not
        // like that.
        //
        // We could try to be smart about this sure, but there is always a risk to outsmart yourself
        // pooping your own pants on the way.
        let num_meta_byte_pages = num_meta_byte_pages(num_pages) as usize;
        let mut extra_meta_pages: Vec<Page> =
            Vec::with_capacity(num_meta_byte_pages);
        for _ in 0..num_meta_byte_pages {
            let mut buf = Page::zeroed();
            store_file.read_exact(&mut buf)?;
            extra_meta_pages.push(buf);
        }
        let mut meta_bytes = Vec::with_capacity(num_meta_byte_pages * PAGE_SIZE);
        for extra_meta_page in extra_meta_pages {
            meta_bytes.extend_from_slice(&*extra_meta_page);
        }

        let data_page_offset = 1 + num_meta_byte_pages as u64;

        Ok((
            Store {
                shared: Arc::new(Shared {
                    store_file,
                    data_page_offset,
                }),
            },
            MetaMap::from_bytes(meta_bytes, num_pages as usize),
        ))
    }

    /// Get the data page offset.
    #[allow(unused)]
    pub fn data_page_offset(&self) -> u64 {
        self.shared.data_page_offset
    }

    /// Returns the page number of the `ix`th item in the data section of the store.
    pub fn data_page_index(&self, ix: u64) -> u64 {
        self.shared.data_page_offset + ix
    }

    /// Returns the page number of the `ix`th item in the meta bytes section of the store.
    pub fn meta_bytes_index(&self, ix: u64) -> u64 {
        1 + ix
    }

    pub fn store_fd(&self) -> RawFd {
        self.shared.store_file.as_raw_fd()
    }
}

fn expected_file_len(num_pages: u32) -> u64 {
    (1 + num_meta_byte_pages(num_pages) + num_pages) as u64 * PAGE_SIZE as u64
}

fn num_meta_byte_pages(num_pages: u32) -> u32 {
    (num_pages + 4095) / PAGE_SIZE as u32
}

/// Creates the store file. Fails if store file already exists.
///
/// Generates a random seed, lays out the meta page, and fills the file with zeroes.
pub fn create(path: PathBuf, num_pages: u32) -> std::io::Result<()> {
    const WRITE_BATCH_SIZE: usize = PAGE_SIZE; // 16MB

    let start = std::time::Instant::now();
    let store_path = path.join("ht");
    let mut file = OpenOptions::new().append(true).create(true).open(store_path)?;

    // dummy page + number of pages + pages required for meta bits.
    let page_count = 1 + num_pages + num_meta_byte_pages(num_pages);

    let mut pages_remaining = page_count as usize;
    let zero_buf = vec![0u8; PAGE_SIZE * WRITE_BATCH_SIZE];
    while pages_remaining > 0 {
        let pages_to_write = std::cmp::min(pages_remaining, WRITE_BATCH_SIZE);
        let buf = &zero_buf[0..pages_to_write * PAGE_SIZE];
        file.write_all(buf)?;
        pages_remaining -= pages_to_write;
    }

    file.flush()?;
    file.sync_all()?;
    println!(
        "Created file with {} total pages in {}ms",
        page_count,
        start.elapsed().as_millis()
    );
    Ok(())
}
