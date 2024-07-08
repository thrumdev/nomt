use rand::Rng;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    ops::{Deref, DerefMut},
    os::{
        fd::{AsRawFd, RawFd},
        unix::fs::OpenOptionsExt,
    },
    path::PathBuf,
};

use crate::meta_map::MetaMap;

pub mod io;

pub const PAGE_SIZE: usize = 4096;

#[derive(Clone)]
#[repr(align(4096))]
pub struct Page([u8; PAGE_SIZE]);

impl Deref for Page {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Page {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Page {
    pub fn zeroed() -> Self {
        Self([0; PAGE_SIZE])
    }
}

/// The Store is an on disk array of [`Page`]
pub struct Store {
    store_file: File,
    // the number of pages to add to a page number to find its real location in the file,
    // taking account of the meta page and meta byte pages.
    data_page_offset: u64,
    // the seed for hashes in this store.
    seed: [u8; 32],
}

impl Store {
    /// Create a new Store given the StoreOptions. Returns a handle to the store file plus the
    /// loaded meta-bits.
    pub fn open(path: PathBuf) -> anyhow::Result<(Self, MetaPage, MetaMap)> {
        let mut store_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)?;

        let mut meta_page_buf = Page::zeroed();
        store_file.read_exact(&mut meta_page_buf)?;

        let meta_page = MetaPage::from_page(&meta_page_buf[..]);

        if store_file.metadata()?.len() != meta_page.expected_file_len() {
            anyhow::bail!("Store corrupted; unexpected file length");
        }

        // Read the extra meta pages. Note that due to O_DIRECT we are only allowed to read into
        // aligned buffers. You cannot really conjure a Vec from raw parts because the Vec doesn't
        // store alignment but deducts it from T before deallocation and the allocator might not
        // like that.
        //
        // We could try to be smart about this sure, but there is always a risk to outsmart yourself
        // pooping your own pants on the way.
        let mut extra_meta_pages: Vec<Page> = Vec::with_capacity(meta_page.num_meta_byte_pages());
        for _ in 0..meta_page.num_meta_byte_pages() {
            let mut buf = Page::zeroed();
            store_file.read_exact(&mut buf)?;
            extra_meta_pages.push(buf);
        }
        let mut meta_bytes = Vec::with_capacity(meta_page.num_meta_byte_pages() * PAGE_SIZE);
        for extra_meta_page in extra_meta_pages {
            meta_bytes.extend_from_slice(&*extra_meta_page);
        }

        let data_page_offset = 1 + meta_page.num_meta_byte_pages() as u64;

        let num_pages = meta_page.num_pages as usize;
        Ok((
            Store {
                store_file,
                data_page_offset,
                seed: meta_page.seed,
            },
            meta_page,
            MetaMap::from_bytes(meta_bytes, num_pages),
        ))
    }

    /// Get the hash seed.
    pub fn seed(&self) -> [u8; 32] {
        self.seed
    }

    /// Get the data page offset.
    #[allow(unused)]
    pub fn data_page_offset(&self) -> u64 {
        self.data_page_offset
    }

    /// Returns the page number of the `ix`th item in the data section of the store.
    pub fn data_page_index(&self, ix: u64) -> u64 {
        self.data_page_offset + ix
    }

    /// Returns the page number of the `ix`th item in the meta bytes section of the store.
    pub fn meta_bytes_index(&self, ix: u64) -> u64 {
        1 + ix
    }

    pub fn store_fd(&self) -> RawFd {
        self.store_file.as_raw_fd()
    }
}

const CURRENT_VERSION: u32 = 1;

pub struct MetaPage {
    version: u32,
    // TODO:
    // storing this is unnecessary as it can be reversed engineered from the length of the file
    // x = num_pages, y = file len in pages
    // 1 + x + floor((x + 4095)/4096) = y
    // 4096x + x + 4095 = 4096(y - 1)
    // 4097x = 4096(y - 1) - 4095
    // x = ceil((4096/4097)(y-1) - (4095/4097))
    num_pages: u64,
    seed: [u8; 32],
    sequence_number: u64,
}

impl MetaPage {
    pub fn to_page(&self) -> Page {
        let mut page = Page::zeroed();
        page[0..4].copy_from_slice(&self.version.to_le_bytes());
        page[4..12].copy_from_slice(&self.num_pages.to_le_bytes());
        page[12..44].copy_from_slice(&self.seed);
        page[44..52].copy_from_slice(&self.sequence_number.to_le_bytes());
        page
    }

    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    pub fn set_sequence_number(&mut self, s: u64) {
        self.sequence_number = s;
    }

    fn num_pages(&self) -> usize {
        self.num_pages as usize
    }

    fn expected_file_len(&self) -> u64 {
        (1 + self.num_meta_byte_pages() as u64 + self.num_pages) * PAGE_SIZE as u64
    }

    fn num_meta_byte_pages(&self) -> usize {
        ((self.num_pages + 4095) / PAGE_SIZE as u64) as usize
    }

    fn from_page(page: &[u8]) -> Self {
        assert_eq!(page.len(), PAGE_SIZE);

        let version = {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&page[..4]);
            u32::from_le_bytes(buf)
        };
        let num_pages = {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&page[4..12]);
            u64::from_le_bytes(buf)
        };
        assert_eq!(
            version, CURRENT_VERSION,
            "Unsupported store version {}",
            version
        );
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&page[12..44]);
        let sequence_number = {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&page[44..52]);
            u64::from_le_bytes(buf)
        };
        MetaPage {
            version,
            num_pages,
            seed,
            sequence_number,
        }
    }
}

/// Creates the store file. Fails if store file already exists.
///
/// Generates a random seed, lays out the meta page, and fills the file with zeroes.
pub fn create(path: PathBuf, num_pages: usize) -> std::io::Result<()> {
    const WRITE_BATCH_SIZE: usize = PAGE_SIZE; // 16MB

    let start = std::time::Instant::now();
    if path.exists() {
        println!("Path {} already exists.", path.display());
        return Ok(());
    }
    let mut file = OpenOptions::new().append(true).create(true).open(path)?;
    let meta_page = MetaPage {
        version: CURRENT_VERSION,
        num_pages: num_pages as u64,
        seed: {
            let mut seed = [0u8; 32];
            rand::thread_rng().fill(&mut seed);
            seed
        },
        sequence_number: 0,
    };

    // number of pages + pages required for meta bits.
    let page_count = meta_page.num_pages() + meta_page.num_meta_byte_pages();

    file.write_all(meta_page.to_page().0.as_slice())?;

    let mut pages_remaining = page_count;
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
        page_count + 1,
        start.elapsed().as_millis()
    );
    Ok(())
}
