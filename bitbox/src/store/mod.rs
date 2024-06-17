use rand::Rng;
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
};

pub mod io;

pub const PAGE_SIZE: usize = 4096;
pub type Page = [u8; PAGE_SIZE];

/// The Store is an on disk array of [`crate::node_pages_map::Page`]
pub struct Store {
    store_file: File,
    // the number of pages to add to a page number to find its real location in the file,
    // taking account of the meta page and meta byte pages.
    data_page_offset: u64,
    // the salt for hashes in this store.
    salt: [u8; 32],
}

impl Store {
    /// Create a new Store given the StoreOptions. Returns a handle to the store file plus the
    /// loaded meta-bits.
    pub fn open(path: PathBuf) -> anyhow::Result<(Self, Vec<u8>)> {
        let mut store_file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)?;

        let mut meta_page_buf = [0u8; PAGE_SIZE];
        store_file.read_exact(&mut meta_page_buf)?;

        let meta_page = MetaPage::from_page(&meta_page_buf[..]);

        if store_file.metadata()?.len() != meta_page.expected_file_len() {
            anyhow::bail!("Store corrupted; unexpected file length");
        }

        let mut meta_bytes = vec![0u8; meta_page.num_meta_byte_pages() * PAGE_SIZE];
        store_file.read_exact(&mut meta_bytes)?;

        let data_page_offset = 1 + meta_page.num_meta_byte_pages() as u64;

        Ok((
            Store {
                store_file,
                data_page_offset,
                salt: meta_page.salt,
            },
            meta_bytes,
        ))
    }

    pub fn data_page_offset(&self) -> u64 {
        self.data_page_offset
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
    salt: [u8; 32],
}

impl MetaPage {
    fn to_page(&self) -> [u8; PAGE_SIZE] {
        let mut page = [0u8; PAGE_SIZE];
        page[0..4].copy_from_slice(&self.version.to_le_bytes());
        page[4..12].copy_from_slice(&self.num_pages.to_le_bytes());
        page[12..44].copy_from_slice(&self.salt);
        page
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
        let mut salt = [0u8; 32];
        salt.copy_from_slice(&page[12..44]);
        MetaPage {
            version,
            num_pages,
            salt,
        }
    }
}

/// Creates the store file. Fails if store file already exists.
///
/// Generates a random salt, lays out the meta page, and fills the file with zeroes.
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
        salt: {
            let mut salt = [0u8; 32];
            rand::thread_rng().fill(&mut salt);
            salt
        },
    };

    // number of pages + pages required for meta bits.
    let page_count = meta_page.num_pages() + meta_page.num_meta_byte_pages();

    file.write_all(meta_page.to_page().as_slice())?;

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
