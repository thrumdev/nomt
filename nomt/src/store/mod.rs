//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use crate::{
    beatree, bitbox,
    io::{self, IoPool, Page},
    page_diff::PageDiff,
};
use meta::Meta;
use nomt_core::{page_id::PageId, trie::KeyPath};
use parking_lot::Mutex;
use std::{
    fs::{File, OpenOptions},
    os::fd::AsRawFd as _,
    sync::Arc,
};

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt as _;

pub use self::page_loader::{PageLoad, PageLoadAdvance, PageLoadCompletion, PageLoader};
pub use bitbox::BucketIndex;

mod meta;
mod page_loader;
mod writeout;

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
    sync: Arc<Mutex<Sync>>,
}

struct Shared {
    bitbox_num_pages: u32,
    bitbox_seed: [u8; 16],
    panic_on_sync: bool,
    values: beatree::Tree,
    pages: bitbox::DB,
    io_pool: IoPool,
    meta_fd: File,
    ln_fd: File,
    bbn_fd: File,
    ht_fd: File,
    // keep alive.
    #[allow(unused)]
    wal_fd: File,
}

struct Sync {
    sync_seqn: u32,
    wal_blob_builder: bitbox::WalBlobBuilder,
    io_handle: io::IoHandle,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        if !o.path.exists() {
            create(o)?;
        }

        let io_pool = io::start_io_pool(o.io_workers);

        let meta_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&o.path.join("meta"))?
        };

        let ln_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&o.path.join("ln"))?
        };
        let bbn_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&o.path.join("bbn"))?
        };
        let ht_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&o.path.join("ht"))?
        };
        let wal_fd = {
            let options = &mut OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            options.custom_flags(libc::O_DIRECT);
            options.open(&o.path.join("wal"))?
        };

        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(meta_fd.as_raw_fd(), libc::F_NOCACHE, 1);
            libc::fcntl(ln_fd.as_raw_fd(), libc::F_NOCACHE, 1);
            libc::fcntl(bbn_fd.as_raw_fd(), libc::F_NOCACHE, 1);
            libc::fcntl(ht_fd.as_raw_fd(), libc::F_NOCACHE, 1);
            libc::fcntl(wal_fd.as_raw_fd(), libc::F_NOCACHE, 1);
        }

        let meta = meta::Meta::read(&meta_fd)?;
        let values = beatree::Tree::open(
            &io_pool,
            meta.ln_freelist_pn,
            meta.bbn_freelist_pn,
            meta.ln_bump,
            meta.bbn_bump,
            &bbn_fd,
            &ln_fd,
        )?;
        let pages = bitbox::DB::open(meta.bitbox_num_pages, meta.bitbox_seed, &ht_fd, &wal_fd)?;
        let io_handle = io_pool.make_handle();
        let wal_blob_builder = bitbox::WalBlobBuilder::new();
        Ok(Self {
            shared: Arc::new(Shared {
                bitbox_num_pages: meta.bitbox_num_pages,
                bitbox_seed: meta.bitbox_seed,
                panic_on_sync: o.panic_on_sync,
                values,
                pages,
                meta_fd,
                ln_fd,
                bbn_fd,
                ht_fd,
                wal_fd,
                io_pool,
            }),
            sync: Arc::new(Mutex::new(Sync {
                sync_seqn: meta.sync_seqn,
                io_handle,
                wal_blob_builder,
            })),
        })
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.shared.values.lookup(key))
    }

    /// Loads the given page, blocking the current thread.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<(Box<Page>, BucketIndex)>> {
        let page_loader = self.page_loader();
        let mut page_load = page_loader.start_load(page_id);
        loop {
            if !page_loader.advance(&mut page_load, 0)? {
                return Ok(None);
            }

            let completion = page_loader.complete()?;
            assert_eq!(completion.user_data(), 0);
            if let Some(res) = completion.apply_to(&mut page_load) {
                return Ok(Some(res));
            }
        }
    }

    /// Creates a new [`PageLoader`].
    pub fn page_loader(&self) -> PageLoader {
        let page_loader = bitbox::PageLoader::new(&self.shared.pages, self.io_pool().make_handle());
        PageLoader {
            shared: self.shared.clone(),
            inner: page_loader,
        }
    }

    /// Access the underlying IoPool.
    pub fn io_pool(&self) -> &IoPool {
        &self.shared.io_pool
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        Transaction {
            batch: Vec::new(),
            new_pages: vec![],
            bucket_allocator: self.shared.pages.bucket_allocator(),
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, tx: Transaction) -> anyhow::Result<()> {
        let mut sync = self.sync.lock();
        sync.sync_seqn += 1;

        // store the sequence number of this sync.
        let sync_seqn = sync.sync_seqn;

        self.shared.values.commit(tx.batch);

        let bitbox_writeout_data = self
            .shared
            .pages
            .prepare_sync(tx.new_pages, &mut sync.wal_blob_builder)?;
        let beatree_writeout_data = self.shared.values.prepare_sync();

        let new_meta = Meta {
            ln_freelist_pn: beatree_writeout_data.ln_freelist_pn,
            ln_bump: beatree_writeout_data.ln_bump,
            bbn_freelist_pn: beatree_writeout_data.bbn_freelist_pn,
            bbn_bump: beatree_writeout_data.bbn_bump,
            sync_seqn,
            bitbox_num_pages: self.shared.bitbox_num_pages,
            bitbox_seed: self.shared.bitbox_seed,
        };

        let bitbox_wal_blob = sync.wal_blob_builder.finalize();
        writeout::run(
            sync.io_handle.clone(),
            self.shared.wal_fd.as_raw_fd(),
            self.shared.bbn_fd.as_raw_fd(),
            self.shared.ln_fd.as_raw_fd(),
            self.shared.ht_fd.as_raw_fd(),
            self.shared.meta_fd.as_raw_fd(),
            bitbox_wal_blob,
            beatree_writeout_data.bbn,
            beatree_writeout_data.bbn_freelist_pages,
            beatree_writeout_data.bbn_extend_file_sz,
            beatree_writeout_data.ln,
            beatree_writeout_data.ln_extend_file_sz,
            bitbox_writeout_data.ht_pages,
            new_meta,
            self.shared.panic_on_sync,
        );

        self.shared.values.finish_sync(
            beatree_writeout_data.bbn_index,
            beatree_writeout_data.obsolete_branches,
        );

        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    batch: Vec<(KeyPath, Option<Vec<u8>>)>,
    bucket_allocator: bitbox::BucketAllocator,
    new_pages: Vec<(PageId, BucketIndex, Option<(Box<Page>, PageDiff)>)>,
}

impl Transaction {
    /// Write a value to flat storage.
    pub fn write_value(&mut self, path: KeyPath, value: Option<&[u8]>) {
        self.batch.push((path, value.map(|v| v.to_vec())))
    }

    /// Write a page to storage in its entirety.
    pub fn write_page(
        &mut self,
        page_id: PageId,
        bucket: Option<BucketIndex>,
        page: &Page,
        page_diff: PageDiff,
    ) -> BucketIndex {
        let bucket_index =
            bucket.unwrap_or_else(|| self.bucket_allocator.allocate(page_id.clone()));
        self.new_pages.push((
            page_id,
            bucket_index,
            Some((Box::new(page.clone()), page_diff)),
        ));
        bucket_index
    }
    /// Delete a page from storage.
    pub fn delete_page(&mut self, page_id: PageId, bucket: BucketIndex) {
        self.bucket_allocator.free(bucket);
        self.new_pages.push((page_id, bucket, None));
    }
}

fn create(o: &crate::Options) -> anyhow::Result<()> {
    use std::io::Write as _;

    // Create the directory and its parent directories.
    std::fs::create_dir_all(&o.path)?;

    let mut meta_fd = std::fs::File::create(o.path.join("meta"))?;
    let mut buf = [0u8; 4096];
    Meta {
        ln_freelist_pn: 0,
        ln_bump: 1,
        bbn_freelist_pn: 0,
        bbn_bump: 1,
        sync_seqn: 0,
        bitbox_num_pages: o.bitbox_num_pages,
        bitbox_seed: o.bitbox_seed,
    }
    .encode_to(&mut buf[0..40]);
    meta_fd.write_all(&buf)?;
    meta_fd.sync_all()?;
    drop(meta_fd);

    bitbox::create(o.path.clone(), o.bitbox_num_pages)?;
    beatree::create(&o.path)?;

    // As the last step, sync the directory.
    std::fs::File::open(&o.path)?.sync_all()?;
    Ok(())
}
