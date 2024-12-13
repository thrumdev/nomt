//! The store module abstracts the storage layer of Nomt.
//!
//! It provides storage facilities for the binary trie pages based on a hash-table (bitbox) and the
//! b-tree key-value storage (beatree).

use crate::{
    beatree, bitbox,
    io::{self, page_pool::FatPage, IoPool, PagePool},
    merkle,
    page_cache::PageCache,
    page_diff::PageDiff,
    rollback::Rollback,
    ValueHasher,
};
use meta::Meta;
use nomt_core::{page_id::PageId, trie::KeyPath};
use parking_lot::Mutex;
use std::{
    fs::{File, OpenOptions},
    sync::Arc,
};

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt as _;

pub use self::page_loader::{PageLoad, PageLoader};
pub use bitbox::BucketIndex;

mod flock;
mod meta;
mod page_loader;
mod sync;

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
    sync: Arc<Mutex<sync::Sync>>,
}

struct Shared {
    values: beatree::Tree,
    pages: bitbox::DB,
    rollback: Option<Rollback>,
    page_pool: PagePool,
    io_pool: IoPool,
    meta_fd: File,
    #[allow(unused)]
    flock: flock::Flock,

    // Retained for the lifetime of the store.
    _db_dir_fd: Arc<File>,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options, page_pool: PagePool) -> anyhow::Result<Self> {
        let db_dir_fd = if !o.path.exists() {
            // NB: note TOCTOU here. Deemed acceptable for this case.
            create(&page_pool, &o)?
        } else {
            let mut options = OpenOptions::new();
            options.read(true);
            options.open(&o.path)?
        };
        let db_dir_fd = Arc::new(db_dir_fd);
        let flock = flock::Flock::lock(&o.path, ".lock")?;

        cfg_if::cfg_if! {
            if #[cfg(target_os = "linux")] {
                let is_tmpfs = crate::sys::linux::tmpfs_check(&db_dir_fd);
                let iopoll = !is_tmpfs;
            } else {
                let iopoll = true;
            }
        }

        let io_pool = io::start_io_pool(o.io_workers, iopoll, page_pool.clone());

        let meta_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            if !is_tmpfs {
                options.custom_flags(libc::O_DIRECT);
            }
            options.open(&o.path.join("meta"))?
        };

        let ln_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            if !is_tmpfs {
                options.custom_flags(libc::O_DIRECT);
            }
            Arc::new(options.open(&o.path.join("ln"))?)
        };
        let bbn_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            if !is_tmpfs {
                options.custom_flags(libc::O_DIRECT);
            }
            Arc::new(options.open(&o.path.join("bbn"))?)
        };
        let ht_fd = {
            let mut options = OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            if !is_tmpfs {
                options.custom_flags(libc::O_DIRECT);
            }
            options.open(&o.path.join("ht"))?
        };
        let wal_fd = {
            let options = &mut OpenOptions::new();
            options.read(true).write(true);
            #[cfg(target_os = "linux")]
            if !is_tmpfs {
                options.custom_flags(libc::O_DIRECT);
            }
            options.open(&o.path.join("wal"))?
        };

        #[cfg(target_os = "macos")]
        {
            use std::os::fd::AsRawFd as _;
            unsafe {
                libc::fcntl(meta_fd.as_raw_fd(), libc::F_NOCACHE, 1);
                libc::fcntl(ln_fd.as_raw_fd(), libc::F_NOCACHE, 1);
                libc::fcntl(bbn_fd.as_raw_fd(), libc::F_NOCACHE, 1);
                libc::fcntl(ht_fd.as_raw_fd(), libc::F_NOCACHE, 1);
                libc::fcntl(wal_fd.as_raw_fd(), libc::F_NOCACHE, 1);
            }
        }

        let meta = meta::Meta::read(&page_pool, &meta_fd)?;
        meta.validate()?;
        let values = beatree::Tree::open(
            page_pool.clone(),
            &io_pool,
            meta.ln_freelist_pn,
            meta.bbn_freelist_pn,
            meta.ln_bump,
            meta.bbn_bump,
            bbn_fd,
            ln_fd,
            o.commit_concurrency,
        )?;
        let pages = bitbox::DB::open(
            meta.bitbox_num_pages,
            meta.bitbox_seed,
            page_pool.clone(),
            ht_fd,
            wal_fd,
        )?;
        let rollback = o
            .rollback
            .then(|| {
                Rollback::read(
                    o.max_rollback_log_len,
                    o.rollback_tp_size,
                    o.path.clone(),
                    Arc::clone(&db_dir_fd),
                    meta.rollback_start_live,
                    meta.rollback_end_live,
                )
            })
            .transpose()?;
        Ok(Self {
            sync: Arc::new(Mutex::new(sync::Sync::new(
                meta.sync_seqn,
                meta.bitbox_num_pages,
                meta.bitbox_seed,
                o.panic_on_sync,
            ))),
            shared: Arc::new(Shared {
                rollback,
                page_pool,
                values,
                pages,
                io_pool,
                _db_dir_fd: db_dir_fd,
                meta_fd,
                flock,
            }),
        })
    }

    /// Returns a handle to the rollback object. `None` if the rollback feature is not enabled.
    pub fn rollback(&self) -> Option<&Rollback> {
        self.shared.rollback.as_ref()
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.shared.values.lookup(key))
    }

    /// Loads the given page, blocking the current thread.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<(FatPage, BucketIndex)>> {
        let page_loader = self.page_loader();
        let io_handle = self.io_pool().make_handle();
        let mut page_load = page_loader.start_load(page_id);
        loop {
            if !page_loader.probe(&mut page_load, &io_handle, 0)? {
                return Ok(None);
            }

            let completion = io_handle.recv()?;
            completion.result?;
            assert_eq!(completion.command.user_data, 0);

            // UNWRAP: page loader always submits a `Read` command that yields a fat page.
            let page = completion.command.kind.unwrap_buf();

            if let Some(res) = page_load.try_complete(page) {
                return Ok(Some(res));
            }
        }
    }

    /// Creates a new [`beatree::ReadTransaction`]. `sync` will be blocked until this is dropped.
    pub fn read_transaction(&self) -> beatree::ReadTransaction {
        self.shared.values.read_transaction()
    }

    /// Creates a new [`PageLoader`].
    pub fn page_loader(&self) -> PageLoader {
        let page_loader = bitbox::PageLoader::new(&self.shared.pages);
        PageLoader { inner: page_loader }
    }

    /// Access the underlying IoPool.
    pub fn io_pool(&self) -> &IoPool {
        &self.shared.io_pool
    }

    /// Create a new raw value transaction to be applied against this database.
    pub fn new_value_tx(&self) -> ValueTransaction {
        ValueTransaction { batch: Vec::new() }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(
        &self,
        value_tx: ValueTransaction,
        page_cache: PageCache,
        page_diffs: merkle::PageDiffs,
    ) -> anyhow::Result<()> {
        let mut sync = self.sync.lock();

        sync.sync(
            &self.shared,
            value_tx,
            self.shared.pages.clone(),
            self.shared.values.clone(),
            self.shared.rollback.clone(),
            page_cache,
            page_diffs,
        )
        .unwrap();
        Ok(())
    }
}

/// An atomic transaction on raw key/value pairs to be applied against the store
/// with [`Store::commit`].
pub struct ValueTransaction {
    batch: Vec<(KeyPath, beatree::ValueChange)>,
}

impl ValueTransaction {
    /// Write a value to flat storage.
    pub fn write_value<T: ValueHasher>(&mut self, path: KeyPath, value: Option<Vec<u8>>) {
        self.batch
            .push((path, beatree::ValueChange::from_option::<T>(value)))
    }
}

/// An atomic transaction on merkle tree pages to be applied against the store
/// with [`Store::commit`].
pub struct MerkleTransaction {
    pub(crate) page_pool: PagePool,
    pub(crate) bucket_allocator: bitbox::BucketAllocator,
    pub(crate) new_pages: Vec<(PageId, BucketIndex, Option<(FatPage, PageDiff)>)>,
}

impl MerkleTransaction {
    /// Write a page to storage in its entirety.
    pub fn write_page(
        &mut self,
        page_id: PageId,
        bucket: Option<BucketIndex>,
        page: &FatPage,
        page_diff: PageDiff,
    ) -> BucketIndex {
        let bucket_index =
            bucket.unwrap_or_else(|| self.bucket_allocator.allocate(page_id.clone()));

        // Perform a deep clone of the page. For that allocate a new page and copy the data over.
        //
        // TODO: get rid of this copy.
        let mut new_page = self.page_pool.alloc_fat_page();
        new_page.copy_from_slice(page);

        self.new_pages
            .push((page_id, bucket_index, Some((new_page, page_diff))));
        bucket_index
    }

    /// Delete a page from storage.
    pub fn delete_page(&mut self, page_id: PageId, bucket: BucketIndex) {
        self.bucket_allocator.free(bucket);
        self.new_pages.push((page_id, bucket, None));
    }
}

/// Creates and initializes a new empty database at the specified path.
///
/// This function:
/// - Creates all necessary directories along the path
/// - Initializes required database files
/// - Returns a file descriptor for the database directory
///
/// The database directory must not exist when calling this function.
fn create(page_pool: &PagePool, o: &crate::Options) -> anyhow::Result<File> {
    // Create the directory and its parent directories.
    std::fs::create_dir_all(&o.path)?;
    let db_dir_fd = std::fs::File::open(&o.path)?;

    let meta_fd = std::fs::File::create(o.path.join("meta"))?;
    let meta = Meta::create_new(o.bitbox_seed, o.bitbox_num_pages);
    Meta::write(page_pool, &meta_fd, &meta)?;
    drop(meta_fd);

    bitbox::create(o.path.clone(), o.bitbox_num_pages, o.preallocate_ht)?;
    beatree::create(&o.path)?;

    // As the last step, sync the directory. This makes sure that the directory is properly
    // written to disk.
    db_dir_fd.sync_all()?;
    Ok(db_dir_fd)
}
