//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use crate::{
    beatree, bitbox,
    io::{self, page_pool::FatPage, IoPool, PagePool},
    page_diff::PageDiff,
};
use meta::Meta;
use nomt_core::{page_id::PageId, trie::KeyPath};
use parking_lot::Mutex;
use std::{
    fs::{File, OpenOptions},
    mem,
    os::fd::AsRawFd as _,
    sync::Arc,
};
use threadpool::ThreadPool;

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
    commit_concurrency: usize,
    values: beatree::Tree,
    pages: bitbox::DB,
    page_pool: PagePool,
    io_pool: IoPool,
    sync_tp: ThreadPool,
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
    io_handle: io::IoHandle,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options, page_pool: PagePool) -> anyhow::Result<Self> {
        if !o.path.exists() {
            create(o)?;
        }

        let io_pool = io::start_io_pool(o.io_workers, page_pool.clone());

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

        let meta = meta::Meta::read(&page_pool, &meta_fd)?;
        let values = beatree::Tree::open(
            page_pool.clone(),
            &io_pool,
            meta.ln_freelist_pn,
            meta.bbn_freelist_pn,
            meta.ln_bump,
            meta.bbn_bump,
            &bbn_fd,
            &ln_fd,
        )?;
        let pages = bitbox::DB::open(
            meta.bitbox_num_pages,
            meta.bitbox_seed,
            &page_pool,
            &ht_fd,
            &wal_fd,
        )?;
        let io_handle = io_pool.make_handle();
        Ok(Self {
            shared: Arc::new(Shared {
                page_pool,
                bitbox_num_pages: meta.bitbox_num_pages,
                bitbox_seed: meta.bitbox_seed,
                panic_on_sync: o.panic_on_sync,
                commit_concurrency: o.commit_concurrency,
                values,
                pages,
                sync_tp: ThreadPool::with_name(
                    "nomt-sync".into(),
                    o.commit_concurrency + 2, /* + sync_beatree + sync_bitbox thread */
                ),
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
            })),
        })
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.shared.values.lookup(key))
    }

    /// Loads the given page, blocking the current thread.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<(FatPage, BucketIndex)>> {
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
            page_pool: self.shared.page_pool.clone(),
            batch: Vec::new(),
            new_pages: vec![],
            bucket_allocator: self.shared.pages.bucket_allocator(),
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, mut tx: Transaction) -> anyhow::Result<()> {
        let mut sync = self.sync.lock();
        sync.sync_seqn += 1;

        // store the sequence number of this sync.
        let sync_seqn = sync.sync_seqn;

        let bitbox_wd = self.sync_bitbox(&mut tx)?;
        let beatree_wd = self.sync_beatree(&mut tx)?;

        let new_meta = Meta {
            ln_freelist_pn: beatree_wd.ln_freelist_pn,
            ln_bump: beatree_wd.ln_bump,
            bbn_freelist_pn: beatree_wd.bbn_freelist_pn,
            bbn_bump: beatree_wd.bbn_bump,
            sync_seqn,
            bitbox_num_pages: self.shared.bitbox_num_pages,
            bitbox_seed: self.shared.bitbox_seed,
        };

        writeout::run(
            sync.io_handle.clone(),
            self.shared.wal_fd.as_raw_fd(),
            self.shared.bbn_fd.as_raw_fd(),
            self.shared.ln_fd.as_raw_fd(),
            self.shared.ht_fd.as_raw_fd(),
            self.shared.meta_fd.as_raw_fd(),
            bitbox_wd.wal_blob,
            beatree_wd.bbn,
            beatree_wd.bbn_freelist_pages,
            beatree_wd.bbn_extend_file_sz,
            beatree_wd.ln,
            beatree_wd.ln_free_list_pages,
            beatree_wd.ln_extend_file_sz,
            bitbox_wd.ht_pages,
            new_meta,
            self.shared.panic_on_sync,
        );

        self.shared.values.finish_sync(beatree_wd.bbn_index);

        Ok(())
    }

    fn sync_bitbox(&self, tx: &mut Transaction) -> anyhow::Result<bitbox::WriteoutData> {
        let shared = self.shared.clone();
        // oneshot for getting the result.
        let (result_tx, result_rx) = crossbeam::channel::bounded(1);
        let new_pages = mem::take(&mut tx.new_pages);
        self.shared.sync_tp.execute(move || {
            let writeout_data = shared
                .pages
                .prepare_sync(&shared.page_pool, new_pages)
                .unwrap(); // TODO: handle error.
                           // UNWRAP: this should never fail because the channel cannot be disconnected since
                           //         we are waiting on the result just below.
            result_tx.send(writeout_data).unwrap();
        });

        // This unwrap is fine, because it can only happen if the channel is disconnected, which
        // indicates a panic happened in the thread above, and we don't want to be silent about
        // that. That said,
        // TODO: we should actually catch_unwind followed by resume_unwind to improve the panic
        // message.
        Ok(result_rx.recv().unwrap())
    }

    fn sync_beatree(&self, tx: &mut Transaction) -> anyhow::Result<beatree::WriteoutData> {
        let shared = self.shared.clone();
        let batch = mem::take(&mut tx.batch);
        let (result_tx, result_rx) = crossbeam::channel::bounded(1);
        self.shared.sync_tp.execute({
            let tp = self.shared.sync_tp.clone();
            let commit_concurrency = self.shared.commit_concurrency;
            move || {
                shared.values.commit(batch);
                let data = shared.values.prepare_sync(tp, commit_concurrency);
                // UNWRAP: this should never fail because the channel cannot be disconnected since
                //         we are waiting on the result just below.
                result_tx.send(data).unwrap();
            }
        });

        // This unwrap is fine, because it can only happen if the channel is disconnected, which
        // indicates a panic happened in the thread above, and we don't want to be silent about
        // that. That said,
        // TODO: we should actually catch_unwind followed by resume_unwind to improve the panic
        // message.
        Ok(result_rx.recv().unwrap())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    page_pool: PagePool,
    batch: Vec<(KeyPath, Option<Vec<u8>>)>,
    bucket_allocator: bitbox::BucketAllocator,
    new_pages: Vec<(PageId, BucketIndex, Option<(FatPage, PageDiff)>)>,
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
