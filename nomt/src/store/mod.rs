//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use crate::{beatree, bitbox, io, page_cache::PageDiff};
use meta::Meta;
use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use parking_lot::Mutex;
use std::{
    fs::{File, OpenOptions},
    os::{fd::AsRawFd as _, unix::fs::OpenOptionsExt as _},
    sync::Arc,
};

pub use bitbox::BucketIndex;

mod meta;
mod writeout;

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
    sync: Arc<Mutex<Sync>>,
}

struct Shared {
    root: Mutex<Node>,
    bitbox_num_pages: u32,
    values: beatree::Tree,
    pages: bitbox::DB,
    meta_fd: File,
    ln_fd: File,
    bbn_fd: File,
}

struct Sync {
    sync_seqn: u32,
    io_sender: crossbeam_channel::Sender<io::IoCommand>,
    io_handle_index: usize,
    io_receiver: crossbeam_channel::Receiver<io::CompleteIo>,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        if !o.path.exists() {
            create(o)?;
        }
        let meta_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&o.path.join("meta"))?;
        let ln_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(&o.path.join("ln"))?;
        let bbn_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(&o.path.join("bbn"))?;
        let meta = meta::Meta::read(&meta_fd)?;
        let values = beatree::Tree::open(
            meta.ln_freelist_pn,
            meta.bbn_freelist_pn,
            meta.ln_bump,
            meta.bbn_bump,
            &bbn_fd,
            &ln_fd,
        )?;
        let pages = bitbox::DB::open(
            meta.sync_seqn,
            meta.bitbox_num_pages,
            o.num_rings,
            o.path.clone(),
        )?;
        let (io_sender, mut receivers) = io::start_io_worker(1, 3);
        Ok(Self {
            shared: Arc::new(Shared {
                bitbox_num_pages: meta.bitbox_num_pages,
                values,
                pages,
                root: TERMINATOR.into(),
                meta_fd,
                ln_fd,
                bbn_fd,
            }),
            sync: Arc::new(Mutex::new(Sync {
                sync_seqn: meta.sync_seqn,
                io_sender,
                io_handle_index: 0,
                io_receiver: receivers.remove(0),
            })),
        })
    }

    /// Load the root node from the database. Fails only on I/O.
    /// Returns [`nomt_core::trie::TERMINATOR`] on an empty trie.
    pub fn load_root(&self) -> anyhow::Result<Node> {
        Ok(self.shared.root.lock().clone())
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.shared.values.lookup(key))
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<(Vec<u8>, BucketIndex)>> {
        self.shared.pages.get(&page_id)
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        Transaction {
            batch: Vec::new(),
            new_pages: vec![],
            bucket_allocator: self.shared.pages.bucket_allocator(),
            new_root: None,
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
        let prev_wal_size = self.shared.pages.sync_begin(tx.new_pages, sync_seqn)?;
        if let Some(new_root) = tx.new_root {
            *self.shared.root.lock() = new_root;
        }
        let data = self.shared.values.prepare_sync();

        let new_meta = Meta {
            ln_freelist_pn: data.ln_freelist_pn,
            ln_bump: data.ln_bump,
            bbn_freelist_pn: data.bbn_freelist_pn,
            bbn_bump: data.bbn_bump,
            sync_seqn,
            bitbox_num_pages: self.shared.bitbox_num_pages,
        };

        writeout::run(
            sync.io_sender.clone(),
            sync.io_handle_index,
            sync.io_receiver.clone(),
            self.shared.bbn_fd.as_raw_fd(),
            self.shared.ln_fd.as_raw_fd(),
            self.shared.meta_fd.as_raw_fd(),
            data.bbn,
            data.bbn_freelist_pages,
            data.bbn_extend_file_sz,
            data.ln,
            data.ln_extend_file_sz,
            new_meta,
        );

        self.shared.pages.sync_end(prev_wal_size)?;
        self.shared
            .values
            .finish_sync(data.bbn_index, data.obsolete_branches);

        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    batch: Vec<(KeyPath, Option<Vec<u8>>)>,
    bucket_allocator: bitbox::BucketAllocator,
    new_pages: Vec<(PageId, BucketIndex, Option<(Vec<u8>, PageDiff)>)>,
    new_root: Option<Node>,
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
        page: &[u8],
        page_diff: PageDiff,
    ) -> BucketIndex {
        let bucket_index =
            bucket.unwrap_or_else(|| self.bucket_allocator.allocate(page_id.clone()));
        self.new_pages
            .push((page_id, bucket_index, Some((page.to_vec(), page_diff))));
        bucket_index
    }
    /// Delete a page from storage.
    pub fn delete_page(&mut self, page_id: PageId, bucket: BucketIndex) {
        self.bucket_allocator.free(bucket);
        self.new_pages.push((page_id, bucket, None));
    }

    /// Write the root to metadata.
    pub fn write_root(&mut self, root: Node) {
        self.new_root = Some(root);
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
    }
    .encode_to(&mut buf[0..24]);
    meta_fd.write_all(&buf)?;
    meta_fd.sync_all()?;
    drop(meta_fd);

    bitbox::create(o.path.clone(), o.bitbox_num_pages)?;
    beatree::create(&o.path)?;

    // As the last step, sync the directory.
    std::fs::File::open(&o.path)?.sync_all()?;
    Ok(())
}
