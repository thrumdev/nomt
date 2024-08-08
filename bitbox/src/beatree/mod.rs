use allocator::PageNumber;
use anyhow::{Context, Result};
use branch::BRANCH_NODE_SIZE;
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    mem,
    ops::DerefMut,
    os::{
        fd::{AsRawFd as _, RawFd},
        unix::fs::OpenOptionsExt,
    },
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    beatree::{bbn::BbnStoreCommitOutput, leaf::store::LeafStoreCommitOutput},
    io::{self, CompleteIo, IoCommand, Mode},
};

use crossbeam_channel::{Receiver, Sender};
use leaf::store::{LeafStoreReader, LeafStoreWriter};
use meta::Meta;

use self::bbn::BbnStoreWriter;

mod allocator;
mod bbn;
mod branch;
mod index;
mod leaf;
mod meta;
mod ops;
mod writeout;

pub type Key = [u8; 32];

pub struct Tree {
    shared: Arc<Mutex<Shared>>,
    sync: Arc<Mutex<Sync>>,
}

struct Shared {
    bbn_index: index::Index,
    leaf_store_rd: LeafStoreReader,
    branch_node_pool: branch::BranchNodePool,
    /// Primary staging collects changes that are committed but not synced yet. Upon sync, changes
    /// from here are moved to secondary staging.
    primary_staging: BTreeMap<Key, Option<Vec<u8>>>,
    /// Secondary staging collects committed changes that are currently being synced. This is None
    /// if there is no sync in progress.
    secondary_staging: Option<Arc<BTreeMap<Key, Option<Vec<u8>>>>>,
}

struct Sync {
    leaf_store_wr: LeafStoreWriter,
    leaf_store_rd: LeafStoreReader,
    bbn_store_wr: BbnStoreWriter,
    sync_io_handle_index: usize,
    sync_io_sender: Sender<IoCommand>,
    sync_io_receiver: Receiver<CompleteIo>,
    meta_fd: RawFd,
    ln_fd: RawFd,
    bbn_fd: RawFd,
}

impl Shared {
    fn take_staged_changeset(&mut self) -> Arc<BTreeMap<Key, Option<Vec<u8>>>> {
        assert!(self.secondary_staging.is_none());
        let staged = Arc::new(mem::take(&mut self.primary_staging));
        self.secondary_staging = Some(staged.clone());
        staged
    }
}

impl Tree {
    pub fn open(db_dir: impl AsRef<Path>) -> Result<Tree> {
        const IO_IX_RD_LN_SHARED: usize = 0;
        const IO_IX_RD_LN_SYNC: usize = 1;
        const IO_IX_WR_LN: usize = 2;
        const IO_IX_BBN: usize = 3;
        const IO_IX_SYNC: usize = 4;
        const NUM_IO_HANDLES: usize = 5;

        let (io_sender, io_recv) = io::start_io_worker(NUM_IO_HANDLES, Mode::Real { num_rings: 3 });

        if !db_dir.as_ref().exists() {
            use std::io::Write as _;

            // Create the directory
            std::fs::create_dir_all(db_dir.as_ref())?;
            // Create the files
            let ln_fd = File::create(db_dir.as_ref().join("ln"))?;
            let bbn_fd = File::create(db_dir.as_ref().join("bbn"))?;
            ln_fd.set_len(BRANCH_NODE_SIZE as u64)?;
            bbn_fd.set_len(BRANCH_NODE_SIZE as u64)?;

            let mut meta_fd = File::create(db_dir.as_ref().join("meta"))?;
            let buf = [0u8; 4096];
            Meta {
                ln_freelist_pn: 0,
                ln_bump: 1,
                bbn_freelist_pn: 0,
                bbn_bump: 1,
            }
            .encode_to(&mut buf[0..20].try_into().unwrap());
            meta_fd.write_all(&buf)?;

            // Sync files and the directory. I am not sure if syncing files is necessar, but it
            // is necessary to make sure that the directory is synced.
            meta_fd.sync_all()?;
            ln_fd.sync_all()?;
            bbn_fd.sync_all()?;
            File::open(db_dir.as_ref())?.sync_all()?;
        }

        let bbn_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(db_dir.as_ref().join("bbn"))?;
        let ln_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(db_dir.as_ref().join("ln"))?;
        let meta_fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(db_dir.as_ref().join("meta"))?;

        let bbn_raw_fd = bbn_fd.as_raw_fd();
        let ln_raw_fd = ln_fd.as_raw_fd();
        let meta_raw_fd = meta_fd.as_raw_fd();

        let meta = meta::Meta::read(&meta_fd)?;
        let ln_freelist_pn = Some(meta.ln_freelist_pn)
            .filter(|&x| x != 0)
            .map(PageNumber);
        let bbn_freelist_pn = Some(meta.bbn_freelist_pn)
            .filter(|&x| x != 0)
            .map(PageNumber);
        let ln_bump = PageNumber(meta.ln_bump);
        let bbn_bump = PageNumber(meta.bbn_bump);

        let (leaf_store_rd_shared, leaf_store_rd_sync, leaf_store_wr) = {
            let rd_io_handle_index_shared = IO_IX_RD_LN_SHARED;
            let rd_io_sender_shared = io_sender.clone();
            let rd_io_receiver_shared = io_recv[rd_io_handle_index_shared].clone();
            let rd_io_handle_index_sync = IO_IX_RD_LN_SYNC;
            let rd_io_sender_sync = io_sender.clone();
            let rd_io_receiver_sync = io_recv[rd_io_handle_index_sync].clone();
            let wr_io_handle_index = IO_IX_WR_LN;
            let wr_io_sender = io_sender.clone();
            let wr_io_receiver = io_recv[wr_io_handle_index].clone();

            leaf::store::create(
                ln_fd,
                ln_freelist_pn,
                ln_bump,
                rd_io_handle_index_shared,
                rd_io_sender_shared,
                rd_io_receiver_shared,
                rd_io_handle_index_sync,
                rd_io_sender_sync,
                rd_io_receiver_sync,
                wr_io_handle_index,
                wr_io_sender,
                wr_io_receiver,
            )
        };

        let (bbn_store_wr, bbn_freelist) = {
            let bbn_fd = bbn_fd.try_clone().unwrap();
            let io_handle_index = IO_IX_BBN;
            let io_sender = io_sender.clone();
            let io_receiver = io_recv[io_handle_index].clone();
            bbn::create(
                bbn_fd,
                bbn_freelist_pn,
                bbn_bump,
                io_handle_index,
                io_sender,
                io_receiver,
            )
        };
        let mut bnp = branch::BranchNodePool::new();
        let index = ops::reconstruct(bbn_fd, &mut bnp, &bbn_freelist, bbn_bump)
            .with_context(|| format!("failed to reconstruct btree from bbn store file"))?;
        let shared = Shared {
            bbn_index: index,
            leaf_store_rd: leaf_store_rd_shared,
            branch_node_pool: bnp,
            primary_staging: BTreeMap::new(),
            secondary_staging: None,
        };

        let sync_io_handle_index = IO_IX_SYNC;
        let sync_io_sender = io_sender.clone();
        let sync_io_receiver = io_recv[sync_io_handle_index].clone();
        let sync = Sync {
            leaf_store_wr,
            leaf_store_rd: leaf_store_rd_sync,
            bbn_store_wr,
            sync_io_handle_index,
            sync_io_sender,
            sync_io_receiver,
            meta_fd: meta_raw_fd,
            ln_fd: ln_raw_fd,
            bbn_fd: bbn_raw_fd,
        };

        Ok(Tree {
            shared: Arc::new(Mutex::new(shared)),
            sync: Arc::new(Mutex::new(sync)),
        })
    }

    /// Lookup a key in the btree.
    pub fn lookup(&self, key: Key) -> Option<Vec<u8>> {
        let shared = self.shared.lock().unwrap();

        // First look up in the primary staging which contains the most recent changes.
        if let Some(val) = shared.primary_staging.get(&key) {
            return val.clone();
        }

        // Then check the secondary staging which is a bit older, but fresher still than the btree.
        if let Some(val) = shared.secondary_staging.as_ref().and_then(|x| x.get(&key)) {
            return val.clone();
        }

        // Finally, look up in the btree.
        ops::lookup(
            key,
            &shared.bbn_index,
            &shared.branch_node_pool,
            &shared.leaf_store_rd,
        )
        .unwrap()
    }

    /// Commit a set of changes to the btree.
    ///
    /// The changeset is a list of key value pairs to be added or removed from the btree.
    /// The changeset is applied atomically. If the changeset is empty, the btree is not modified.
    pub fn commit(&self, changeset: Vec<(Key, Option<Vec<u8>>)>) {
        if changeset.is_empty() {
            return;
        }
        let mut inner = self.shared.lock().unwrap();
        let staging = &mut inner.primary_staging;
        for (key, value) in changeset {
            staging.insert(key, value);
        }
    }

    /// Asynchronously dump all changes performed by commits to the underlying storage medium.
    ///
    /// Either blocks or panics if another sync is inflight.
    pub fn sync(&self) {
        // Take the sync lock.
        //
        // That will exclude any other syncs from happening. This is a long running operation.
        //
        // Note the ordering of taking locks is important.
        let mut sync = self.sync.lock().unwrap();

        // Take the shared lock. Briefly.
        let staged_changeset;
        let mut bbn_index;
        let mut branch_node_pool;
        {
            let mut shared = self.shared.lock().unwrap();
            staged_changeset = shared.take_staged_changeset();
            bbn_index = shared.bbn_index.clone();
            branch_node_pool = shared.branch_node_pool.clone();
        }

        let obsolete_branches = {
            let sync = sync.deref_mut();

            // Update will use the branch_node_pool in a cow manner to make lookups
            // possible during sync.
            // Nodes that need to be modified are not modified in place but they are
            // returned as obsolete, and new ones (implying the one created as modification
            // of existing nodes) are allocated using the previous state of the free list.
            //
            // Thus during the update:
            // + The index will just be modified, being a copy of the one used in parallel during lookups
            // + Allocation and releases of the leaf_store_wr will be executed normally,
            //   as everything will be in a pending state until commit
            // + The branch_node_pool will only allocate new BranchIds.
            //   All releases will be cached to be performed at the end of the sync.
            //   This makes it possible to keep the previous state of the tree (before this sync)
            //   available and reachable from the old index
            // + The bbn_store_wr follows the same reasoning as leaf_store_wr,
            //   so things will be allocated and released following what is being performed
            //   on the branch_node_pool and commited later on onto disk
            ops::update(
                &staged_changeset,
                &mut bbn_index,
                &mut branch_node_pool,
                &sync.leaf_store_rd,
                &mut sync.leaf_store_wr,
                &mut sync.bbn_store_wr,
            )
            .unwrap()
        };

        let (ln, ln_freelist_pn, ln_bump, ln_extend_file_sz) = {
            let LeafStoreCommitOutput {
                pages,
                extend_file_sz,
                freelist_head,
                bump,
            } = sync.leaf_store_wr.commit();
            (pages, freelist_head.0, bump.0, extend_file_sz)
        };

        let (bbn, bbn_freelist_pages, bbn_freelist_pn, bbn_bump, bbn_extend_file_sz) = {
            let BbnStoreCommitOutput {
                bbn,
                free_list_pages,
                extend_file_sz,
                freelist_head,
                bump,
            } = sync.bbn_store_wr.commit();
            (
                bbn,
                free_list_pages,
                freelist_head.0,
                bump.0,
                extend_file_sz,
            )
        };

        let new_meta = Meta {
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
        };

        writeout::run(
            sync.sync_io_sender.clone(),
            sync.sync_io_handle_index,
            sync.sync_io_receiver.clone(),
            sync.bbn_fd,
            sync.ln_fd,
            sync.meta_fd,
            bbn,
            bbn_freelist_pages,
            bbn_extend_file_sz,
            ln,
            ln_extend_file_sz,
            new_meta,
        );

        // Take the shared lock again to complete the update to the new shared state
        let mut inner = self.shared.lock().unwrap();
        inner.secondary_staging = None;
        inner.bbn_index = bbn_index;
        for id in obsolete_branches {
            inner.branch_node_pool.release(id);
        }
    }
}

#[allow(unused)]
pub fn test_btree() {
    let tree = Tree::open("test.store").unwrap();
    let mut changeset = Vec::new();
    let key = [1u8; 32];
    changeset.push((key.clone(), Some(b"value1".to_vec())));
    tree.commit(changeset);
    assert_eq!(tree.lookup(key), Some(b"value1".to_vec()));
    tree.sync();
}
