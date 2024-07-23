use allocator::PageNumber;
use anyhow::Result;
use branch::BRANCH_NODE_SIZE;
use std::{
    collections::BTreeMap,
    fs::File,
    mem,
    ops::DerefMut,
    os::fd::{AsRawFd as _, RawFd},
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
    staging: BTreeMap<Key, Option<Vec<u8>>>,
}

struct Sync {
    leaf_store_wr: LeafStoreWriter,
    bbn_store_wr: BbnStoreWriter,
    sync_io_handle_index: usize,
    sync_io_sender: Sender<IoCommand>,
    sync_io_receiver: Receiver<CompleteIo>,
    meta_fd: RawFd,
    ln_fd: RawFd,
    bbn_fd: RawFd,
}

impl Shared {
    fn take_staged_changeset(&mut self) -> BTreeMap<Key, Option<Vec<u8>>> {
        mem::take(&mut self.staging)
    }
}

impl Tree {
    pub fn open(db_dir: impl AsRef<Path>) -> Result<Tree> {
        const IO_IX_RD_LN: usize = 0;
        const IO_IX_WR_LN: usize = 1;
        const IO_IX_RD_BBN: usize = 2;
        const IO_IX_WR_BBN: usize = 3;
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
            let mut buf = [0u8; 20];
            Meta {
                ln_freelist_pn: 0,
                ln_bump: 1,
                bbn_freelist_pn: 0,
                bbn_bump: 1,
            }
            .encode_to(&mut buf);
            meta_fd.write_all(&buf)?;

            // Sync files and the directory. I am not sure if syncing files is necessar, but it
            // is necessary to make sure that the directory is synced.
            meta_fd.sync_all()?;
            ln_fd.sync_all()?;
            bbn_fd.sync_all()?;
            File::open(db_dir.as_ref())?.sync_all()?;
        }

        let bbn_fd = File::open(db_dir.as_ref().join("bbn"))?;
        let ln_fd = File::open(db_dir.as_ref().join("ln"))?;
        let meta_fd = File::open(db_dir.as_ref().join("meta"))?;

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

        let (leaf_store_rd, leaf_store_wr) = {
            let wr_io_handle_index = IO_IX_WR_LN;
            let wr_io_sender = io_sender.clone();
            let wr_io_receiver = io_recv[wr_io_handle_index].clone();
            let rd_io_handle_index = IO_IX_RD_LN;
            let rd_io_sender = io_sender.clone();
            let rd_io_receiver = io_recv[rd_io_handle_index].clone();
            leaf::store::create(
                ln_fd,
                ln_freelist_pn,
                ln_bump,
                wr_io_handle_index,
                wr_io_sender,
                wr_io_receiver,
                rd_io_handle_index,
                rd_io_sender,
                rd_io_receiver,
            )
        };

        let (bbn_store_wr, bbn_freelist) = {
            let bbn_fd = bbn_fd.try_clone().unwrap();
            let wr_io_handle_index = IO_IX_WR_BBN;
            let wr_io_sender = io_sender.clone();
            let wr_io_receiver = io_recv[wr_io_handle_index].clone();
            let rd_io_handle_index = IO_IX_RD_BBN;
            let rd_io_sender = io_sender.clone();
            let rd_io_receiver = io_recv[rd_io_handle_index].clone();
            bbn::create(
                bbn_fd,
                bbn_freelist_pn,
                bbn_bump,
                wr_io_handle_index,
                wr_io_sender,
                wr_io_receiver,
                rd_io_handle_index,
                rd_io_sender,
                rd_io_receiver,
            )
        };
        let mut bnp = branch::BranchNodePool::new();
        let index = ops::reconstruct(bbn_fd, &mut bnp, &bbn_freelist)?;
        let shared = Shared {
            bbn_index: index,
            leaf_store_rd,
            branch_node_pool: bnp,
            staging: BTreeMap::new(),
        };

        let sync_io_handle_index = IO_IX_SYNC;
        let sync_io_sender = io_sender.clone();
        let sync_io_receiver = io_recv[sync_io_handle_index].clone();
        let sync = Sync {
            leaf_store_wr,
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
        let staging = &mut inner.staging;
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
            let mut inner = self.shared.lock().unwrap();
            staged_changeset = inner.take_staged_changeset();
            bbn_index = inner.bbn_index.clone();
            branch_node_pool = inner.branch_node_pool.clone();
        }

        let obsolete_branches = {
            // TODO: I don't know if this is the best way to handle this,
            // but doing directly `&mut sync.leaf_store_wr` and `&mut sync.bbn_store_wr`
            // doesn't work because the borrow checker doesn't allow two
            // mutable borrow of `MutexGuard<_, Sync>`
            let sync = sync.deref_mut();

            ops::update(
                staged_changeset,
                &mut bbn_index,
                &mut branch_node_pool,
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

        // TODO: workaround to make it compile. Obviously broken. see the comment below.
        let bnp = branch::BranchNodePool::new();
        writeout::run(
            sync.sync_io_sender.clone(),
            sync.sync_io_handle_index,
            sync.sync_io_receiver.clone(),
            bnp,
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

        {
            let mut inner = self.shared.lock().unwrap();
            inner.bbn_index = bbn_index;
            // TODO: watch out for the lock contention here. If we change 50k values, that might
            // result in 50k bottom-level branch nodes being released. On top of that, there are
            // also upper-level branch nodes. We are also adding a lot of nodes one by one.
            //
            // If you think about it, it's all not necessary, because we only need to access to
            // the BNP freelist, which should be used exclusively by the sync thread. But note
            // once crucial thing: the obsolete branches are not released until the BBN index is
            // overwritten in the line above.
            //
            // Also, the sync needs to query the branch pages in the course of the sync, so that's
            // another indication for splitting the read/write parts.
            for id in obsolete_branches {
                inner.branch_node_pool.release(id);
            }
        }

        todo!()
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
