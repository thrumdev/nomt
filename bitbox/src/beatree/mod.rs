use std::{
    collections::BTreeMap,
    mem,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    beatree::leaf::store::LeafStoreCommitOutput,
    io::{CompleteIo, IoCommand},
};

use branch::BranchId;
use crossbeam_channel::{Receiver, Sender};
use leaf::store::{LeafStoreReader, LeafStoreWriter};
use meta::Meta;

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
    sync_seqn: u32,
    next_bbn_seqn: u32,
    sync_io_handle_index: usize,
    sync_io_sender: Sender<IoCommand>,
    sync_io_receiver: Receiver<CompleteIo>,
}

impl Shared {
    fn take_staged_changeset(&mut self) -> BTreeMap<Key, Option<Vec<u8>>> {
        mem::take(&mut self.staging)
    }
}

impl Tree {
    pub fn open(db_dir: impl AsRef<Path>) -> Tree {
        let _ = db_dir;
        todo!()
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

        let sync_seqn = sync.sync_seqn;
        let mut next_bbn_seqn = sync.next_bbn_seqn;

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

        let obsolete_branches = ops::update(
            sync_seqn,
            &mut next_bbn_seqn,
            staged_changeset,
            &mut bbn_index,
            &mut branch_node_pool,
            &mut sync.leaf_store_wr,
        )
        .unwrap();

        let (ln, ln_freelist_pn, ln_bump, ln_extend_file_sz) = {
            let LeafStoreCommitOutput {
                pages,
                extend_file_sz,
                freelist_head,
                bump,
            } = sync.leaf_store_wr.commit();
            (pages, freelist_head.0, bump.0, extend_file_sz)
        };

        // TODO: BBN dumping.

        let new_meta = Meta {
            sync_seqn,
            next_bbn_seqn,
            ln_freelist_pn,
            ln_bump,
            bbn_bump: 0,
        };

        // writeout::run(
        //     sync.sync_io_sender,
        //     sync.sync_io_handle_index,
        //     sync.sync_io_receiver,
        //     bnp,
        //     bbn_fd,
        //     ln_fd,
        //     meta_fd,
        //     bbn,
        //     bbn_extend_file_sz,
        //     ln,
        //     ln_extend_file_sz,
        //     new_meta,
        // );
        let _ = (
            new_meta,
            ln,
            ln_extend_file_sz,
            &sync.sync_io_sender,
            &sync.sync_io_handle_index,
            &sync.sync_io_receiver,
            &writeout::run,
        );

        sync.next_bbn_seqn = next_bbn_seqn;
        sync.sync_seqn = sync_seqn + 1;

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
    let tree = Tree::open("test.store");
    let mut changeset = Vec::new();
    let key = [1u8; 32];
    changeset.push((key.clone(), Some(b"value1".to_vec())));
    tree.commit(changeset);
    assert_eq!(tree.lookup(key), Some(b"value1".to_vec()));
    tree.sync();
}
