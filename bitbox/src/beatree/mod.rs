use std::{
    collections::BTreeMap,
    mem,
    path::Path,
    sync::{Arc, Mutex},
};

use branch::BranchId;
use crossbeam_channel::{Receiver, Sender};
use leaf::store::{LeafStoreReader, LeafStoreWriter};
use meta::Meta;

use crate::io::{CompleteIo, IoCommand};

mod bbn;
mod branch;
mod btree;
mod leaf;
mod meta;
mod writeout;

pub type Key = [u8; 32];

pub struct Tree {
    shared: Arc<Mutex<Shared>>,
    sync: Arc<Mutex<Sync>>,
}

struct Shared {
    root: Option<BranchId>,
    leaf_store_rd: LeafStoreReader,
    branch_node_pool: branch::BranchNodePool,
    primary_staging: BTreeMap<Key, Option<Vec<u8>>>,
    secondary_staging: BTreeMap<Key, Option<Vec<u8>>>,
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
        assert!(self.secondary_staging.is_empty());
        mem::take(&mut self.primary_staging)
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
        let Some(ref root) = shared.root else {
            return None;
        };
        btree::lookup(key, *root, &shared.branch_node_pool, &shared.leaf_store_rd).unwrap()
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
        let primary_staging = &mut inner.primary_staging;
        for (key, value) in changeset {
            primary_staging.insert(key, value);
        }
    }

    /// Asynchronously dump all changes performed by commits to the underlying storage medium.
    ///
    /// Either blocks or panics if another sync is inflight.
    pub fn sync(&self) {
        // - assert that the secondary staging is empty.
        // - move the primary staging to secondary staging.
        //     (from this point on, the commits will be editing the primary staging.)
        // - a new version of the index is built from the secondary staging.
        //     - the untouched nodes from the previous index are reused as is via references.
        // - then atomically
        //     - the secondary staging is discarded.
        //     - the new index replaces the old one.
        //     - the nodes of the old index are freed up.
        //     - the new BBNs and LNs are dumped into io engine and other sync-stuff is performed like metadata fsync.

        // Take the sync lock.
        //
        // That will exclude any other syncs from happening. This is a long running operation.
        //
        // Note the ordering of taking locks is important.
        let mut sync = self.sync.lock().unwrap();

        let sync_seqn = sync.sync_seqn;
        let mut next_bbn_seqn = sync.next_bbn_seqn;
        let mut leaf_store_tx = sync.leaf_store_wr.start_tx();

        // Take the shared lock. Briefly.
        let staged_changeset;
        let root;
        let mut branch_node_pool;
        {
            let mut inner = self.shared.lock().unwrap();
            staged_changeset = inner.take_staged_changeset();
            root = inner.root.unwrap();
            branch_node_pool = inner.branch_node_pool.clone();
        }

        let (new_root, obsolete_branches) = btree::update(
            sync_seqn,
            &mut next_bbn_seqn,
            staged_changeset,
            root,
            &mut branch_node_pool,
            &mut leaf_store_tx,
        )
        .unwrap();

        let (ln, ln_freelist_pn, ln_bump, ln_extend_file_sz) = {
            let o = leaf_store_tx.commit();
            let ln: Vec<()> = o
                .to_allocate
                .into_iter()
                .chain(o.exceeded)
                .map(|(_pn, _page)| ())
                .collect();
            let ln_freelist_pn = o.new_free_list_head.0;
            let ln_bump = 0; // TODO: commit should return the bump;
            let ln_extend_file_sz: Option<u64> = None;
            (ln, ln_freelist_pn, ln_bump, ln_extend_file_sz)
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
            inner.root = Some(new_root);
            // TODO: watch out for the lock contention here. If we change 50k values, that might
            // result in 50k bottom-level branch nodes being released. On top of that, there are
            // also upper-level branch nodes. We are also adding a lot of nodes one by one.
            //
            // If you think about it, it's all not necessary, because we only need to access to
            // the BNP freelist, which should be used exclusively by the sync thread.
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
