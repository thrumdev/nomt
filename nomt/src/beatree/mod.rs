use allocator::{PageNumber, FREELIST_EMPTY};
use anyhow::{Context, Result};
use branch::{BranchId, BranchNode, BRANCH_NODE_SIZE};
use index::Index;
use std::{
    collections::BTreeMap,
    fs::File,
    mem,
    ops::DerefMut,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::io::{IoPool, Page};

use leaf::store::{LeafStoreReader, LeafStoreWriter};

use self::{
    bbn::{BbnStoreCommitOutput, BbnStoreWriter},
    leaf::store::LeafStoreCommitOutput,
};

pub(crate) mod allocator;
mod bbn;
pub(crate) mod branch;
mod index;
mod leaf;
pub(crate) mod ops;

#[cfg(feature = "benchmarks")]
pub mod benches;

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
    pub fn open(
        io_pool: &IoPool,
        ln_freelist_pn: u32,
        bbn_freelist_pn: u32,
        ln_bump: u32,
        bbn_bump: u32,
        bbn_file: &File,
        ln_file: &File,
    ) -> Result<Tree> {
        let ln_freelist_pn = Some(ln_freelist_pn)
            .map(PageNumber)
            .filter(|&x| x != FREELIST_EMPTY);
        let bbn_freelist_pn = Some(bbn_freelist_pn)
            .map(PageNumber)
            .filter(|&x| x != FREELIST_EMPTY);

        let ln_bump = PageNumber(ln_bump);
        let bbn_bump = PageNumber(bbn_bump);

        let (leaf_store_rd_shared, leaf_store_rd_sync, leaf_store_wr) = {
            let ln_file = ln_file.try_clone().unwrap();

            leaf::store::open(ln_file, ln_freelist_pn, ln_bump, &io_pool)
        };

        let (bbn_store_wr, bbn_freelist_tracked) = {
            let bbn_fd = bbn_file.try_clone().unwrap();
            bbn::open(bbn_fd, bbn_freelist_pn, bbn_bump)
        };
        let mut bnp = branch::BranchNodePool::new();
        let index = ops::reconstruct(
            bbn_file.try_clone().unwrap(),
            &mut bnp,
            &bbn_freelist_tracked,
            bbn_bump,
        )
        .with_context(|| format!("failed to reconstruct btree from bbn store file"))?;
        let shared = Shared {
            bbn_index: index,
            leaf_store_rd: leaf_store_rd_shared,
            branch_node_pool: bnp,
            primary_staging: BTreeMap::new(),
            secondary_staging: None,
        };

        let sync = Sync {
            leaf_store_wr,
            leaf_store_rd: leaf_store_rd_sync,
            bbn_store_wr,
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
    pub fn prepare_sync(&self) -> WriteoutData {
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

        WriteoutData {
            bbn,
            bbn_freelist_pages,
            bbn_extend_file_sz,
            ln,
            ln_extend_file_sz,
            ln_freelist_pn,
            ln_bump,
            bbn_freelist_pn,
            bbn_bump,
            bbn_index,
            obsolete_branches,
        }
    }

    pub fn finish_sync(&self, bbn_index: Index, obsolete_branches: Vec<BranchId>) {
        // Take the shared lock again to complete the update to the new shared state
        let mut inner = self.shared.lock().unwrap();
        inner.secondary_staging = None;
        inner.bbn_index = bbn_index;
        for id in obsolete_branches {
            inner.branch_node_pool.release(id);
        }
    }
}

pub struct WriteoutData {
    pub bbn: Vec<BranchNode>,
    pub bbn_freelist_pages: Vec<(PageNumber, Box<Page>)>,
    pub bbn_extend_file_sz: Option<u64>,
    pub ln: Vec<(PageNumber, Box<Page>)>,
    pub ln_extend_file_sz: Option<u64>,
    pub ln_freelist_pn: u32,
    pub ln_bump: u32,
    pub bbn_freelist_pn: u32,
    pub bbn_bump: u32,
    pub bbn_index: Index,
    pub obsolete_branches: Vec<BranchId>,
}

/// Creates the required files for the beatree.
pub fn create(db_dir: impl AsRef<Path>) -> anyhow::Result<()> {
    // Create the files.
    //
    // Size them to have an empty page at the beginning, this is reserved for the nil page.
    let ln_fd = File::create(db_dir.as_ref().join("ln"))?;
    let bbn_fd = File::create(db_dir.as_ref().join("bbn"))?;
    ln_fd.set_len(BRANCH_NODE_SIZE as u64)?;
    bbn_fd.set_len(BRANCH_NODE_SIZE as u64)?;

    // Sync files and the directory. I am not sure if syncing files is necessar, but it
    // is necessary to make sure that the directory is synced.
    ln_fd.sync_all()?;
    bbn_fd.sync_all()?;
    Ok(())
}
