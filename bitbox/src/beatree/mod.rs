use std::{
    collections::BTreeMap,
    mem,
    path::Path,
    sync::{Arc, Mutex},
};

use branch::BranchId;

mod branch;
mod btree;
mod leaf;
mod meta;
mod bbn;

pub struct Tree {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    root: Option<BranchId>,
    leaf_store: leaf::LeafStore,
    branch_node_pool: branch::BranchNodePool,
    primary_staging: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    secondary_staging: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
}

impl Inner {
    fn blit(&mut self) {
        assert!(self.secondary_staging.is_empty());
        mem::swap(&mut self.primary_staging, &mut self.secondary_staging);
    }
}

impl Tree {
    pub fn open(db_dir: impl AsRef<Path>) -> Tree {
        todo!()
    }

    /// Lookup a key in the btree.
    pub fn lookup(&self, key: Vec<u8>) -> Option<Vec<u8>> {
        let inner = self.inner.lock().unwrap();
        let Some(ref root) = inner.root else {
            return None;
        };
        btree::lookup(key, *root, &inner.branch_node_pool, &inner.leaf_store).unwrap()
    }

    /// Commit a set of changes to the btree.
    ///
    /// The changeset is a list of key value pairs to be added or removed from the btree.
    /// The changeset is applied atomically. If the changeset is empty, the btree is not modified.
    pub fn commit(&self, changeset: Vec<(Vec<u8>, Option<Vec<u8>>)>) {
        if changeset.is_empty() {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
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

        // Under a lock, swap the primary map with the secondary map, then drop the lock.
        {
            let mut inner = self.inner.lock().unwrap();
            inner.blit();
        }

        todo!()
    }
}

#[allow(unused)]
pub fn test_btree() {
    let tree = Tree::open("test.store");
    let mut changeset = Vec::new();
    changeset.push((b"key1".to_vec(), Some(b"value1".to_vec())));
    tree.commit(changeset);
    assert_eq!(tree.lookup(b"key1".to_vec()), Some(b"value1".to_vec()));
    tree.sync();
}
