use std::collections::{BTreeMap, HashMap};

use crate::beatree::{
    allocator::PageNumber,
    branch::BranchNodePool,
    index::Index,
    leaf::{node::LeafNode, store::LeafStoreReader},
    ops::search_branch,
    Key,
};

use super::{
    leaf_updater::{BaseLeaf, DigestResult as LeafDigestResult, LeafUpdater},
    reconstruct_key,
};

pub struct ChangedLeafEntry {
    pub deleted: Option<PageNumber>,
    pub inserted: Option<LeafNode>,
}

#[derive(Default)]
pub struct LeafChanges {
    inner: BTreeMap<Key, ChangedLeafEntry>,
    overflow_deleted: Vec<Vec<u8>>,
}

impl LeafChanges {
    pub fn delete(&mut self, key: Key, pn: PageNumber) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedLeafEntry {
            deleted: None,
            inserted: None,
        });

        // we can only delete a leaf once.
        assert!(entry.deleted.is_none());

        entry.deleted = Some(pn);
    }

    pub fn insert(&mut self, key: Key, node: LeafNode) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedLeafEntry {
            deleted: None,
            inserted: None,
        });

        if let Some(_prev) = entry.inserted.replace(node) {
            // TODO: this is where we'd clean up.
        }
    }

    fn delete_overflow(&mut self, overflow_cell: &[u8]) {
        self.overflow_deleted.push(overflow_cell.to_vec());
    }
}

fn reset_leaf_base(
    bbn_index: &Index,
    bnp: &BranchNodePool,
    leaf_cache: &mut HashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    leaf_changes: &mut LeafChanges,
    leaf_updater: &mut LeafUpdater,
    key: Key,
) {
    let Some((_, branch_id)) = bbn_index.lookup(key) else {
        return;
    };

    // UNWRAP: branches in index always exist.
    let branch = bnp.checkout(branch_id).unwrap();
    let Some((i, leaf_pn)) = search_branch(&branch, key) else {
        return;
    };
    let separator = reconstruct_key(branch.prefix(), branch.separator(i));

    // we intend to work on this leaf, therefore, we delete it. any new leaves produced by the
    // updater will replace it.
    leaf_changes.delete(separator, leaf_pn);

    let cutoff = if i + 1 < branch.n() as usize {
        Some(reconstruct_key(branch.prefix(), branch.separator(i + 1)))
    } else {
        bbn_index.next_after(key).map(|(cutoff, _)| cutoff)
    };

    let base = BaseLeaf {
        node: leaf_cache.remove(&leaf_pn).unwrap_or_else(|| LeafNode {
            inner: leaf_reader.query(leaf_pn),
        }),
        iter_pos: 0,
        separator,
    };

    leaf_updater.reset_base(Some(base), cutoff);
}

pub fn run(
    bbn_index: &Index,
    bnp: &BranchNodePool,
    mut leaf_cache: HashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    changeset: Vec<(Key, Option<(Vec<u8>, bool)>)>,
) -> (BTreeMap<Key, ChangedLeafEntry>, Vec<Vec<u8>>) {
    if changeset.is_empty() {
        return (BTreeMap::new(), Vec::new());
    }
    let mut leaf_changes = LeafChanges::default();

    let mut leaf_updater = LeafUpdater::new(None, None);

    // point leaf updater at first leaf.
    reset_leaf_base(
        bbn_index,
        &bnp,
        &mut leaf_cache,
        &leaf_reader,
        &mut leaf_changes,
        &mut leaf_updater,
        // UNWRAP: size checked
        changeset.first().unwrap().0,
    );

    for (key, op) in changeset {
        // ensure key is in scope for leaf updater. if not, digest it. merge rightwards until
        //    done _or_ key is in scope.
        while !leaf_updater.is_in_scope(&key) {
            let k = if let LeafDigestResult::NeedsMerge(cutoff) =
                leaf_updater.digest(&mut leaf_changes)
            {
                cutoff
            } else {
                key
            };

            reset_leaf_base(
                bbn_index,
                &bnp,
                &mut leaf_cache,
                &leaf_reader,
                &mut leaf_changes,
                &mut leaf_updater,
                k,
            );
        }

        let (value_change, overflow) = match op {
            None => (None, false),
            Some((v, overflow)) => (Some(v), overflow),
        };

        let delete_overflow = |overflow_cell: &[u8]| leaf_changes.delete_overflow(overflow_cell);
        leaf_updater.ingest(key, value_change, overflow, delete_overflow);
    }

    loop {
        if let LeafDigestResult::NeedsMerge(cutoff) = leaf_updater.digest(&mut leaf_changes) {
            reset_leaf_base(
                bbn_index,
                &bnp,
                &mut leaf_cache,
                &leaf_reader,
                &mut leaf_changes,
                &mut leaf_updater,
                cutoff,
            );
            continue;
        }
        break;
    }

    (leaf_changes.inner, leaf_changes.overflow_deleted)
}
