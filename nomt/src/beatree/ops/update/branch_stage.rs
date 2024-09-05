use std::collections::BTreeMap;

use crate::beatree::{
    allocator::PageNumber,
    branch::node::BranchNode,
    index::Index,
    Key,
};

use crate::io::PagePool;

use super::{
    branch_updater::{BaseBranch, BranchUpdater, DigestResult as BranchDigestResult},
};

pub struct ChangedBranchEntry {
    pub deleted: Option<PageNumber>,
    pub inserted: Option<BranchNode>,
}

#[derive(Default)]
pub struct BranchChanges {
    inner: BTreeMap<Key, ChangedBranchEntry>,
}

impl BranchChanges {
    pub fn delete(&mut self, key: Key, pn: PageNumber) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedBranchEntry {
            deleted: None,
            inserted: None,
        });

        // we can only delete a branch once.
        assert!(entry.deleted.is_none());

        entry.deleted = Some(pn);
    }

    pub fn insert(&mut self, key: Key, node: BranchNode) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedBranchEntry {
            deleted: None,
            inserted: None,
        });

        entry.inserted = Some(node);
    }
}

fn reset_branch_base(
    bbn_index: &Index,
    branch_changes: &mut BranchChanges,
    branch_updater: &mut BranchUpdater,
    key: Key,
) {
    let Some((separator, branch)) = bbn_index.lookup(key) else {
        return;
    };

    let cutoff = bbn_index.next_after(key).map(|(cutoff, _)| cutoff);

    branch_changes.delete(separator, branch.bbn_pn().into());

    let base = BaseBranch {
        node: branch,
        iter_pos: 0,
    };
    branch_updater.reset_base(Some(base), cutoff);
}

pub fn run(
    bbn_index: &Index,
    page_pool: PagePool,
    changeset: Vec<(Key, Option<PageNumber>)>,
) -> BTreeMap<Key, ChangedBranchEntry> {
    if changeset.is_empty() {
        return BTreeMap::new();
    }
    let mut branch_changes = BranchChanges::default();

    let mut branch_updater = BranchUpdater::new(page_pool, None, None);

    // point branch updater at first branch.
    reset_branch_base(
        bbn_index,
        &mut branch_changes,
        &mut branch_updater,
        // UNWRAP: size checked
        changeset.first().unwrap().0,
    );

    for (key, op) in changeset {
        // ensure key is in scope for branch updater. if not, digest it. merge rightwards until
        //    done _or_ key is in scope.
        while !branch_updater.is_in_scope(&key) {
            let k = if let BranchDigestResult::NeedsMerge(cutoff) =
                branch_updater.digest(&mut branch_changes)
            {
                cutoff
            } else {
                key
            };

            reset_branch_base(bbn_index, &mut branch_changes, &mut branch_updater, k);
        }

        branch_updater.ingest(key, op);
    }

    loop {
        if let BranchDigestResult::NeedsMerge(cutoff) =
            branch_updater.digest(&mut branch_changes)
        {
            reset_branch_base(
                bbn_index,
                &mut branch_changes,
                &mut branch_updater,
                cutoff,
            );
            continue;
        }
        break;
    }

    branch_changes.inner
}
