use anyhow::Result;
use bitvec::prelude::*;

use std::collections::{BTreeMap, HashMap};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{node::BranchNode, BranchNodePool, BRANCH_NODE_BODY_SIZE},
    index::Index,
    leaf::{
        node::{LeafNode, LEAF_NODE_BODY_SIZE, MAX_LEAF_VALUE_SIZE},
        overflow,
        store::{LeafStoreReader, LeafStoreWriter},
    },
    Key,
};
use crate::io::PagePool;

use super::BranchId;
use branch::{BaseBranch, BranchUpdater, DigestResult as BranchDigestResult};
use leaf::{BaseLeaf, DigestResult as LeafDigestResult, LeafUpdater};

pub(crate) mod branch;
pub(crate) mod leaf;

// All nodes less than this body size will be merged with a neighboring node.
const BRANCH_MERGE_THRESHOLD: usize = BRANCH_NODE_BODY_SIZE / 2;

// At 180% of the branch size, we perform a 'bulk split' which follows a different algorithm
// than a simple split. Bulk splits are encountered when there are a large number of insertions
// on a single node, typically when inserting into a fresh database.
const BRANCH_BULK_SPLIT_THRESHOLD: usize = (BRANCH_NODE_BODY_SIZE * 9) / 5;
// When performing a bulk split, we target 75% fullness for all of the nodes we create except the
// last.
const BRANCH_BULK_SPLIT_TARGET: usize = (BRANCH_NODE_BODY_SIZE * 3) / 4;

const LEAF_MERGE_THRESHOLD: usize = LEAF_NODE_BODY_SIZE / 2;
const LEAF_BULK_SPLIT_THRESHOLD: usize = (LEAF_NODE_BODY_SIZE * 9) / 5;
const LEAF_BULK_SPLIT_TARGET: usize = (LEAF_NODE_BODY_SIZE * 3) / 4;

/// Change the btree in the specified way. Updates the branch index in-place and returns
/// a list of branches which have become obsolete.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    changeset: &BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    bnp: &mut BranchNodePool,
    leaf_reader: &LeafStoreReader,
    leaf_writer: &mut LeafStoreWriter,
    bbn_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let leaf_cache = preload_leaves(leaf_reader, &bbn_index, &bnp, changeset.keys().cloned())?;

    let changeset = changeset
        .iter()
        .map(|(k, v)| match v {
            Some(v) if v.len() <= MAX_LEAF_VALUE_SIZE => (*k, Some((v.clone(), false))),
            Some(large_value) => {
                let pages = overflow::chunk(&large_value, leaf_writer);
                let cell = overflow::encode_cell(large_value.len(), &pages);
                (*k, Some((cell, true)))
            }
            None => (*k, None),
        })
        .collect::<_>();

    let leaf_changes = leaf_stage(
        &bbn_index,
        &bnp,
        leaf_cache,
        leaf_reader,
        leaf_writer.page_pool().clone(),
        changeset,
    );

    let branch_changeset = leaf_changes
        .inner
        .into_iter()
        .map(|(key, leaf_entry)| {
            let leaf_pn = leaf_entry.inserted.map(|leaf| leaf_writer.write(leaf));
            if let Some(prev_pn) = leaf_entry.deleted {
                leaf_writer.release(prev_pn);
            }

            (key, leaf_pn)
        })
        .collect::<Vec<_>>();

    for overflow_deleted in leaf_changes.overflow_deleted {
        overflow::delete(&overflow_deleted, leaf_reader, leaf_writer);
    }

    let branch_changes = branch_stage(&bbn_index, &bnp, branch_changeset);

    let mut removed_branches = Vec::new();
    for (key, changed_branch) in branch_changes.inner {
        match changed_branch.inserted {
            Some((branch_id, node)) => {
                bbn_index.insert(key, branch_id);
                bbn_writer.allocate(node);
            }
            None => {
                bbn_index.remove(&key);
            }
        }

        if let Some((deleted_branch_id, deleted_pn)) = changed_branch.deleted {
            removed_branches.push(deleted_branch_id);
            bbn_writer.release(deleted_pn);
        }
    }

    for branch_id in branch_changes.fresh_released {
        bnp.release(branch_id);
    }

    Ok(removed_branches)
}

struct ChangedLeafEntry {
    deleted: Option<PageNumber>,
    inserted: Option<LeafNode>,
}

#[derive(Default)]
struct LeafChanges {
    inner: BTreeMap<Key, ChangedLeafEntry>,
    overflow_deleted: Vec<Vec<u8>>,
}

impl LeafChanges {
    fn delete(&mut self, key: Key, pn: PageNumber) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedLeafEntry {
            deleted: None,
            inserted: None,
        });

        // we can only delete a leaf once.
        assert!(entry.deleted.is_none());

        entry.deleted = Some(pn);
    }

    fn insert(&mut self, key: Key, node: LeafNode) {
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
    let Some((i, leaf_pn)) = super::search_branch(&branch, key) else {
        return;
    };
    let separator = get_key(&branch, i);

    // we intend to work on this leaf, therefore, we delete it. any new leaves produced by the
    // updater will replace it.
    leaf_changes.delete(separator, leaf_pn);

    let cutoff = if i + 1 < branch.n() as usize {
        Some(get_key(&branch, i + 1))
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

fn leaf_stage(
    bbn_index: &Index,
    bnp: &BranchNodePool,
    mut leaf_cache: HashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    page_pool: PagePool,
    changeset: Vec<(Key, Option<(Vec<u8>, bool)>)>,
) -> LeafChanges {
    if changeset.is_empty() {
        return LeafChanges::default();
    }
    let mut leaf_changes = LeafChanges::default();

    let mut leaf_updater = LeafUpdater::new(page_pool, None, None);

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

    leaf_changes
}

struct ChangedBranchEntry {
    deleted: Option<(BranchId, PageNumber)>,
    inserted: Option<(BranchId, BranchNode)>,
}

#[derive(Default)]
struct BranchChanges {
    inner: BTreeMap<Key, ChangedBranchEntry>,
    fresh_released: Vec<BranchId>,
}

impl BranchChanges {
    fn delete(&mut self, key: Key, branch_id: BranchId, pn: PageNumber) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedBranchEntry {
            deleted: None,
            inserted: None,
        });

        // we can only delete a branch once.
        assert!(entry.deleted.is_none());

        entry.deleted = Some((branch_id, pn));
    }

    fn insert(&mut self, key: Key, branch_id: BranchId, node: BranchNode) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedBranchEntry {
            deleted: None,
            inserted: None,
        });

        if let Some((prev_id, _)) = entry.inserted.replace((branch_id, node)) {
            self.fresh_released.push(prev_id);
        }
    }
}

fn reset_branch_base(
    bbn_index: &Index,
    bnp: &BranchNodePool,
    branch_changes: &mut BranchChanges,
    branch_updater: &mut BranchUpdater,
    key: Key,
) {
    let Some((separator, branch_id)) = bbn_index.lookup(key) else {
        return;
    };

    // UNWRAP: all indexed branches exist.
    let branch = bnp.checkout(branch_id).unwrap();
    let cutoff = bbn_index.next_after(key).map(|(cutoff, _)| cutoff);

    branch_changes.delete(separator, branch_id, branch.bbn_pn().into());

    let base = BaseBranch {
        node: branch,
        iter_pos: 0,
    };
    branch_updater.reset_base(Some(base), cutoff);
}

fn branch_stage(
    bbn_index: &Index,
    bnp: &BranchNodePool,
    changeset: Vec<(Key, Option<PageNumber>)>,
) -> BranchChanges {
    if changeset.is_empty() {
        return BranchChanges::default();
    }
    let mut branch_changes = BranchChanges::default();

    let mut branch_updater = BranchUpdater::new(None, None);

    // point branch updater at first branch.
    reset_branch_base(
        bbn_index,
        &bnp,
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
                branch_updater.digest(&bnp, &mut branch_changes)
            {
                cutoff
            } else {
                key
            };

            reset_branch_base(bbn_index, &bnp, &mut branch_changes, &mut branch_updater, k);
        }

        branch_updater.ingest(key, op);
    }

    loop {
        if let BranchDigestResult::NeedsMerge(cutoff) =
            branch_updater.digest(&bnp, &mut branch_changes)
        {
            reset_branch_base(
                bbn_index,
                &bnp,
                &mut branch_changes,
                &mut branch_updater,
                cutoff,
            );
            continue;
        }
        break;
    }

    branch_changes
}

pub fn reconstruct_key(prefix: Option<&BitSlice<u8, Msb0>>, separator: &BitSlice<u8, Msb0>) -> Key {
    let mut key = [0u8; 32];
    match prefix {
        Some(prefix) => {
            key.view_bits_mut::<Msb0>()[..prefix.len()].copy_from_bitslice(prefix);
            key.view_bits_mut::<Msb0>()[prefix.len()..][..separator.len()]
                .copy_from_bitslice(separator);
        }
        None => {
            key.view_bits_mut::<Msb0>()[..separator.len()].copy_from_bitslice(separator);
        }
    }
    key
}

pub fn get_key(node: &BranchNode, index: usize) -> Key {
    let prefix = if index < node.prefix_compressed() as usize {
        Some(node.prefix())
    } else {
        None
    };
    reconstruct_key(prefix, node.separator(index))
}

fn preload_leaves(
    leaf_reader: &LeafStoreReader,
    bbn_index: &Index,
    bnp: &BranchNodePool,
    keys: impl IntoIterator<Item = Key>,
) -> Result<HashMap<PageNumber, LeafNode>> {
    let mut leaf_pages = HashMap::new();
    let mut last_pn = None;

    let mut submissions = 0;
    for key in keys {
        let Some((_, branch_id)) = bbn_index.lookup(key) else {
            continue;
        };
        // UNWRAP: all branches in index exist.
        let branch = bnp.checkout(branch_id).unwrap();
        let Some((_, leaf_pn)) = super::search_branch(&branch, key) else {
            continue;
        };
        if last_pn == Some(leaf_pn) {
            continue;
        }
        last_pn = Some(leaf_pn);
        leaf_reader
            .io_handle()
            .send(leaf_reader.io_command(leaf_pn, leaf_pn.0 as u64))
            .expect("I/O Pool Disconnected");

        submissions += 1;
    }

    for _ in 0..submissions {
        let completion = leaf_reader
            .io_handle()
            .recv()
            .expect("I/O Pool Disconnected");
        completion.result?;
        let pn = PageNumber(completion.command.user_data as u32);
        let page = completion.command.kind.unwrap_buf();
        leaf_pages.insert(pn, LeafNode { inner: page });
    }

    Ok(leaf_pages)
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use bitvec::{prelude::Msb0, view::BitView};
    use criterion::{BenchmarkId, Criterion};
    use rand::RngCore;

    pub fn reconstruct_key_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("reconstruct_key");

        for prefix_bytes in [0, 4, 8, 12, 16, 20] {
            let mut rand = rand::thread_rng();
            let mut key = [0; 32];
            rand.fill_bytes(&mut key);

            let prefix = &key.view_bits::<Msb0>()[0..prefix_bytes * 8];
            let separator = &key.view_bits::<Msb0>()[prefix_bytes * 8..];

            group.bench_function(BenchmarkId::new("prefix_len_bytes", prefix_bytes), |b| {
                b.iter(|| super::reconstruct_key(Some(prefix), separator))
            });
        }

        group.finish();
    }
}
