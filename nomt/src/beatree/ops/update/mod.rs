use anyhow::Result;
use bitvec::prelude::*;

use std::collections::{BTreeMap, HashMap};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{BranchNodePool, BRANCH_NODE_BODY_SIZE},
    index::Index,
    leaf::{
        node::{LeafNode, LEAF_NODE_BODY_SIZE, MAX_LEAF_VALUE_SIZE},
        overflow,
        store::{LeafStoreReader, LeafStoreWriter},
    },
    Key,
};

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
    let leaf_pages = preload_leaves(leaf_reader, &bbn_index, &bnp, changeset.keys().cloned())?;

    let mut ctx = Ctx {
        bbn_index,
        bbn_writer,
        bnp,
        leaf_reader,
        leaf_writer,
    };

    let mut updater = Updater::new(&ctx, leaf_pages);
    for (key, value_change) in changeset {
        updater.ingest(*key, value_change.clone(), &mut ctx);
    }

    updater.complete(&mut ctx);

    Ok(updater.obsolete_branches)
}

struct Ctx<'a> {
    bbn_index: &'a mut Index,
    bbn_writer: &'a mut bbn::BbnStoreWriter,
    bnp: &'a mut BranchNodePool,
    leaf_reader: &'a LeafStoreReader,
    leaf_writer: &'a mut LeafStoreWriter,
}

struct Updater {
    branch_updater: BranchUpdater,
    leaf_updater: LeafUpdater,
    obsolete_branches: Vec<BranchId>,
    leaf_pages: HashMap<PageNumber, LeafNode>,
}

impl Updater {
    fn new(ctx: &Ctx, mut leaf_pages: HashMap<PageNumber, LeafNode>) -> Self {
        let first = ctx.bbn_index.first();

        // UNWRAP: all nodes in index must exist.
        let first_branch = first.as_ref().map(|(_, id)| ctx.bnp.checkout(*id).unwrap());
        let first_branch_cutoff = first
            .as_ref()
            .and_then(|(k, _)| ctx.bbn_index.next_after(*k))
            .map(|(k, _)| k);

        // first leaf cutoff is the separator of the second leaf _or_ the separator of the next
        // branch if there is only 1 leaf, or nothing.
        let first_leaf_cutoff = first_branch
            .as_ref()
            .and_then(|node| {
                if node.n() > 1 {
                    Some(reconstruct_key(node.prefix(), node.separator(1)))
                } else {
                    None
                }
            })
            .or(first_branch_cutoff);

        let first_leaf = first_branch
            .as_ref()
            .map(|node| {
                (
                    PageNumber::from(node.node_pointer(0)),
                    reconstruct_key(node.prefix(), node.separator(0)),
                )
            })
            .map(|(id, separator)| BaseLeaf {
                id,
                node: leaf_pages.remove(&id).unwrap_or_else(|| LeafNode {
                    inner: ctx.leaf_reader.query(id),
                }),
                iter_pos: 0,
                separator,
            });

        // start with the first branch, cut-off second branch key.
        let branch_updater = BranchUpdater::new(
            first_branch.map(|node| BaseBranch {
                // UNWRAP: node can only exist if ID does.
                id: first.as_ref().unwrap().1,
                node,
                iter_pos: 0,
            }),
            first_branch_cutoff,
        );

        // start with first leaf in first branch, cut-off second leaf key.
        let leaf_updater = LeafUpdater::new(first_leaf, first_leaf_cutoff);

        Updater {
            branch_updater,
            leaf_updater,
            obsolete_branches: Vec::new(),
            leaf_pages,
        }
    }

    fn ingest(&mut self, key: Key, value_change: Option<Vec<u8>>, ctx: &mut Ctx) {
        self.digest_until(Some(key), ctx);

        let (value_change, overflow) = if let Some(ref large_value) = value_change
            .as_ref()
            .filter(|v| v.len() > MAX_LEAF_VALUE_SIZE)
        {
            let pages = overflow::chunk(large_value, &mut ctx.leaf_writer);
            (Some(overflow::encode_cell(large_value.len(), &pages)), true)
        } else {
            (value_change, false)
        };

        let delete_overflow = |overflow_cell: &[u8]| {
            overflow::delete(overflow_cell, &ctx.leaf_reader, &mut ctx.leaf_writer)
        };
        self.leaf_updater
            .ingest(key, value_change, overflow, delete_overflow);
    }

    fn complete(&mut self, ctx: &mut Ctx) {
        self.digest_until(None, ctx);
    }

    fn digest_until(&mut self, until: Option<Key>, ctx: &mut Ctx) {
        while until.map_or(true, |k| !self.leaf_updater.is_in_scope(&k)) {
            match self
                .leaf_updater
                .digest(&mut self.branch_updater, &mut ctx.leaf_writer)
            {
                LeafDigestResult::Finished => {
                    self.digest_branches_until(until, ctx);
                    let Some(until) = until else { break };

                    // UNWRAP: branch updater base must be `Some` as an empty DB would never
                    // pass the loop condition if `until` is `Some`.
                    //
                    // UNWRAP: branch updater base must contain `until` as a postcondition of
                    // digest_branches_until.
                    self.reset_leaf_base(until, ctx).unwrap();
                }
                LeafDigestResult::NeedsMerge(key) => {
                    self.digest_branches_until(Some(key), ctx);

                    // UNWRAP: branch updater base must be `Some` as an empty DB would never
                    // pass the loop condition _and_ the merge key is always a known leaf.
                    //
                    // UNWRAP: branch updater base must contain `key` as a postcondition of
                    // digest_branches_until.
                    self.reset_leaf_base(key, ctx).unwrap();
                }
            }
        }
    }

    // post condition: if `until` is `Some`, `branch_updater`'s base is always set to the branch
    // containing the `until` key.
    fn digest_branches_until(&mut self, until: Option<Key>, ctx: &mut Ctx) {
        while until.map_or(true, |k| !self.branch_updater.is_in_scope(&k)) {
            let (old_branch, digest_result) =
                self.branch_updater
                    .digest(&mut ctx.bbn_index, &mut ctx.bnp, &mut ctx.bbn_writer);

            self.obsolete_branches.extend(old_branch);

            match digest_result {
                BranchDigestResult::Finished => {
                    let Some(until) = until else { break };
                    self.reset_branch_base(until, &*ctx);
                }
                BranchDigestResult::NeedsMerge(key) => {
                    self.reset_branch_base(key, &*ctx);
                }
            }
        }
    }

    // panics if branch updater base is not a branch containing the target.
    fn reset_leaf_base(&mut self, target: Key, ctx: &Ctx) -> Result<(), ()> {
        let branch = self.branch_updater.base().ok_or(())?;
        let (i, leaf_pn) = super::search_branch(&branch.node, target).ok_or(())?;
        let leaf = self
            .leaf_pages
            .remove(&leaf_pn)
            .unwrap_or_else(|| LeafNode {
                inner: ctx.leaf_reader.query(leaf_pn),
            });

        let separator = reconstruct_key(branch.node.prefix(), branch.node.separator(i));

        let cutoff = if branch.node.n() as usize > i + 1 {
            Some(reconstruct_key(
                branch.node.prefix(),
                branch.node.separator(i + 1),
            ))
        } else {
            self.branch_updater.cutoff()
        };

        let base_leaf = BaseLeaf {
            node: leaf,
            id: leaf_pn,
            iter_pos: 0,
            separator,
        };

        self.leaf_updater.reset_base(Some(base_leaf), cutoff);
        Ok(())
    }

    fn reset_branch_base(&mut self, target: Key, ctx: &Ctx) {
        let target_branch = ctx.bbn_index.lookup(target);
        let cutoff = ctx.bbn_index.next_after(target).map(|(k, _)| k);
        let base = target_branch.map(|id| {
            // UNWRAP: all nodes in index must exist.
            let node = ctx.bnp.checkout(id).unwrap();
            BaseBranch {
                id: id,
                node,
                iter_pos: 0,
            }
        });

        self.branch_updater.reset_base(base, cutoff);
    }
}

pub fn reconstruct_key(prefix: &BitSlice<u8, Msb0>, separator: &BitSlice<u8, Msb0>) -> Key {
    let mut key = [0u8; 32];
    key.view_bits_mut::<Msb0>()[..prefix.len()].copy_from_bitslice(prefix);
    key.view_bits_mut::<Msb0>()[prefix.len()..][..separator.len()].copy_from_bitslice(separator);
    key
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
        let Some(branch_id) = bbn_index.lookup(key) else {
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
                b.iter(|| super::reconstruct_key(prefix, separator))
            });
        }

        group.finish();
    }
}
