use anyhow::{bail, ensure, Result};
use bitvec::prelude::*;
use crossbeam_channel::Receiver;
use itertools::{Itertools, EitherOrBoth};
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::File,
    io::{ErrorKind, Read, Seek},
};

use crate::beatree::{
    allocator::PageNumber,
    bbn,
    branch::{self, BRANCH_NODE_SIZE, BRANCH_NODE_BODY_SIZE},
    index::Index, 
    leaf::{self, node::{LeafNode, LEAF_NODE_BODY_SIZE}}, Key,
};

use super::BranchId;

const BRANCH_MERGE_THRESHOLD: usize = BRANCH_NODE_BODY_SIZE / 2;

// At 180% of the branch size, we perform a 'bulk split' which follows a different algorithm
// than a simple split. Bulk splits are encountered when there are a large number of insertions
// on a single node, typically when inserting into a fresh database.
const BRANCH_BULK_SPLIT_THRESHOLD: usize = (BRANCH_NODE_BODY_SIZE * 9) / 5;

const LEAF_MERGE_THRESHOLD: usize = LEAF_NODE_BODY_SIZE / 2;
const LEAF_BULK_SPLIT_THRESHOLD: usize = (LEAF_NODE_BODY_SIZE * 9) / 5;

/// Change the btree in the specified way. Updates the branch index in-place and returns
/// a list of branches which have become obsolete.
///
/// The changeset is a list of key value pairs to be added or removed from the btree.
pub fn update(
    changeset: &BTreeMap<Key, Option<Vec<u8>>>,
    bbn_index: &mut Index,
    bnp: &mut branch::BranchNodePool,
    leaf_reader: &leaf::store::LeafStoreReader,
    leaf_writer: &mut leaf::store::LeafStoreWriter,
    bbn_store_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let mut updater = Updater::new(&*bbn_index, &*bnp, leaf_reader);
    for (key, value_change) in changeset {
        updater.ingest(*key, value_change.clone());
    }

    updater.complete();

    Ok(todo!())
}

struct Updater {
    active_branch: ActiveBranch,
    active_leaf: ActiveLeaf,
}

impl Updater {
    fn new(
        bbn_index: &Index,
        bnp: &branch::BranchNodePool,
        leaf_reader: &leaf::store::LeafStoreReader,
    ) -> Self {
        let first = bbn_index.first();

        // UNWRAP: all nodes in index must exist.
        let first_branch = first.as_ref().map(|(_, id)| bnp.checkout(*id).unwrap());
        let first_branch_cutoff = first.as_ref()
            .and_then(|(k, _)| bbn_index.next_after(*k))
            .map(|(k, _)| k);

        // first leaf cutoff is the separator of the second leaf _or_ the separator of the next
        // branch if there is only 1 leaf, or nothing.
        let first_leaf_cutoff = first_branch.as_ref().and_then(|node| if node.n() > 1 {
            Some(reconstruct_key(node.prefix(), node.separator(1)))
        } else {
            None
        }).or(first_branch_cutoff);

        let first_leaf = first_branch.as_ref()
            .map(|node| PageNumber::from(node.node_pointer(0)))
            .map(|id| (id, leaf_reader.query(id)));

        // active branch: first branch, cut-off second branch key.
        let active_branch = ActiveBranch {
            base: first_branch.map(|node| BaseBranch {
                // UNWRAP: node can only exist if ID does.
                id: first.as_ref().unwrap().1,
                node,
                iter_pos: 0,
            }),
            cutoff: first_branch_cutoff,
            ops: Vec::new(),
        };

        // active leaf: first leaf in first branch, cut-off second leaf key.
        let active_leaf = ActiveLeaf {
            base: first_leaf.map(|(id, node)| BaseLeaf {
                id,
                node,
                iter_pos: 0,
            }),
            cutoff: first_leaf_cutoff,
            ops: Vec::new(),
            gauge: LeafGauge::default(),
        };

        Updater {
            active_branch,
            active_leaf,
        }
    }

    fn ingest(&mut self, key: Key, value_change: Option<Vec<u8>>) {
        // This is a while loop because merges may require multiple iterations.
        while !self.active_leaf.is_in_scope(&key) {
            self.active_leaf.complete(&mut self.active_branch);
            // TODO: determine where the active leaf should be set next. if merging, it's just the
            // 'next' leaf. If done, it's the leaf for the key. Set active branch to the branch
            // for the active leaf.

            if self.active_branch.is_in_scope(&key) {
                continue
            }
            // TODO: complete branch. set active to relevant leaf and branch.
        }

        self.active_leaf.ingest(key, value_change);
    }

    fn complete(&mut self) {
        self.active_leaf.complete(&mut self.active_branch);
        self.active_branch.complete();
    }
}

struct BaseBranch {
    node: branch::BranchNode,
    id: branch::BranchId,
    iter_pos: usize,
}

struct ActiveBranch {
    // the 'base' node we are working from. does not exist if DB is empty.
    base: Option<BaseBranch>,
    // the cutoff key, which determines if an operation is in-scope.
    // does not exist for the last branch in the database.
    cutoff: Option<Key>,
    ops: Vec<(Key, Option<PageNumber>)>,
}

impl ActiveBranch {
    fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    fn complete(&mut self) {
        todo!()
    }
}

struct BaseLeaf {
    node: LeafNode,
    id: PageNumber,
    iter_pos: usize,
}

impl BaseLeaf {
    // How the key compares to the next key in the base node.
    // If the base node is exhausted `Less` is returned.
    fn cmp_next(&self, key: &Key) -> Ordering {
        if self.iter_pos >= self.node.n() {
            Ordering::Less
        } else {
            key.cmp(&self.node.key(self.iter_pos))
        }
    }

    fn next_value(&self) -> &[u8] {
        self.node.value(self.iter_pos)
    }

    fn advance_iter(&mut self) {
        self.iter_pos += 1;
    }
}

enum LeafOp {
    Insert(Key, Vec<u8>),
    Keep(usize),
}

struct ActiveLeaf {
    // the 'base' node we are working from. does not exist if DB is empty.
    base: Option<BaseLeaf>,
    // the cutoff key, which determines if an operation is in-scope.
    // does not exist for the last leaf in the database.
    cutoff: Option<Key>,
    ops: Vec<LeafOp>,
    gauge: LeafGauge,
}

impl ActiveLeaf {
    fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    fn ingest(&mut self, key: Key, value_change: Option<Vec<u8>>) {
        if let Some(ref mut base_node) = self.base {
            while base_node.cmp_next(&key) != Ordering::Less {
                let size = base_node.next_value().len();
                self.ops.push(LeafOp::Keep(base_node.iter_pos));
                base_node.advance_iter();

                self.gauge.ingest(size);

                if self.gauge.body_size() >= LEAF_BULK_SPLIT_THRESHOLD {
                    // TODO: big split.
                }
            }
        }

        if let Some(value) = value_change {
            self.ops.push(LeafOp::Insert(key, value));
        }

        if self.gauge.body_size() >= LEAF_BULK_SPLIT_THRESHOLD {
            // TODO: big split.
        }
    }

    fn complete(&mut self, active_branch: &mut ActiveBranch) {
    }
}

struct BranchGauge {
    first_separator: Option<Key>,
    prefix_len: usize,
    // total separator len, not counting prefix.
    separator_len: usize,
    n: usize,
}

impl BranchGauge {
    fn new() -> Self {
        BranchGauge {
            first_separator: None,
            prefix_len: 0,
            separator_len: 0,
            n: 0,
        }
    }

    fn ingest(&mut self, key: Key) {
        let Some(ref first) = self.first_separator else {
            self.first_separator = Some(key);
            self.separator_len = separator_len(&key);
            self.prefix_len = self.separator_len;

            self.n = 1;
            return
        };

        self.separator_len = std::cmp::max(separator_len(&key), self.separator_len);
        self.prefix_len = prefix_len(first, &key);
        self.n += 1;
    }

    fn body_size(&self) -> usize {
        branch::body_size(self.prefix_len, self.separator_len - self.prefix_len, self.n)
    }
}

#[derive(Default)]
struct LeafGauge {
    n: usize,
    value_size_sum: usize,
}

impl LeafGauge {
    fn ingest(&mut self, value_size: usize) {
        self.n += 1;
        self.value_size_sum += value_size;
    }

    fn body_size(&self) -> usize {
        leaf::node::body_size(self.n, self.value_size_sum)
    }
}

fn prefix_len(key_a: &Key, key_b: &Key) -> usize {
    key_a.view_bits::<Lsb0>()
        .iter()
        .zip(key_b.view_bits::<Lsb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
}

fn separator_len(key: &Key) -> usize {
    std::cmp::min(1, key.view_bits::<Lsb0>().last_one().unwrap_or(1))
}

fn reconstruct_key(
    prefix: &BitSlice<u8>,
    separator: &BitSlice<u8>,
) -> Key {
    let mut key = [0u8; 32];
    key.view_bits_mut::<Lsb0>()[..prefix.len()].copy_from_bitslice(prefix);
    key.view_bits_mut::<Lsb0>()[prefix.len()..][..separator.len()].copy_from_bitslice(prefix);
    key
}
