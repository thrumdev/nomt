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
    leaf::{
        self, 
        node::{LeafNode, LeafBuilder, LEAF_NODE_BODY_SIZE},
        store::{LeafStoreReader, LeafStoreWriter},
    }, Key,
};

use super::BranchId;

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
    bnp: &mut branch::BranchNodePool,
    leaf_reader: &LeafStoreReader,
    leaf_writer: &mut LeafStoreWriter,
    bbn_store_writer: &mut bbn::BbnStoreWriter,
) -> Result<Vec<BranchId>> {
    let mut updater = Updater::new(&*bbn_index, &*bnp, leaf_reader);
    for (key, value_change) in changeset {
        updater.ingest(*key, value_change.clone(), leaf_writer);
    }

    updater.complete(leaf_writer);

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
        leaf_reader: &LeafStoreReader,
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
            .map(|node| (
                PageNumber::from(node.node_pointer(0)), 
                reconstruct_key(node.prefix(), node.separator(0)),
            ))
            .map(|(id, separator)| BaseLeaf {
                id,
                node: leaf_reader.query(id),
                iter_pos: 0,
                separator,
            });

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
            base: first_leaf,
            cutoff: first_leaf_cutoff,
            separator_override: None,
            ops: Vec::new(),
            gauge: LeafGauge::default(),
            bulk_split: None,
        };

        Updater {
            active_branch,
            active_leaf,
        }
    }

    fn ingest(
        &mut self, 
        key: Key, 
        value_change: Option<Vec<u8>>,
        leaf_writer: &mut LeafStoreWriter,
    ) {
        // This is a while loop because merges may require multiple iterations.
        while !self.active_leaf.is_in_scope(&key) {
            self.active_leaf.complete(&mut self.active_branch, leaf_writer);
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

    fn complete(&mut self, leaf_writer: &mut LeafStoreWriter) {
        self.active_leaf.complete(&mut self.active_branch, leaf_writer);
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

    fn ingest(&mut self, key: Key, pn: Option<PageNumber>) {
        todo!()
    } 

    fn complete(&mut self) {
        todo!()
    }
}

struct BaseLeaf {
    node: LeafNode,
    id: PageNumber,
    iter_pos: usize,
    separator: Key,
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

    fn key(&self, i: usize) -> Key {
        self.node.key(i)
    }

    fn key_value(&self, i: usize) -> (Key, &[u8]) {
        (self.node.key(i), self.node.value(i))
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
    Keep(usize, usize),
}

struct ActiveLeaf {
    // the 'base' node we are working from. does not exist if DB is empty.
    base: Option<BaseLeaf>,
    // the cutoff key, which determines if an operation is in-scope.
    // does not exist for the last leaf in the database.
    cutoff: Option<Key>,
    // a separator override. this is set as `Some` either as part of a bulk split or when the
    // leaf is having values merged in from some earlier node.
    separator_override: Option<Key>,
    ops: Vec<LeafOp>,
    // gauges total size of leaf after ops applied. 
    // if bulk split is undergoing, this just stores the total size of the last leaf,
    // and the gauges for the previous leaves are stored in `bulk_split`.
    gauge: LeafGauge,
    bulk_split: Option<LeafBulkSplitter>,
}

impl ActiveLeaf {
    fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    fn begin_bulk_split(&mut self) {
        let mut splitter = LeafBulkSplitter::default();

        let mut n = 0;
        let mut gauge = LeafGauge::default();
        for op in &self.ops {
            match op {
                LeafOp::Insert(_, val) => gauge.ingest(val.len()),
                LeafOp::Keep(_, val_size) => gauge.ingest(*val_size),
            }

            n += 1;

            if gauge.body_size() >= LEAF_BULK_SPLIT_TARGET {
                splitter.push(n);
                n = 0;
                gauge = LeafGauge::default();
            }
        }

        self.gauge = gauge;
        self.bulk_split = Some(splitter);
    }

    // check whether bulk split needs to start, and if so, start it.
    // if ongoing, check if we need to cut off.
    fn bulk_split_step(&mut self) {
        match self.bulk_split {
            None if self.gauge.body_size() >= LEAF_BULK_SPLIT_THRESHOLD => {
                self.begin_bulk_split();
            },
            Some(ref mut bulk_splitter) if self.gauge.body_size() >= LEAF_BULK_SPLIT_TARGET => {
                // push onto bulk splitter & restart gauge.
                self.gauge = LeafGauge::default();
                let n = self.ops.len() - bulk_splitter.total_count;
                bulk_splitter.push(n);
            },
            _ => {}
        }

        // TODO: set separator override based on last key from bulk split and first key if inserted.
    }

    fn ingest(&mut self, key: Key, value_change: Option<Vec<u8>>) {
        loop {
            if let Some(ref mut base_node) = self.base {
                if base_node.cmp_next(&key) == Ordering::Less {
                    break
                }

                let size = base_node.next_value().len();
                self.ops.push(LeafOp::Keep(base_node.iter_pos, size));
                base_node.advance_iter();

                self.gauge.ingest(size);
            } else {
                break
            }

            self.bulk_split_step();
        }

        if let Some(value) = value_change {
            self.ops.push(LeafOp::Insert(key, value));
        }

        self.bulk_split_step();
    }

    fn build_bulk_splitter_leaves(
        &mut self, 
        active_branch: &mut ActiveBranch,
        leaf_writer: &mut LeafStoreWriter,
    ) -> (usize, bool) {
        let Some(mut splitter) = self.bulk_split.take() else { return (0, false) };

        let mut start = 0;
        let mut old_separator_deleted = false;
        for item_count in splitter.items {
            let leaf_ops = &self.ops[start..][..item_count];
            start += item_count;

            let separator = if start == 0 {
                self.separator_override
                    .or(self.base.as_ref().map(|base| base.separator))
                    .unwrap_or([0u8; 32])
            } else {
                // UNWRAP: separator override is always set when more items follow after a bulk
                // split.
                self.separator_override.take().unwrap()
            };
            let new_node = build_leaf(self.base.as_ref(), leaf_ops);

            // if our split creates a node with a separator greater than the old separator of the
            // node and we haven't yet deleted the old one, do so now.
            if !old_separator_deleted && self.base.as_ref()
                .map_or(false, |b| b.separator < separator) 
            {
                old_separator_deleted = true;

                // UNWRAP: just checked existence above.
                active_branch.ingest(self.base.as_ref().map(|b| b.separator).unwrap(), None);
                leaf_writer.release(self.base.as_ref().map(|b| b.id).unwrap());
            }

            // set the separator override for the next 
            if let Some(op) = self.ops.get(start + item_count + 1) {
                let next = match op {
                    LeafOp::Insert(k, _) => *k,
                    LeafOp::Keep(i, _) => self.base.as_ref().unwrap().key(*i),
                };

                let last = new_node.key(new_node.n() - 1);
                self.separator_override = Some(separate(&last, &next));
            }

            // write the node and provide it to the branch above.
            let pn = leaf_writer.allocate(new_node);
            active_branch.ingest(separator, Some(pn));

        }

        (start, old_separator_deleted)
    }

    fn complete(&mut self, active_branch: &mut ActiveBranch, leaf_writer: &mut LeafStoreWriter) {
        let body_size = self.gauge.body_size();

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let (last_ops_start, old_separator_deleted) 
            = self.build_bulk_splitter_leaves(active_branch, leaf_writer);

        if let Some(ref base) = self.base {
            leaf_writer.release(base.id);
        }

        if self.gauge.body_size() == 0 {
            return
        } else if self.gauge.body_size() > LEAF_NODE_BODY_SIZE {
            // TODO: split
        } else if self.gauge.body_size() > LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = build_leaf(self.base.as_ref(), &self.ops);
            let pn = leaf_writer.allocate(node);
            let separator = self.separator_override
                .or(self.base.as_ref().map(|b| b.separator))
                .unwrap_or([0u8; 32]); // first leaf always gets separator of all 0.
            
            active_branch.ingest(separator, Some(pn));
        } else {
            // merge is only performed when there is a following node to merge with
            
        }
    }
}

fn build_leaf(base: Option<&BaseLeaf>, ops: &[LeafOp]) -> LeafNode {
    let leaf = LeafNode::zeroed();
    let mut leaf_builder = LeafBuilder::new(leaf, ops.len());
    for op in ops {
        // UNWRAP: base must exist if a Keep op is present.
        let (k, v) = match op {
            LeafOp::Keep(i, _) => base.as_ref().unwrap().key_value(*i),
            LeafOp::Insert(k, v) => (*k, &v[..]),
        };

        leaf_builder.push(k, v);
    }
    leaf_builder.finish()
}

#[derive(Default)]
struct LeafBulkSplitter {
    items: Vec<usize>,
    total_count: usize,
}

impl LeafBulkSplitter {
    fn push(&mut self, count: usize) {
        self.items.push(count);
        self.total_count += count;
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

// separate two keys a and b where b > a
fn separate(a: &Key, b: &Key) -> Key {
    // if b > a at some point b must have a 1 where a has a 0 and they are equal up to that point.
    let len = a.view_bits::<Lsb0>()
        .iter()
        .zip(b.view_bits::<Lsb0>().iter())
        .take_while(|(a, b)| a == b)
        .count() + 1;

    let mut separator = [0u8; 32];
    separator.view_bits_mut::<Lsb0>()[..len].copy_from_bitslice(&b.view_bits::<Lsb0>());
    separator
}
