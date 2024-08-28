use bitvec::prelude::*;
use std::cmp::Ordering;

use crate::beatree::{
    allocator::PageNumber,
    leaf::{
        node::{self as leaf_node, LeafBuilder, LeafNode, LEAF_NODE_BODY_SIZE},
        store::LeafStoreWriter,
    },
    Key,
};

use super::{
    branch::BranchUpdater, LEAF_BULK_SPLIT_TARGET, LEAF_BULK_SPLIT_THRESHOLD, LEAF_MERGE_THRESHOLD,
};

pub struct BaseLeaf {
    pub node: LeafNode,
    pub id: PageNumber,
    pub iter_pos: usize,
    pub separator: Key,
}

impl BaseLeaf {
    fn next_key(&self) -> Option<Key> {
        if self.iter_pos >= self.node.n() as usize {
            None
        } else {
            Some(self.key(self.iter_pos))
        }
    }

    fn key(&self, i: usize) -> Key {
        self.node.key(i)
    }

    fn key_cell(&self, i: usize) -> (Key, &[u8], bool) {
        let (value, overflow) = self.node.value(i);
        (self.node.key(i), value, overflow)
    }

    fn next_cell(&self) -> (&[u8], bool) {
        self.node.value(self.iter_pos)
    }

    fn advance_iter(&mut self) {
        self.iter_pos += 1;
    }
}

enum LeafOp {
    Insert(Key, Vec<u8>, bool),
    Keep(usize, usize),
}

pub enum DigestResult {
    NeedsMerge(Key),
    Finished,
}

pub struct LeafUpdater {
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

impl LeafUpdater {
    pub fn new(base: Option<BaseLeaf>, cutoff: Option<Key>) -> Self {
        LeafUpdater {
            base,
            cutoff,
            separator_override: None,
            ops: Vec::new(),
            gauge: LeafGauge::default(),
            bulk_split: None,
        }
    }

    pub fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    pub fn reset_base(&mut self, base: Option<BaseLeaf>, cutoff: Option<Key>) {
        self.base = base;
        self.cutoff = cutoff;
    }

    /// Ingest a key/cell pair. Provide a callback which is called if this deletes an existing
    /// overflow cell.
    pub fn ingest(
        &mut self,
        key: Key,
        value_change: Option<Vec<u8>>,
        overflow: bool,
        with_deleted_overflow: impl FnMut(&[u8]),
    ) {
        self.keep_up_to(Some(&key), with_deleted_overflow);

        if let Some(value) = value_change {
            self.ops.push(LeafOp::Insert(key, value, overflow));
            self.bulk_split_step(self.ops.len() - 1);
        }
    }

    // If `NeedsMerge` is returned, `ops` are prepopulated with the merged values and
    // separator_override is set.
    // If `Finished` is returned, `ops` is guaranteed empty and separator_override is empty.
    pub fn digest(
        &mut self,
        branch_updater: &mut BranchUpdater,
        leaf_writer: &mut LeafStoreWriter,
    ) -> DigestResult {
        if let Some(ref base) = self.base {
            branch_updater.possibly_delete(base.separator);
        }

        // no cells are going to be deleted from this point onwards - this keeps everything.
        self.keep_up_to(None, |_| {});

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_leaves(branch_updater, leaf_writer);

        if let Some(ref base) = self.base {
            leaf_writer.release(base.id);
        }

        if self.gauge.body_size() == 0 {
            self.ops.clear();
            self.separator_override = None;

            DigestResult::Finished
        } else if self.gauge.body_size() > LEAF_NODE_BODY_SIZE {
            assert_eq!(
                last_ops_start, 0,
                "normal split can only occur when not bulk splitting"
            );
            self.split(branch_updater, leaf_writer)
        } else if self.gauge.body_size() >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = self.build_leaf(&self.ops[last_ops_start..]);
            let separator = self.separator();

            let pn = leaf_writer.write(node);
            branch_updater.ingest(separator, pn);

            self.ops.clear();
            self.gauge = LeafGauge::default();
            self.separator_override = None;
            DigestResult::Finished
        } else {
            // UNWRAP: if cutoff exists, then base must too.
            // merge is only performed when not at the rightmost leaf. this is protected by the
            // check on self.cutoff above.
            if self.separator_override.is_none() {
                self.separator_override = Some(self.base.as_ref().unwrap().separator);
            }

            self.prepare_merge_ops(last_ops_start);

            DigestResult::NeedsMerge(self.cutoff.unwrap())
        }
    }

    fn keep_up_to(&mut self, up_to: Option<&Key>, mut with_deleted_overflow: impl FnMut(&[u8])) {
        while let Some(next_key) = self.base.as_ref().and_then(|b| b.next_key()) {
            let Some(ref mut base_node) = self.base else {
                return;
            };
            let order = up_to
                .map(|up_to| up_to.cmp(&next_key))
                .unwrap_or(Ordering::Greater);
            if order == Ordering::Less {
                break;
            }

            let (val, overflow) = base_node.next_cell();
            if order == Ordering::Greater {
                self.ops.push(LeafOp::Keep(base_node.iter_pos, val.len()));
                base_node.advance_iter();
                self.bulk_split_step(self.ops.len() - 1);
            } else {
                if overflow {
                    with_deleted_overflow(val);
                }
                base_node.advance_iter();
            }
        }
    }

    // check whether bulk split needs to start, and if so, start it.
    // if ongoing, check if we need to cut off.
    fn bulk_split_step(&mut self, op_index: usize) {
        let item_size = match self.ops[op_index] {
            LeafOp::Keep(_, size) => size,
            LeafOp::Insert(_, ref val, _) => val.len(),
        };

        let body_size_after = self.gauge.body_size_after(item_size);
        match self.bulk_split {
            None if body_size_after >= LEAF_BULK_SPLIT_THRESHOLD => {
                self.bulk_split = Some(LeafBulkSplitter::default());
                self.gauge = LeafGauge::default();
                for i in 0..=op_index {
                    self.bulk_split_step(i);
                }
            }
            Some(ref mut bulk_splitter) if body_size_after >= LEAF_BULK_SPLIT_TARGET => {
                let accept_item = body_size_after <= LEAF_NODE_BODY_SIZE || {
                    if self.gauge.body_size() < LEAF_MERGE_THRESHOLD {
                        // super degenerate split! node grew from underfull to overfull in one
                        // item. only thing to do here is merge leftwards, unfortunately.
                        // save this for later to do another pass with.
                        todo!()
                    }

                    false
                };

                let n = if accept_item {
                    self.gauge.ingest(item_size);
                    op_index + 1 - bulk_splitter.total_count
                } else {
                    op_index - bulk_splitter.total_count
                };

                // push onto bulk splitter & restart gauge.
                self.gauge = LeafGauge::default();
                bulk_splitter.push(n);

                if !accept_item {
                    self.gauge.ingest(item_size);
                }
            }
            _ => self.gauge.ingest(item_size),
        }
    }

    fn build_bulk_splitter_leaves(
        &mut self,
        branch_updater: &mut BranchUpdater,
        leaf_writer: &mut LeafStoreWriter,
    ) -> usize {
        let Some(splitter) = self.bulk_split.take() else {
            return 0;
        };

        let mut start = 0;
        for item_count in splitter.items {
            let leaf_ops = &self.ops[start..][..item_count];

            let separator = if start == 0 {
                self.separator()
            } else {
                // UNWRAP: separator override is always set when more items follow after a bulk
                // split.
                self.separator_override.take().unwrap()
            };
            let new_node = self.build_leaf(leaf_ops);

            // set the separator override for the next
            if let Some(op) = self.ops.get(start + item_count) {
                let next = self.op_key(op);
                let last = new_node.key(new_node.n() - 1);
                self.separator_override = Some(separate(&last, &next));
            }

            // write the node and provide it to the branch above.
            let pn = leaf_writer.write(new_node);
            branch_updater.ingest(separator, pn);

            start += item_count;
        }

        start
    }

    fn separator(&self) -> Key {
        // the first leaf always gets a separator of all 0.
        self.separator_override
            .or(self.base.as_ref().map(|b| b.separator))
            .unwrap_or([0u8; 32])
    }

    fn split(
        &mut self,
        branch_updater: &mut BranchUpdater,
        leaf_writer: &mut LeafStoreWriter,
    ) -> DigestResult {
        let midpoint = self.gauge.body_size() / 2;
        let mut split_point = 0;

        let mut left_gauge = LeafGauge::default();
        while left_gauge.body_size() < midpoint {
            let item_size = match self.ops[split_point] {
                LeafOp::Keep(_, size) => size,
                LeafOp::Insert(_, ref val, _) => val.len(),
            };

            if left_gauge.body_size_after(item_size) > LEAF_NODE_BODY_SIZE {
                if left_gauge.body_size() < LEAF_MERGE_THRESHOLD {
                    // super degenerate split! jumped from underfull to overfull in a single step.
                    todo!()
                }

                break;
            }

            left_gauge.ingest(item_size);
            split_point += 1;
        }

        let left_ops = &self.ops[..split_point];
        let right_ops = &self.ops[split_point..];

        let left_key = self.op_key(&self.ops[split_point - 1]);
        let right_key = self.op_key(&self.ops[split_point]);

        let left_separator = self.separator();
        let right_separator = separate(&left_key, &right_key);

        let left_leaf = self.build_leaf(left_ops);

        let left_pn = leaf_writer.write(left_leaf);

        branch_updater.ingest(left_separator, left_pn);

        let mut right_gauge = LeafGauge::default();
        for op in &self.ops[split_point..] {
            let item_size = match op {
                LeafOp::Keep(_, size) => *size,
                LeafOp::Insert(_, ref val, _) => val.len(),
            };

            right_gauge.ingest(item_size);
        }

        if right_gauge.body_size() >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let right_leaf = self.build_leaf(right_ops);
            let right_pn = leaf_writer.write(right_leaf);
            branch_updater.ingest(right_separator, right_pn);

            self.ops.clear();
            self.gauge = LeafGauge::default();
            self.separator_override = None;

            DigestResult::Finished
        } else {
            // degenerate split: impossible to create two nodes with >50%. Merge remainder into
            // sibling node.

            self.separator_override = Some(right_separator);
            self.prepare_merge_ops(split_point);

            self.gauge = right_gauge;

            // UNWRAP: protected above.
            DigestResult::NeedsMerge(self.cutoff.unwrap())
        }
    }

    fn prepare_merge_ops(&mut self, split_point: usize) {
        self.ops.drain(..split_point);

        let Some(ref base) = self.base else { return };

        // then replace `Keep` ops with pure key-value ops, preparing for the base to be changed.
        for op in self.ops.iter_mut() {
            let LeafOp::Keep(i, _) = *op else { continue };
            let (k, v, o) = base.key_cell(i);
            *op = LeafOp::Insert(k, v.to_vec(), o);
        }
    }

    fn op_key(&self, leaf_op: &LeafOp) -> Key {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, _, _) => *k,
            LeafOp::Keep(i, _) => self.base.as_ref().unwrap().key(*i),
        }
    }

    fn op_cell<'a>(&'a self, leaf_op: &'a LeafOp) -> (Key, &'a [u8], bool) {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, v, o) => (*k, &v[..], *o),
            LeafOp::Keep(i, _) => self.base.as_ref().unwrap().key_cell(*i),
        }
    }

    fn build_leaf(&self, ops: &[LeafOp]) -> LeafNode {
        let total_value_size = ops
            .iter()
            .map(|op| match op {
                LeafOp::Keep(_, size) => *size,
                LeafOp::Insert(_, v, _) => v.len(),
            })
            .sum();

        let mut leaf_builder = LeafBuilder::new(ops.len(), total_value_size);
        for op in ops {
            let (k, v, o) = self.op_cell(op);

            leaf_builder.push_cell(k, v, o);
        }
        leaf_builder.finish()
    }
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

    fn body_size_after(&self, value_size: usize) -> usize {
        leaf_node::body_size(self.n + 1, self.value_size_sum + value_size)
    }

    fn body_size(&self) -> usize {
        leaf_node::body_size(self.n, self.value_size_sum)
    }
}

// separate two keys a and b where b > a
fn separate(a: &Key, b: &Key) -> Key {
    // if b > a at some point b must have a 1 where a has a 0 and they are equal up to that point.
    let len = a
        .view_bits::<Msb0>()
        .iter()
        .zip(b.view_bits::<Msb0>().iter())
        .take_while(|(a, b)| a == b)
        .count()
        + 1;

    let mut separator = [0u8; 32];
    separator.view_bits_mut::<Msb0>()[..len].copy_from_bitslice(&b.view_bits::<Msb0>()[..len]);
    separator
}

#[cfg(feature = "benchmarks")]
pub mod benches {
    use crate::beatree::benches::get_key_pair;
    use criterion::{BenchmarkId, Criterion};

    pub fn separate_benchmark(c: &mut Criterion) {
        let mut group = c.benchmark_group("separate");

        for shared_bytes in [0, 4, 8, 12, 16] {
            let (key1, key2) = get_key_pair(shared_bytes);
            group.bench_function(BenchmarkId::new("shared_bytes", shared_bytes), |b| {
                b.iter(|| super::separate(&key1, &key2));
            });
        }

        group.finish();
    }
}
