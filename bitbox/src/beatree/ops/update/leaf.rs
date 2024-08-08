
use bitvec::prelude::*;
use std::{
    cmp::Ordering,
    io::{Read},
};

use crate::beatree::{
    allocator::PageNumber, 
    leaf::{
        node::{self as leaf_node, LeafNode, LeafBuilder, LEAF_NODE_BODY_SIZE},
        store::{LeafStoreWriter},
    }, Key,
};

use super::{
    branch::ActiveBranch, LEAF_MERGE_THRESHOLD, LEAF_BULK_SPLIT_TARGET, LEAF_BULK_SPLIT_THRESHOLD,
};

pub struct BaseLeaf {
    pub node: LeafNode,
    pub id: PageNumber,
    pub iter_pos: usize,
    pub separator: Key,
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

pub enum LeafCompletionStatus {
    NeedsMerge(Key),
    Finished,
}

pub struct ActiveLeaf {
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
    pub fn new(base: Option<BaseLeaf>, cutoff: Option<Key>) -> Self {
        ActiveLeaf {
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

    pub fn ingest(&mut self, key: Key, value_change: Option<Vec<u8>>) {
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

        // If `NeedsMerge` is returned, `ops` are prepopulated with the merged values and
    // separator_override is set.
    // If `Finished` is returned, `ops` is guaranteed empty and separator_override is empty.
    pub fn complete(
        &mut self, 
        active_branch: &mut ActiveBranch, 
        leaf_writer: &mut LeafStoreWriter,
    ) -> LeafCompletionStatus {
        let _body_size = self.gauge.body_size();

        if let Some(ref base) = self.base {
            active_branch.possibly_delete(base.separator);
        }

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_leaves(active_branch, leaf_writer);

        if let Some(ref base) = self.base {
            leaf_writer.release(base.id);
        }

        if self.gauge.body_size() == 0 {
            self.ops.clear();
            self.separator_override = None;

            LeafCompletionStatus::Finished
        } else if self.gauge.body_size() > LEAF_NODE_BODY_SIZE {
            assert_eq!(last_ops_start, 0, "normal split can only occur when not bulk splitting");
            self.split(active_branch, leaf_writer)
        } else if self.gauge.body_size() >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = self.build_leaf(&self.ops);
            let pn = leaf_writer.allocate(node);
            let separator = self.separator();
            
            active_branch.ingest(separator, pn);

            self.ops.clear();
            self.separator_override = None;
            LeafCompletionStatus::Finished
        } else {
            // UNWRAP: if cutoff exists, then base must too.
            // merge is only performed when not at the rightmost leaf. this is protected by the
            // check on self.cutoff above.
            if self.separator_override.is_none() {
                self.separator_override = Some(self.base.as_ref().unwrap().separator);
            }

            self.prepare_merge_ops(last_ops_start);

            LeafCompletionStatus::NeedsMerge(self.cutoff.unwrap())
        }
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
    }

    fn build_bulk_splitter_leaves(
        &mut self, 
        active_branch: &mut ActiveBranch,
        leaf_writer: &mut LeafStoreWriter,
    ) -> usize {
        let Some(splitter) = self.bulk_split.take() else { return 0 };

        let mut start = 0;
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
            let new_node = self.build_leaf(leaf_ops);

            // set the separator override for the next 
            if let Some(op) = self.ops.get(start + item_count + 1) {
                let next = self.op_key(op);
                let last = new_node.key(new_node.n() - 1);
                self.separator_override = Some(separate(&last, &next));
            }

            // write the node and provide it to the branch above.
            let pn = leaf_writer.allocate(new_node);
            active_branch.ingest(separator, pn);
        }

        start
    }

    fn separator(&self) -> Key {
        // the first leaf always gets a separator of all 0.
        self.separator_override.or(self.base.as_ref().map(|b| b.separator)).unwrap_or([0u8; 32])
    }

    fn split(&mut self, active_branch: &mut ActiveBranch, leaf_writer: &mut LeafStoreWriter) 
        -> LeafCompletionStatus
    {
        let midpoint = self.gauge.body_size() / 2;
        let mut left_size = 0;
        let mut split_point = 0;

        while left_size < midpoint {
            let item_size = match self.ops[split_point] {
                LeafOp::Keep(_, size) => size,
                LeafOp::Insert(_, ref val) => val.len(),
            };

            left_size += item_size;
            split_point += 1;
        }

        let left_ops = &self.ops[..split_point];
        let right_ops = &self.ops[split_point..];

        let left_key = self.op_key(&self.ops[split_point - 1]);
        let right_key = self.op_key(&self.ops[split_point]);

        let _left_node = self.build_leaf(left_ops);
        let right_separator = separate(&left_key, &right_key);

        let left_leaf = self.build_leaf(left_ops);
        let left_pn = leaf_writer.allocate(left_leaf);

        let left_separator = self.separator();

        active_branch.ingest(left_separator, left_pn);

        if self.gauge.body_size() - left_size >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let right_leaf = self.build_leaf(right_ops);
            let right_pn = leaf_writer.allocate(right_leaf);
            active_branch.ingest(right_separator, right_pn);

            LeafCompletionStatus::Finished
        } else {
            // degenerate split: impossible to create two nodes with >50%. Merge remainder into
            // sibling node.

            self.separator_override = Some(right_separator);
            self.prepare_merge_ops(split_point);

            // UNWRAP: protected above.
            LeafCompletionStatus::NeedsMerge(self.cutoff.unwrap())
        }
    }

    fn prepare_merge_ops(&mut self, split_point: usize) {
        // copy the left over operations to the front of the vector.
        let count = self.ops.len() - split_point;
        for i in 0..count {
            self.ops.swap(i, i + split_point);
        }
        self.ops.truncate(count);

        let Some(ref base) = self.base else { return };

        // then replace `Keep` ops with pure key-value ops, preparing for the base to be changed.
        for op in self.ops.iter_mut() {
            let LeafOp::Keep(i, _) = *op else { continue };
            let (k, v) = base.key_value(i);
            *op = LeafOp::Insert(k, v.to_vec());
        }
    }

    fn op_key(&self, leaf_op: &LeafOp) -> Key {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, _) => *k,
            LeafOp::Keep(i, _) => self.base.as_ref().unwrap().key(*i),
        }
    }

    fn op_key_value<'a>(&'a self, leaf_op: &'a LeafOp) -> (Key, &'a [u8]) {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, v) => (*k, &v[..]),
            LeafOp::Keep(i, _) => self.base.as_ref().unwrap().key_value(*i),
        }
    }

    fn build_leaf(&self, ops: &[LeafOp]) -> LeafNode {
        let mut leaf_builder = LeafBuilder::new(ops.len());
        for op in ops {
            let (k, v) = self.op_key_value(op);
    
            leaf_builder.push(k, v);
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

    fn body_size(&self) -> usize {
        leaf_node::body_size(self.n, self.value_size_sum)
    }
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
