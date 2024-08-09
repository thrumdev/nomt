
use bitvec::prelude::*;

use std::cmp::Ordering;

use crate::beatree::{
    allocator::PageNumber,
    bbn::BbnStoreWriter,
    branch::{self as branch_node, BranchNode, BranchNodePool, BRANCH_NODE_BODY_SIZE},
    index::Index,
    Key,
};

use super::{reconstruct_key, BranchId, BRANCH_MERGE_THRESHOLD, BRANCH_BULK_SPLIT_TARGET, BRANCH_BULK_SPLIT_THRESHOLD};

pub struct BaseBranch {
    pub node: BranchNode,
    pub id: BranchId,
    pub iter_pos: usize,
}

impl BaseBranch {
    fn next_key(&self) -> Option<Key> {
        if self.iter_pos >= self.node.n() as usize {
            None
        } else {
            Some(self.key(self.iter_pos))
        }
    }

    fn key(&self, i: usize) -> Key {
        reconstruct_key(self.node.prefix(), self.node.separator(i))
    }

    fn key_value(&self, i: usize) -> (Key, PageNumber) {
        (self.key(i), self.node.node_pointer(i).into())
    }

    fn separator(&self) -> Key {
        self.key(0)
    }

    fn advance_iter(&mut self) {
        self.iter_pos += 1;
    }
}

enum BranchOp {
    Insert(Key, PageNumber),
    Keep(usize, usize),
}

pub enum DigestResult {
    Finished,
    NeedsMerge(Key),
}

pub struct BranchUpdater {
    // the 'base' node we are working from. does not exist if DB is empty.
    base: Option<BaseBranch>,
    // the cutoff key, which determines if an operation is in-scope.
    // does not exist for the last branch in the database.
    cutoff: Option<Key>,
    // a separator override. this is set as `Some` either as part of a bulk split or when the
    // leaf is having values merged in from some earlier node.
    separator_override: Option<Key>,
    // separator keys which are to be possibly deleted and will be at the point that another
    // separator greater than it is ingested.
    possibly_deleted: Vec<Key>,
    ops: Vec<BranchOp>,
    // gauges total size of branch after ops applied.
    // if bulk split is undergoing, this just stores the total size of the last branch,
    // and the gauges for the previous branches are stored in `bulk_split`.
    gauge: BranchGauge,
    bulk_split: Option<BranchBulkSplitter>,
}

impl BranchUpdater {
    pub fn new(base: Option<BaseBranch>, cutoff: Option<Key>) -> Self {
        BranchUpdater {
            base,
            cutoff,
            possibly_deleted: Vec::new(),
            separator_override: None,
            ops: Vec::new(),
            gauge: BranchGauge::new(),
            bulk_split: None,
        }
    }

    /// Ingest a key and page number into the branch updater. If this returns `NeedsBranch`,
    /// then digest, reset the base, and attempt again.
    pub fn ingest(&mut self, key: Key, pn: PageNumber) {
        self.keep_up_to(Some(&key));
        self.ops.push(BranchOp::Insert(key, pn));
        self.gauge.ingest(key, separator_len(&key));
        self.bulk_split_step();
    }

    pub fn base(&self) -> Option<&BaseBranch> {
        self.base.as_ref()
    }

    pub fn cutoff(&self) -> Option<Key> {
        self.cutoff
    }

    pub fn digest(
        &mut self,
        bbn_index: &mut Index,
        bnp: &mut BranchNodePool,
        bbn_writer: &mut BbnStoreWriter,
    ) -> (Option<BranchId>, DigestResult) {
        if let Some(ref base) = self.base {
            bbn_index.remove(&base.separator());
            bbn_writer.release(base.node.bbn_pn().into());
        }

        let old_branch_id = self.base.as_ref().map(|b| b.id);

        self.keep_up_to(None);

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_branches();

        if self.gauge.body_size() == 0 {
            self.ops.clear();
            self.separator_override = None;

            (old_branch_id, DigestResult::Finished)
        } else if self.gauge.body_size() > BRANCH_NODE_BODY_SIZE {
            assert_eq!(last_ops_start, 0, "normal split can only occur when not bulk splitting");
            (old_branch_id, self.split())
        } else if self.gauge.body_size() >= BRANCH_MERGE_THRESHOLD || self.cutoff.is_none() {
            let (branch_id, node) = self.build_branch(&self.ops, &self.gauge, bnp);
            let separator = self.separator();

            bbn_index.insert(separator, branch_id);
            bbn_writer.allocate(node);

            self.ops.clear();
            self.gauge = BranchGauge::new();
            self.separator_override = None;
            (old_branch_id, DigestResult::Finished)
        } else {
            // UNWRAP: if cutoff exists, then base must too.
            // merge is only performed when not at the rightmost leaf. this is protected by the
            // check on self.cutoff above.
            if self.separator_override.is_none() {
                self.separator_override = Some(self.base.as_ref().unwrap().separator());
            }

            self.prepare_merge_ops(last_ops_start);

            (old_branch_id, DigestResult::NeedsMerge(self.cutoff.unwrap()))
        }
    }

    pub fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    pub fn possibly_delete(&mut self, key: Key) {
        self.possibly_deleted.push(key);
    }

    pub fn reset_base(&mut self, base: Option<BaseBranch>, cutoff: Option<Key>) {
        self.base = base;
        self.cutoff = cutoff;
        self.possibly_deleted.clear();
    }

    fn separator(&self) -> Key {
        self.separator_override
            .or_else(|| self.base.as_ref().map(|b| b.separator()))
            .unwrap_or([0u8; 32])
    }

    fn keep_up_to(&mut self, up_to: Option<&Key>) {
        while let Some(next_key) = self.base.as_ref().and_then(|b| b.next_key()) {
            let Some(ref mut base_node) = self.base else { return };
            if up_to.map_or(false, |up_to| up_to.cmp(&next_key) != Ordering::Greater) {
                break
            }

            if self.possibly_deleted.first() == Some(&next_key) {
                self.possibly_deleted.remove(0);
                base_node.advance_iter();
                continue;
            }

            // TODO: is this right? if the key is equal to `up_to` we should advance the iterator
            // but not keep.

            let separator_len = separator_len(&next_key);
            self.ops.push(BranchOp::Keep(base_node.iter_pos, separator_len));
            base_node.advance_iter();

            self.gauge.ingest(next_key, separator_len);

            self.bulk_split_step();
        }
    }

    fn begin_bulk_split(&mut self) {
        let mut splitter = BranchBulkSplitter::default();

        let mut n = 0;
        let mut gauge = BranchGauge::new();
        for op in &self.ops {
            match op {
                BranchOp::Insert(key, pn) => gauge.ingest(*key, separator_len(&key)),
                BranchOp::Keep(i, separator_len) => gauge.ingest(
                    // UNWRAP: `Keep` ops require base node to exist.
                    self.base.as_ref().unwrap().key(*i),
                    *separator_len,
                ),
            }

            n += 1;

            if gauge.body_size() >= BRANCH_BULK_SPLIT_TARGET {
                splitter.push(n);
                n = 0;
                gauge = BranchGauge::new();
            }
        }

        self.gauge = gauge;
        self.bulk_split = Some(splitter);
    }

    // check whether bulk split needs to start, and if so, start it.
    // if ongoing, check if we need to cut off.
    fn bulk_split_step(&mut self) {
        match self.bulk_split {
            None if self.gauge.body_size() >= BRANCH_BULK_SPLIT_THRESHOLD => {
                self.begin_bulk_split();
            },
            Some(ref mut bulk_splitter) if self.gauge.body_size() >= BRANCH_BULK_SPLIT_TARGET => {
                // push onto bulk splitter & restart gauge.
                self.gauge = BranchGauge::new();
                let n = self.ops.len() - bulk_splitter.total_count;
                bulk_splitter.push(n);
            },
            _ => {}
        }
    }

    fn build_bulk_splitter_branches(&mut self) -> usize {
        todo!()
    }

    fn split(&mut self) -> DigestResult {
        todo!()
    }

    fn build_branch(
        &self,
        ops: &[BranchOp],
        gauge: &BranchGauge,
        bnp: &mut BranchNodePool,
    ) -> (BranchId, BranchNode) {
        todo!()
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
            let BranchOp::Keep(i, _) = *op else { continue };
            let (k, pn) = base.key_value(i);
            *op = BranchOp::Insert(k, pn);
        }
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

    fn ingest(&mut self, key: Key, len: usize) {
        let Some(ref first) = self.first_separator else {
            self.first_separator = Some(key);
            self.separator_len = len;
            self.prefix_len = self.separator_len;

            self.n = 1;
            return
        };

        self.separator_len = std::cmp::max(len, self.separator_len);
        self.prefix_len = prefix_len(first, &key);
        self.n += 1;
    }

    fn body_size(&self) -> usize {
        branch_node::body_size(self.prefix_len, self.separator_len - self.prefix_len, self.n)
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
    let key = &key.view_bits::<Lsb0>();
    std::cmp::min(1, key.len() - key.trailing_zeros())
}

#[derive(Default)]
struct BranchBulkSplitter {
    items: Vec<usize>,
    total_count: usize,
}

impl BranchBulkSplitter {
    fn push(&mut self, count: usize) {
        self.items.push(count);
        self.total_count += count;
    }
}
