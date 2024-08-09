
use bitvec::prelude::*;

use std::cmp::Ordering;

use crate::beatree::{
    allocator::PageNumber,
    branch::{self as branch_node, BranchNode}, Key,
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

    pub fn digest(&mut self) -> DigestResult {
        todo!()
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
