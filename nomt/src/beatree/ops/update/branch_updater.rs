use std::cmp::Ordering;
use std::sync::Arc;

use crate::beatree::{
    allocator::PageNumber,
    branch::{self as branch_node, BranchNode, BranchNodeBuilder, BRANCH_NODE_BODY_SIZE},
    ops::bit_ops::{prefix_len, separator_len},
    Key,
};
use crate::io::PagePool;

use super::{
    branch_stage::BranchChanges, get_key, BRANCH_BULK_SPLIT_TARGET, BRANCH_BULK_SPLIT_THRESHOLD,
    BRANCH_MERGE_THRESHOLD,
};

pub struct BaseBranch {
    pub node: Arc<BranchNode>,
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

    pub fn key(&self, i: usize) -> Key {
        get_key(&self.node, i)
    }

    fn key_value(&self, i: usize) -> (Key, PageNumber) {
        (self.key(i), self.node.node_pointer(i).into())
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
    ops: Vec<BranchOp>,
    // gauges total size of branch after ops applied.
    // if bulk split is undergoing, this just stores the total size of the last branch,
    // and the gauges for the previous branches are stored in `bulk_split`.
    gauge: BranchGauge,
    page_pool: PagePool,
    bulk_split: Option<BranchBulkSplitter>,
}

impl BranchUpdater {
    pub fn new(page_pool: PagePool, base: Option<BaseBranch>, cutoff: Option<Key>) -> Self {
        BranchUpdater {
            base,
            cutoff,
            ops: Vec::new(),
            gauge: BranchGauge::new(),
            page_pool,
            bulk_split: None,
        }
    }

    /// Ingest a key and page number into the branch updater.
    pub fn ingest(&mut self, key: Key, pn: Option<PageNumber>) {
        self.keep_up_to(Some(&key));
        if let Some(pn) = pn {
            self.ops.push(BranchOp::Insert(key, pn));
            self.bulk_split_step(self.ops.len() - 1);
        }
    }

    pub fn digest(&mut self, branch_changes: &mut BranchChanges) -> DigestResult {
        self.keep_up_to(None);

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_branches(branch_changes);

        if self.gauge.body_size() == 0 {
            self.ops.clear();

            DigestResult::Finished
        } else if self.gauge.body_size() > BRANCH_NODE_BODY_SIZE {
            assert_eq!(
                last_ops_start, 0,
                "normal split can only occur when not bulk splitting"
            );
            self.split(branch_changes)
        } else if self.gauge.body_size() >= BRANCH_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = self.build_branch(&self.ops[last_ops_start..], &self.gauge);
            let separator = self.op_key(&self.ops[last_ops_start]);
            branch_changes.insert(separator, node);

            self.ops.clear();
            self.gauge = BranchGauge::new();
            DigestResult::Finished
        } else {
            self.prepare_merge_ops(last_ops_start);

            DigestResult::NeedsMerge(self.cutoff.unwrap())
        }
    }

    pub fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    pub fn reset_base(&mut self, base: Option<BaseBranch>, cutoff: Option<Key>) {
        self.base = base;
        self.cutoff = cutoff;
    }

    fn keep_up_to(&mut self, up_to: Option<&Key>) {
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

            if order == Ordering::Greater {
                let separator_len = separator_len(&next_key);
                self.ops
                    .push(BranchOp::Keep(base_node.iter_pos, separator_len));

                base_node.advance_iter();
                self.bulk_split_step(self.ops.len() - 1);
            } else {
                base_node.advance_iter();
                break;
            }
        }
    }

    // check whether bulk split needs to start, and if so, start it.
    // if ongoing, check if we need to cut off.
    fn bulk_split_step(&mut self, op_index: usize) {
        // UNWRAP: `Keep` ops require separator len to exist.
        let (key, separator_len) = match self.ops[op_index] {
            BranchOp::Keep(i, separator_len) => (self.base.as_ref().unwrap().key(i), separator_len),
            BranchOp::Insert(k, _) => (k, separator_len(&k)),
        };

        let body_size_after = self.gauge.body_size_after(key, separator_len);
        match self.bulk_split {
            None if body_size_after >= BRANCH_BULK_SPLIT_THRESHOLD => {
                self.bulk_split = Some(BranchBulkSplitter::default());
                self.gauge = BranchGauge::new();
                for i in 0..=op_index {
                    self.bulk_split_step(i);
                }
            }
            Some(ref mut bulk_splitter) if body_size_after >= BRANCH_BULK_SPLIT_TARGET => {
                let accept_item = body_size_after <= BRANCH_NODE_BODY_SIZE || {
                    if self.gauge.body_size() < BRANCH_MERGE_THRESHOLD {
                        // rare case: body was artifically small due to long shared prefix.
                        // start applying items without prefix compression. we assume items are less
                        // than half the body size, so the next item should apply cleanly.
                        self.gauge.stop_prefix_compression();
                        true
                    } else {
                        false
                    }
                };

                let n = if accept_item {
                    self.gauge.ingest(key, separator_len);
                    op_index + 1 - bulk_splitter.total_count
                } else {
                    op_index - bulk_splitter.total_count
                };

                // push onto bulk splitter & restart gauge.
                let last_gauge = std::mem::replace(&mut self.gauge, BranchGauge::new());
                bulk_splitter.push(n, last_gauge);

                if !accept_item {
                    self.gauge.ingest(key, separator_len);
                }
            }
            _ => self.gauge.ingest(key, separator_len),
        }
    }

    fn build_bulk_splitter_branches(&mut self, branch_changes: &mut BranchChanges) -> usize {
        let Some(splitter) = self.bulk_split.take() else {
            return 0;
        };

        let mut start = 0;
        for (item_count, gauge) in splitter.items {
            let branch_ops = &self.ops[start..][..item_count];
            let separator = self.op_key(&self.ops[start]);
            let new_node = self.build_branch(branch_ops, &gauge);

            branch_changes.insert(separator, new_node);

            start += item_count;
        }

        start
    }

    fn split(&mut self, branch_changes: &mut BranchChanges) -> DigestResult {
        let midpoint = self.gauge.body_size() / 2;
        let mut split_point = 0;

        let mut left_gauge = BranchGauge::new();
        while left_gauge.body_size() < midpoint {
            let (key, separator_len) = match self.ops[split_point] {
                BranchOp::Keep(i, separator_len) => {
                    // UNWRAP: keep ops require base to exist.
                    let k = self.base.as_ref().unwrap().key(i);
                    (k, separator_len)
                }
                BranchOp::Insert(k, _) => (k, separator_len(&k)),
            };

            if left_gauge.body_size_after(key, separator_len) > BRANCH_NODE_BODY_SIZE {
                if left_gauge.body_size() < BRANCH_MERGE_THRESHOLD {
                    // rare case: body was artifically small due to long shared prefix.
                    // start applying items without prefix compression. we assume items are less
                    // than half the body size, so the next item should apply cleanly.
                    left_gauge.stop_prefix_compression();
                } else {
                    break;
                }
            }

            left_gauge.ingest(key, separator_len);
            split_point += 1;
        }

        let left_ops = &self.ops[..split_point];
        let right_ops = &self.ops[split_point..];

        let left_separator = self.op_key(&self.ops[0]);
        let right_separator = self.op_key(&self.ops[split_point]);

        let left_node = self.build_branch(left_ops, &left_gauge);

        branch_changes.insert(left_separator, left_node);

        let mut right_gauge = BranchGauge::new();
        for op in &self.ops[split_point..] {
            let (key, separator_len) = match op {
                BranchOp::Keep(i, separator_len) => {
                    // UNWRAP: keep ops require base to exist.
                    let k = self.base.as_ref().unwrap().key(*i);
                    (k, *separator_len)
                }
                BranchOp::Insert(k, _) => (*k, separator_len(&k)),
            };

            right_gauge.ingest(key, separator_len);
        }

        if right_gauge.body_size() >= BRANCH_MERGE_THRESHOLD || self.cutoff.is_none() {
            let right_node = self.build_branch(right_ops, &right_gauge);

            branch_changes.insert(right_separator, right_node);

            self.ops.clear();
            self.gauge = BranchGauge::new();

            DigestResult::Finished
        } else {
            // degenerate split: impossible to create two nodes with >50%. Merge remainder into
            // sibling node.

            self.prepare_merge_ops(split_point);
            self.gauge = right_gauge;

            // UNWRAP: protected above.
            DigestResult::NeedsMerge(self.cutoff.unwrap())
        }
    }

    fn build_branch(&self, ops: &[BranchOp], gauge: &BranchGauge) -> BranchNode {
        let branch = BranchNode::new_in(&self.page_pool);

        // UNWRAP: freshly allocated branch can always be checked out.
        let mut builder = BranchNodeBuilder::new(
            branch,
            gauge.n,
            gauge.prefix_compressed_items(),
            gauge.prefix_len,
        );

        for op in ops {
            match op {
                BranchOp::Insert(k, pn) => builder.push(*k, separator_len(k), pn.0),
                BranchOp::Keep(i, s) => {
                    let (k, pn) = self.base.as_ref().unwrap().key_value(*i);
                    builder.push(k, *s, pn.0);
                }
            }
        }

        builder.finish()
    }

    fn prepare_merge_ops(&mut self, split_point: usize) {
        self.ops.drain(..split_point);

        let Some(ref base) = self.base else { return };

        // then replace `Keep` ops with pure key-value ops, preparing for the base to be changed.
        for op in self.ops.iter_mut() {
            let BranchOp::Keep(i, _) = *op else { continue };
            let (k, pn) = base.key_value(i);
            *op = BranchOp::Insert(k, pn);
        }
    }

    fn op_key(&self, op: &BranchOp) -> Key {
        // UNWRAP: `Keep` ops require base to exist.
        match op {
            BranchOp::Insert(k, _) => *k,
            BranchOp::Keep(i, _) => self.base.as_ref().unwrap().key(*i),
        }
    }
}

struct BranchGauge {
    // key and length of the first separator if any
    first_separator: Option<(Key, usize)>,
    prefix_len: usize,
    // sum of all separator lengths.
    sum_separator_lengths: usize,
    // the number of items that are prefix compressed, paired with their total lengths prior
    // to compression. `None` means everything will be compressed.
    prefix_compressed: Option<(usize, usize)>,
    n: usize,
}

impl BranchGauge {
    fn new() -> Self {
        BranchGauge {
            first_separator: None,
            prefix_len: 0,
            sum_separator_lengths: 0,
            prefix_compressed: None,
            n: 0,
        }
    }

    fn ingest(&mut self, key: Key, len: usize) {
        let Some((ref first, _)) = self.first_separator else {
            self.first_separator = Some((key, len));
            self.prefix_len = len;

            self.n = 1;
            return;
        };

        if self.prefix_compressed.is_none() {
            self.prefix_len = prefix_len(first, &key);
        }
        self.sum_separator_lengths += len;
        self.n += 1;
    }

    fn stop_prefix_compression(&mut self) {
        assert!(self.prefix_compressed.is_none());
        self.prefix_compressed = Some((self.n, self.sum_separator_lengths));
    }

    fn prefix_compressed_items(&self) -> usize {
        self.prefix_compressed.map(|(k, _)| k).unwrap_or(self.n)
    }

    fn total_separator_lengths(&self, prefix_len: usize) -> usize {
        match self.first_separator {
            Some((_, first_len)) => {
                let (prefix_compressed_items, pre_compression_lengths) = self
                    .prefix_compressed
                    .unwrap_or((self.n, self.sum_separator_lengths));

                let prefix_uncompressed_lengths =
                    self.sum_separator_lengths - pre_compression_lengths;

                // first length can be less than the shared prefix due to trailing zero compression.
                // then add the lengths of the compressed items after compression.
                // then add the lengths of the uncompressed items.
                first_len.saturating_sub(prefix_len) + pre_compression_lengths
                    - (prefix_compressed_items - 1) * prefix_len
                    + prefix_uncompressed_lengths
            }
            None => 0,
        }
    }

    fn body_size_after(&mut self, key: Key, len: usize) -> usize {
        let p;
        let t;
        if let Some((ref first, _)) = self.first_separator {
            if self.prefix_compressed.is_none() {
                p = prefix_len(first, &key);
                t = self.total_separator_lengths(p) + len.saturating_sub(p);
            } else {
                p = self.prefix_len;
                t = self.total_separator_lengths(p) + len;
            }
        } else {
            t = 0;
            p = len;
        }

        branch_node::body_size(p, t, self.n + 1)
    }

    fn body_size(&self) -> usize {
        branch_node::body_size(
            self.prefix_len,
            self.total_separator_lengths(self.prefix_len),
            self.n,
        )
    }
}

#[derive(Default)]
struct BranchBulkSplitter {
    items: Vec<(usize, BranchGauge)>,
    total_count: usize,
}

impl BranchBulkSplitter {
    fn push(&mut self, count: usize, gauge: BranchGauge) {
        self.items.push((count, gauge));
        self.total_count += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beatree::ops::bit_ops::separator_len;

    #[test]
    fn gauge_stop_uncompressed() {
        let mut gauge = BranchGauge::new();

        gauge.ingest([0; 32], 0);

        // push items with a long (16-byte) shared prefix until just before the halfway point.
        let mut items: Vec<Key> = (1..1000u16)
            .map(|i| {
                let mut key = [0; 32];
                key[16..18].copy_from_slice(&i.to_le_bytes());
                key
            })
            .collect();

        items.sort();

        for item in items {
            let len = separator_len(&item);
            if gauge.body_size_after(item, len) >= BRANCH_MERGE_THRESHOLD {
                break;
            }

            gauge.ingest(item, len);
        }

        assert!(gauge.body_size() < BRANCH_MERGE_THRESHOLD);

        // now insert an item that collapses the prefix, causing the previously underfull node to
        // become overfull.
        let unprefixed_key = [0xff; 32];
        assert!(gauge.body_size_after(unprefixed_key, 256) > BRANCH_NODE_BODY_SIZE);

        // stop compression. now we can accept more items without collapsing the prefix.
        gauge.stop_prefix_compression();
        assert!(gauge.body_size_after(unprefixed_key, 256) < BRANCH_NODE_BODY_SIZE);
    }
}
