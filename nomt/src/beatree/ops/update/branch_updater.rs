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
    get_key, BRANCH_BULK_SPLIT_TARGET, BRANCH_BULK_SPLIT_THRESHOLD, BRANCH_MERGE_THRESHOLD,
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

/// A callback which takes ownership of newly created leaves.
pub trait HandleNewBranch {
    fn handle_new_branch(&mut self, separator: Key, node: BranchNode, cutoff: Option<Key>);
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

    pub fn digest(&mut self, new_branches: &mut impl HandleNewBranch) -> DigestResult {
        self.keep_up_to(None);

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // branch of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_branches(new_branches);

        if self.gauge.body_size() == 0 {
            self.ops.clear();

            DigestResult::Finished
        } else if self.gauge.body_size() > BRANCH_NODE_BODY_SIZE {
            assert_eq!(
                last_ops_start, 0,
                "normal split can only occur when not bulk splitting"
            );
            self.split(new_branches)
        } else if self.gauge.body_size() >= BRANCH_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = self.build_branch(&self.ops[last_ops_start..], &self.gauge);
            let separator = self.op_key(&self.ops[last_ops_start]);
            new_branches.handle_new_branch(separator, node, self.cutoff);

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

    pub fn remove_cutoff(&mut self) {
        self.cutoff = None;
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

    fn build_bulk_splitter_branches(&mut self, new_branches: &mut impl HandleNewBranch) -> usize {
        let Some(splitter) = self.bulk_split.take() else {
            return 0;
        };

        let mut start = 0;
        for (item_count, gauge) in splitter.items {
            let branch_ops = &self.ops[start..][..item_count];
            let separator = self.op_key(&self.ops[start]);
            let new_node = self.build_branch(branch_ops, &gauge);

            new_branches.handle_new_branch(separator, new_node, self.cutoff);

            start += item_count;
        }

        start
    }

    fn split(&mut self, new_branches: &mut impl HandleNewBranch) -> DigestResult {
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

        new_branches.handle_new_branch(left_separator, left_node, Some(right_separator));

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

        if right_gauge.body_size() > BRANCH_NODE_BODY_SIZE {
            // This is a rare case left uncovered by the bulk split, the threshold to activate it
            // has not been reached by the sum of all left and right operations. Now the right
            // node is too big, and another split is required to be executed
            self.ops.drain(..split_point);
            self.gauge = right_gauge;
            self.split(new_branches)
        } else if right_gauge.body_size() >= BRANCH_MERGE_THRESHOLD || self.cutoff.is_none() {
            let right_node = self.build_branch(right_ops, &right_gauge);

            new_branches.handle_new_branch(right_separator, right_node, self.cutoff);

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
pub mod tests {
    use super::{
        get_key, prefix_len, Arc, BaseBranch, BranchGauge, BranchNode, BranchNodeBuilder,
        BranchUpdater, DigestResult, HandleNewBranch, Key, PageNumber, PagePool,
        BRANCH_MERGE_THRESHOLD, BRANCH_NODE_BODY_SIZE,
    };
    use crate::beatree::ops::bit_ops::separator_len;
    use std::collections::HashMap;

    lazy_static::lazy_static! {
        static ref PAGE_POOL: PagePool = PagePool::new();
    }

    #[derive(Default)]
    struct TestHandleNewBranch {
        inner: HashMap<Key, (BranchNode, Option<Key>)>,
    }

    impl HandleNewBranch for TestHandleNewBranch {
        fn handle_new_branch(&mut self, separator: Key, node: BranchNode, cutoff: Option<Key>) {
            self.inner.insert(separator, (node, cutoff));
        }
    }

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

    fn prefixed_key(prefix_byte: u8, prefix_len: usize, i: usize) -> Key {
        let mut k = [0u8; 32];
        for x in k.iter_mut().take(prefix_len) {
            *x = prefix_byte;
        }
        k[prefix_len..prefix_len + 2].copy_from_slice(&(i as u16).to_be_bytes());
        k
    }

    fn make_raw_branch(vs: Vec<(Key, usize)>) -> BranchNode {
        let n = vs.len();
        let prefix_len = if vs.len() == 1 {
            separator_len(&vs[0].0)
        } else {
            prefix_len(&vs[0].0, &vs[vs.len() - 1].0)
        };

        let branch = BranchNode::new_in(&PAGE_POOL);
        let mut builder = BranchNodeBuilder::new(branch, n, n, prefix_len);
        for (k, pn) in vs {
            builder.push(k, separator_len(&k), pn as u32);
        }

        builder.finish()
    }

    fn make_branch(vs: Vec<(Key, usize)>) -> Arc<BranchNode> {
        Arc::new(make_raw_branch(vs))
    }

    fn make_branch_with_body_size_target(
        mut key: impl FnMut(usize) -> Key,
        mut body_size_predicate: impl FnMut(usize) -> bool,
    ) -> Arc<BranchNode> {
        let mut gauge = BranchGauge::new();
        let mut items = Vec::new();
        loop {
            let next_key = key(items.len());
            let s_len = separator_len(&next_key);

            let size = gauge.body_size_after(next_key, s_len);
            if !body_size_predicate(size) {
                break;
            }
            items.push((next_key, items.len()));
            gauge.ingest(next_key, s_len);
        }

        make_branch(items)
    }

    // Make a branch node with the specified bbn_pn.
    // Use keys until they are present in the iterator
    // or stop if the body size target is reached.
    pub fn make_branch_until(
        keys: &mut impl Iterator<Item = Key>,
        body_size_target: usize,
        bbn_pn: u32,
    ) -> Arc<BranchNode> {
        let mut gauge = BranchGauge::new();
        let mut items = Vec::new();
        loop {
            let Some(next_key) = keys.next() else {
                break;
            };

            let s_len = separator_len(&next_key);

            let size = gauge.body_size_after(next_key, s_len);
            if size >= body_size_target {
                break;
            }

            items.push((next_key, items.len()));
            gauge.ingest(next_key, s_len);
        }

        let mut branch_node = make_raw_branch(items);
        branch_node.set_bbn_pn(bbn_pn);
        Arc::new(branch_node)
    }

    #[test]
    fn is_in_scope() {
        let mut updater = BranchUpdater::new(PAGE_POOL.clone(), None, None);
        assert!(updater.is_in_scope(&[0xff; 32]));

        updater.reset_base(None, Some([0xfe; 32]));
        assert!(updater.is_in_scope(&[0xf0; 32]));
        assert!(updater.is_in_scope(&[0xfd; 32]));
        assert!(!updater.is_in_scope(&[0xfe; 32]));
        assert!(!updater.is_in_scope(&[0xff; 32]));
    }

    #[test]
    fn update() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch((0..500).map(|i| (key(i), i)).collect());

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        updater.ingest(key(250), Some(9999.into()));
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry = new_branches.inner.get(&key(0)).unwrap();

        let new_branch = &new_branch_entry.0;
        assert_eq!(new_branch.n(), 500);
        assert_eq!(new_branch.node_pointer(0), 0);
        assert_eq!(new_branch.node_pointer(499), 499);
        assert_eq!(new_branch.node_pointer(250), 9999);
    }

    #[test]
    fn insert_rightsized() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch((0..500).map(|i| (key(i * 2), i)).collect());

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        updater.ingest(key(251), Some(9999.into()));
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry = new_branches.inner.get(&key(0)).unwrap();

        let new_branch = &new_branch_entry.0;
        assert_eq!(new_branch.n(), 501);
        assert_eq!(new_branch.node_pointer(0), 0);
        assert_eq!(new_branch.node_pointer(500), 499);
        assert_eq!(new_branch.node_pointer(126), 9999);
    }

    #[test]
    fn insert_overflowing() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch_with_body_size_target(key, |size| size <= BRANCH_NODE_BODY_SIZE);
        let n = branch.n() as usize;

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        updater.ingest(key(n), Some(PageNumber(n as u32)));
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry_1 = new_branches.inner.get(&key(0)).unwrap();
        let new_branch_1 = &new_branch_entry_1.0;

        let new_branch_entry_2 = new_branches
            .inner
            .get(&key(new_branch_1.n() as usize))
            .unwrap();
        let new_branch_2 = &new_branch_entry_2.0;

        assert_eq!(new_branch_1.node_pointer(0), 0);
        assert_eq!(
            new_branch_2.node_pointer((new_branch_2.n() - 1) as usize),
            n as u32
        );
    }

    #[test]
    fn delete() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch((0..500).map(|i| (key(i), i)).collect());

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        updater.ingest(key(250), None);
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry = new_branches.inner.get(&key(0)).unwrap();

        let new_branch = &new_branch_entry.0;
        assert_eq!(new_branch.n(), 499);
        assert_eq!(new_branch.node_pointer(0), 0);
        assert_eq!(new_branch.node_pointer(498), 499);
    }

    #[test]
    fn delete_underflow_and_merge() {
        let key = |i| prefixed_key(0xff, 5, i);
        let key2 = |i| prefixed_key(0xff, 6, i);

        let mut rightsized = false;
        let branch = make_branch_with_body_size_target(key, |size| {
            let res = !rightsized;
            rightsized = size >= BRANCH_MERGE_THRESHOLD;
            res
        });
        rightsized = false;
        let branch2 = make_branch_with_body_size_target(key2, |size| {
            let res = !rightsized;
            rightsized = size >= BRANCH_MERGE_THRESHOLD;
            res
        });

        let n = branch.n() as usize;
        let n2 = branch2.n() as usize;

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            Some(key2(0)),
        );
        let mut new_branches = TestHandleNewBranch::default();

        // delete all except the first
        for i in 1..n {
            updater.ingest(key(i), None);
        }
        let DigestResult::NeedsMerge(_) = updater.digest(&mut new_branches) else {
            panic!()
        };

        updater.reset_base(
            Some(BaseBranch {
                node: branch2,
                iter_pos: 0,
            }),
            None,
        );
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry = new_branches.inner.get(&key(0)).unwrap();

        let new_branch = &new_branch_entry.0;
        assert_eq!(new_branch.n() as usize, n2 + 1);
        assert_eq!(new_branch.node_pointer(0), 0);
        assert_eq!(new_branch.node_pointer(n2), (n2 - 1) as u32);
    }

    #[test]
    fn delete_completely() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch((0..500).map(|i| (key(i), i)).collect());

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        for i in 0..500 {
            updater.ingest(key(i), None);
        }
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        assert!(new_branches.inner.get(&key(0)).is_none());
    }

    #[test]
    fn delete_underflow_rightmost() {
        let key = |i| prefixed_key(0x11, 5, i);
        let branch = make_branch((0..500).map(|i| (key(i), i)).collect());

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            None,
        );
        let mut new_branches = TestHandleNewBranch::default();

        for i in 0..499 {
            updater.ingest(key(i), None);
        }
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry = new_branches.inner.get(&key(499)).unwrap();

        let new_branch = &new_branch_entry.0;
        assert_eq!(new_branch.n(), 1);
    }

    #[test]
    fn shared_prefix_collapse() {
        let key = |i| prefixed_key(0x00, 24, i);
        let key2 = |i| prefixed_key(0xff, 24, i);

        let mut rightsized = false;
        let branch = make_branch_with_body_size_target(key, |size| {
            let res = !rightsized;
            rightsized = size >= BRANCH_MERGE_THRESHOLD;
            res
        });
        rightsized = false;
        let branch2 = make_branch_with_body_size_target(key2, |size| {
            let res = !rightsized;
            rightsized = size >= BRANCH_MERGE_THRESHOLD;
            res
        });

        let n = branch.n() as usize;
        let n2 = branch2.n() as usize;

        let mut updater = BranchUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseBranch {
                node: branch,
                iter_pos: 0,
            }),
            Some(key2(0)),
        );
        let mut new_branches = TestHandleNewBranch::default();

        // delete the last item, causing a situation where prefix compression needs to be
        // disabled.
        updater.ingest(key(n - 1), None);
        let DigestResult::NeedsMerge(_) = updater.digest(&mut new_branches) else {
            panic!()
        };

        updater.reset_base(
            Some(BaseBranch {
                node: branch2,
                iter_pos: 0,
            }),
            None,
        );
        let DigestResult::Finished = updater.digest(&mut new_branches) else {
            panic!()
        };

        let new_branch_entry_1 = new_branches.inner.get(&key(0)).unwrap();
        let new_branch_1 = &new_branch_entry_1.0;

        // first item has no shared prefix with any other key, causing the size to balloon.
        assert!(new_branch_1.prefix_compressed() != new_branch_1.n());

        assert_eq!(
            get_key(&new_branch_1, new_branch_1.n() as usize - 1),
            key2(0)
        );

        let branch_1_body_size = {
            let mut gauge = BranchGauge::new();
            for i in 0..new_branch_1.n() as usize {
                let key = get_key(&new_branch_1, i);
                gauge.ingest(key, separator_len(&key))
            }
            gauge.body_size()
        };
        assert!(branch_1_body_size >= BRANCH_MERGE_THRESHOLD);

        let new_branch_entry_2 = new_branches.inner.get(&key2(1)).unwrap();
        let new_branch_2 = &new_branch_entry_2.0;

        assert_eq!(new_branch_2.n() + new_branch_1.n(), (n + n2 - 1) as u16);
        assert_eq!(
            new_branch_2.node_pointer(new_branch_2.n() as usize - 1),
            n2 as u32 - 1
        );
    }
}
