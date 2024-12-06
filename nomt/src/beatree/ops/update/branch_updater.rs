use std::{ops::Range, sync::Arc};

use crate::beatree::{
    allocator::PageNumber,
    branch::{self as branch_node, node, BranchNode, BranchNodeBuilder, BRANCH_NODE_BODY_SIZE},
    ops::{
        bit_ops::{prefix_len, separator_len},
        find_key_pos,
    },
    Key,
};
use crate::io::PagePool;

use super::{
    get_key, BRANCH_BULK_SPLIT_TARGET, BRANCH_BULK_SPLIT_THRESHOLD, BRANCH_MERGE_THRESHOLD,
};

pub struct BaseBranch {
    node: Arc<BranchNode>,
    low: usize,
}

impl BaseBranch {
    pub fn new(node: Arc<BranchNode>) -> Self {
        BaseBranch { node, low: 0 }
    }

    // Try to find the given key starting from `self.low` up to the end.
    // Returns None if `self.low` is already at the end of the node,
    // or if there are no keys left bigger than the specified one.
    // If there are available keys in the node, then it returns the index
    // of the specified key with the boolean set to true or the index containing
    // the first key bigger than the one specified and the boolean set to false.
    fn find_key(&mut self, key: &Key) -> Option<(bool, usize)> {
        if self.low == self.node.n() as usize {
            return None;
        }

        let (found, pos) = find_key_pos(&self.node, key, Some(self.low));

        if found {
            // the key was present return its index and point to the right after key
            self.low = pos + 1;
            return Some((true, pos));
        } else if pos == self.low {
            // there are no keys left bigger than the specified one
            return None;
        } else {
            // key was not present, return and point to the smallest bigger key
            self.low = pos;
            return Some((false, pos));
        }
    }

    pub fn key(&self, i: usize) -> Key {
        get_key(&self.node, i)
    }

    fn key_value(&self, i: usize) -> (Key, PageNumber) {
        (self.key(i), self.node.node_pointer(i).into())
    }
}

// BranchOp used to create a new node starting off a possible base node.
//
// `KeepChunk` refers to only compressed separator present into the base node.
// Every uncompressed separator taken from the base node or that will be inserted
// into the new node will be represented as an `Insert`
enum BranchOp {
    // Separator and its page number
    Insert(Key, PageNumber),
    // Contains a range of separators that will be transfered unchanged
    // in the new node from the base.
    //
    // These are always items which were prefix-compressed.
    KeepChunk(KeepChunk),
}

#[derive(Debug, Clone, Copy)]
struct KeepChunk {
    start: usize,
    end: usize,
    sum_separator_lengths: usize,
}

impl KeepChunk {
    fn len(&self) -> usize {
        self.end - self.start
    }
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
        // keep all elements that are skipped looking for `key`
        self.keep_up_to(Some(&key));

        if let Some(pn) = pn {
            // a new key or a previous uncompressed separator has been updated
            self.ops.push(BranchOp::Insert(key, pn));
            self.bulk_split_step(self.ops.len() - 1);
        };
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
            let separator = self.op_first_key(&self.ops[last_ops_start]);
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

    // Advance the base looking for `up_to`, stops if a bigger Key is found or the end is reached.
    // Collect in `self.ops` all separators that are skipped.
    // Returns the index at which 'up_to' was found, otherwise, returns None.
    fn keep_up_to(&mut self, up_to: Option<&Key>) {
        if self.base.is_none() {
            // empty db
            return;
        }

        let (maybe_chunk, uncompressed_range) = {
            // UNWRAP: self.base is not None
            let base = self.base.as_mut().unwrap();

            let from = base.low;
            let base_n = base.node.n() as usize;

            let (_found, to) = match up_to {
                // Nothing more to do, the end has already been reached
                None if from == base_n => return,
                // Jump directly to the end of the base node and update `base.low` accordingly
                None => {
                    base.low = base_n;
                    (false, base_n)
                }
                // TODO: maybe totally remove found from `find_key`?
                Some(up_to) => match base.find_key(up_to) {
                    Some(res) => res,
                    // already at the end
                    None => return,
                },
            };

            if from == to {
                // nothing to keep
                return;
            }

            let base_compressed_end = std::cmp::min(to, base.node.prefix_compressed() as usize);

            let maybe_chunk = if from == base_compressed_end {
                None
            } else {
                Some(KeepChunk {
                    start: from,
                    end: base_compressed_end,
                    sum_separator_lengths: node::uncompressed_separator_range_size(
                        base.node.prefix_len() as usize,
                        base.node.separator_range_len(from, base_compressed_end),
                        base_compressed_end - from,
                        separator_len(&base.key(from)),
                    ),
                })
            };

            (maybe_chunk, base_compressed_end..to)
        };

        // push compressed chunk
        if let Some(chunk) = maybe_chunk {
            self.ops.push(BranchOp::KeepChunk(chunk));
            self.bulk_split_step(self.ops.len() - 1);
        }

        // convert every kept uncompressed separator into an Insert operation
        for i in uncompressed_range {
            // UNWRAP: self.base is not None
            let (key, pn) = self.base.as_ref().unwrap().key_value(i);
            self.ops.push(BranchOp::Insert(key, pn));
            self.bulk_split_step(self.ops.len() - 1);
        }
    }

    // check whether bulk split needs to start, and if so, start it.
    // If ongoing, check if we need to cut off.
    // returns the amount of operations consumed for the bulk creation
    fn bulk_split_step(&mut self, op_index: usize) -> usize {
        // UNWRAPs: if the operation is Update or KeepChunk, then it must be a base to keep the chunk from
        let body_size_after = match self.ops[op_index] {
            BranchOp::KeepChunk(ref chunk) => self
                .gauge
                .body_size_after_chunk(self.base.as_ref().unwrap(), &chunk),
            BranchOp::Insert(key, _) => self.gauge.body_size_after(key, separator_len(&key)),
        };

        match self.bulk_split {
            None if body_size_after >= BRANCH_BULK_SPLIT_THRESHOLD => {
                self.bulk_split = Some(BranchBulkSplitter::default());
                self.gauge = BranchGauge::new();
                let mut idx = 0;
                while idx < self.ops.len() {
                    let consumed = self.bulk_split_step(idx);
                    idx += consumed;
                }
                idx
            }
            Some(ref mut bulk_splitter) if body_size_after >= BRANCH_BULK_SPLIT_TARGET => {
                let mut recursive_on_split = false;

                let accept_item = body_size_after <= BRANCH_NODE_BODY_SIZE || {
                    match self.ops[op_index] {
                        BranchOp::Insert(..) => {
                            if self.gauge.body_size() < BRANCH_MERGE_THRESHOLD {
                                // rare case: body was artifically small due to long shared prefix.
                                // start applying items without prefix compression. we assume items are less
                                // than half the body size, so the next item should apply cleanly.
                                self.gauge.stop_prefix_compression();
                                true
                            } else {
                                false
                            }
                        }
                        BranchOp::KeepChunk(chunk) => {
                            let left_n_items = split_keep_chunk(
                                self.base.as_ref().unwrap(),
                                &self.gauge,
                                &mut self.ops,
                                op_index,
                                BRANCH_NODE_BODY_SIZE,
                                BRANCH_NODE_BODY_SIZE,
                            );

                            if left_n_items > 0 {
                                // KeepChunk has been successfully split, the left chunk is now able to fit
                                recursive_on_split = true;
                                true
                            } else {
                                // Not even a single element was able to fit.
                                if self.gauge.body_size() < BRANCH_MERGE_THRESHOLD {
                                    // We can stop prefix compression and separate the first
                                    // element of the keep_chunk into its own.
                                    let (key, pn) =
                                        self.base.as_ref().unwrap().key_value(chunk.start);
                                    let separator_len = separator_len(&key);
                                    self.ops[op_index] = BranchOp::KeepChunk(KeepChunk {
                                        start: chunk.start + 1,
                                        end: chunk.end,
                                        sum_separator_lengths: chunk.sum_separator_lengths
                                            - separator_len,
                                    });

                                    self.ops.insert(op_index, BranchOp::Insert(key, pn));

                                    self.gauge.stop_prefix_compression();
                                    recursive_on_split = true;
                                    true
                                } else {
                                    false
                                }
                            }
                        }
                    }
                };

                let n = if accept_item {
                    self.gauge.ingest_branch_op(&self.base, &self.ops[op_index]);
                    op_index + 1 - bulk_splitter.total_count
                } else {
                    op_index - bulk_splitter.total_count
                };

                // push onto bulk splitter & restart gauge.
                let last_gauge = std::mem::replace(&mut self.gauge, BranchGauge::new());
                bulk_splitter.push(n, last_gauge);

                if !accept_item {
                    self.gauge.ingest_branch_op(&self.base, &self.ops[op_index]);
                }

                if recursive_on_split {
                    1 + self.bulk_split_step(op_index + 1)
                } else {
                    1
                }
            }
            _ => {
                self.gauge.ingest_branch_op(&self.base, &self.ops[op_index]);
                1
            }
        }
    }

    fn build_bulk_splitter_branches(&mut self, new_branches: &mut impl HandleNewBranch) -> usize {
        let Some(splitter) = self.bulk_split.take() else {
            return 0;
        };

        let mut start = 0;
        for (item_count, gauge) in splitter.items {
            let branch_ops = &self.ops[start..][..item_count];
            let separator = self.op_first_key(&self.ops[start]);
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
            match *&self.ops[split_point] {
                BranchOp::Insert(key, _) => {
                    if left_gauge.body_size_after(key, separator_len(&key)) > BRANCH_NODE_BODY_SIZE
                    {
                        if left_gauge.body_size() < BRANCH_MERGE_THRESHOLD {
                            // rare case: body was artifically small due to long shared prefix.
                            // start applying items without prefix compression. we assume items are less
                            // than half the body size, so the next item should apply cleanly.
                            left_gauge.stop_prefix_compression();
                        } else {
                            break;
                        }
                    }
                }
                BranchOp::KeepChunk(chunk) => {
                    // try to split the chunk to make it fit into the available space
                    let n_items = split_keep_chunk(
                        self.base.as_ref().unwrap(),
                        &left_gauge,
                        &mut self.ops,
                        split_point,
                        midpoint,
                        BRANCH_NODE_BODY_SIZE,
                    );

                    if n_items == 0 {
                        // if no item from the chunk is capable to fit then
                        // try to extract the first one and check if it fits
                        // falling into the rare case of BranchOp::Insert
                        let (key, pn) = self.base.as_ref().unwrap().key_value(chunk.start);
                        let separator_len = separator_len(&key);
                        self.ops[split_point] = BranchOp::KeepChunk(KeepChunk {
                            start: chunk.start + 1,
                            end: chunk.end,
                            sum_separator_lengths: chunk.sum_separator_lengths - separator_len,
                        });
                        self.ops.insert(split_point, BranchOp::Insert(key, pn));

                        continue;
                    }
                }
            };

            left_gauge.ingest_branch_op(&self.base, &self.ops[split_point]);
            split_point += 1;
        }

        let left_ops = &self.ops[..split_point];
        let right_ops = &self.ops[split_point..];

        let left_separator = self.op_first_key(&self.ops[0]);
        let right_separator = self.op_first_key(&self.ops[split_point]);

        let left_node = self.build_branch(left_ops, &left_gauge);

        new_branches.handle_new_branch(left_separator, left_node, Some(right_separator));

        let mut right_gauge = BranchGauge::new();
        for op in right_ops {
            right_gauge.ingest_branch_op(&self.base, op);
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

        let Some(base) = self.base.as_ref() else {
            // SAFETY: If no base is avaialble, then all ops are expected to be `BranchOp::Insert`
            for op in ops {
                match op {
                    BranchOp::Insert(key, pn) => builder.push(*key, separator_len(key), pn.0),
                    _ => panic!("Unextected BranchOp creating a BranchNode without BaseBranch"),
                }
            }
            return builder.finish();
        };

        for op in ops {
            match op {
                BranchOp::Insert(key, pn) => {
                    builder.push(*key, separator_len(key), pn.0);
                }
                BranchOp::KeepChunk(chunk) => {
                    builder.push_chunk(&base.node, chunk.start, chunk.end);
                }
            }
        }

        builder.finish()
    }

    fn prepare_merge_ops(&mut self, split_point: usize) {
        self.ops.drain(..split_point);

        let Some(ref base) = self.base else { return };

        // replace `KeepChunk` ops with pure key-value ops,
        // preparing for the base to be changed.
        let mut new_insert = 0;
        for i in 0..self.ops.len() {
            match self.ops[new_insert + i] {
                BranchOp::Insert(_, _) => (),
                BranchOp::KeepChunk(chunk) => {
                    self.ops.remove(new_insert + i);

                    for pos in (chunk.start..chunk.end).into_iter().rev() {
                        let (key, pn) = base.key_value(pos);
                        self.ops.insert(new_insert + i, BranchOp::Insert(key, pn));
                    }
                    new_insert += chunk.end - chunk.start - 1;
                }
            }
        }
    }

    fn op_first_key(&self, branch_op: &BranchOp) -> Key {
        // UNWRAP: `KeepChunk` leaf ops only exist when base is `Some`.
        match branch_op {
            BranchOp::Insert(k, _) => *k,
            BranchOp::KeepChunk(chunk) => self.base.as_ref().unwrap().key(chunk.start),
        }
    }
}

// Given a vector of `BranchOp`, try to split the `index` operation,
// which is expected to be KeepChunk, into two halves,
// targeting a `target` size and and not exceeding a `limit`.
//
// `target` and `limit` are required to understand when to accept a split
// with a final size smaller than the target. Constraining the split to always
// be bigger than the target causes the update algorithm to frequently
// fall into underfull to overfull scenarios.
fn split_keep_chunk(
    base: &BaseBranch,
    gauge: &BranchGauge,
    ops: &mut Vec<BranchOp>,
    index: usize,
    target: usize,
    limit: usize,
) -> usize {
    let BranchOp::KeepChunk(chunk) = ops[index] else {
        panic!("Attempted to split non `BranchOp::KeepChunk` operation");
    };

    let mut left_chunk_n_items = 0;
    let mut left_chunk_sum_separator_lengths = 0;
    let mut gauge = gauge.clone();
    for i in chunk.start..chunk.end {
        left_chunk_n_items += 1;

        let key = get_key(&base.node, i);
        let separator_len = separator_len(&key);
        let body_size_after = gauge.body_size_after(key, separator_len);

        if body_size_after >= target {
            // if an item jumps from below the target to bigger then the limit, do not use it
            if body_size_after > limit {
                left_chunk_n_items -= 1;
            } else {
                gauge.ingest_key(key, separator_len);
                left_chunk_sum_separator_lengths += separator_len;
            }
            break;
        }
        left_chunk_sum_separator_lengths += separator_len;
        gauge.ingest_key(key, separator_len);
    }

    // if none or all elements are taken then nothing needs to be changed
    if left_chunk_n_items != 0 && chunk.len() != left_chunk_n_items {
        let left_chunk = KeepChunk {
            start: chunk.start,
            end: chunk.start + left_chunk_n_items,
            sum_separator_lengths: left_chunk_sum_separator_lengths,
        };

        let right_chunk = KeepChunk {
            start: chunk.start + left_chunk_n_items,
            end: chunk.end,
            sum_separator_lengths: chunk.sum_separator_lengths - left_chunk_sum_separator_lengths,
        };

        ops.insert(index, BranchOp::KeepChunk(left_chunk));
        ops[index + 1] = BranchOp::KeepChunk(right_chunk);
    }

    left_chunk_n_items
}

#[derive(Clone)]
struct BranchGauge {
    // key and length of the first separator if any
    first_separator: Option<(Key, usize)>,
    prefix_len: usize,
    // sum of all separator lengths (not including the first key).
    sum_separator_lengths: usize,
    // the number of items that are prefix compressed.`None` means everything will be compressed.
    prefix_compressed: Option<usize>,
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

    fn ingest_key(&mut self, key: Key, len: usize) {
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

    fn ingest_branch_op(&mut self, base: &Option<BaseBranch>, op: &BranchOp) {
        // UNWRAPs: if op is Update or KeepChunk, then it must be a base
        match op {
            BranchOp::KeepChunk(ref chunk) => {
                // UNWRAP: `KeepChunk` requires the base branch to exist.
                self.ingest_chunk(base.as_ref().unwrap(), chunk);
            }
            BranchOp::Insert(key, _) => {
                self.ingest_key(*key, separator_len(key));
            }
        }
    }

    fn ingest_chunk(&mut self, base: &BaseBranch, chunk: &KeepChunk) {
        if let Some((ref first, _)) = self.first_separator {
            if self.prefix_compressed.is_none() {
                let chunk_last_key = base.key(chunk.end - 1);
                self.prefix_len = prefix_len(first, &chunk_last_key);
            }
            self.sum_separator_lengths += chunk.sum_separator_lengths;
            self.n += chunk.len();
        } else {
            let chunk_first_key = base.key(chunk.start);
            let chunk_last_key = base.key(chunk.end - 1);
            let first_separator_len = separator_len(&chunk_first_key);

            self.prefix_len = prefix_len(&chunk_first_key, &chunk_last_key);
            self.first_separator = Some((chunk_first_key, first_separator_len));
            self.sum_separator_lengths = chunk.sum_separator_lengths - first_separator_len;
            self.n = chunk.len();
        };
    }

    fn stop_prefix_compression(&mut self) {
        assert!(self.prefix_compressed.is_none());
        self.prefix_compressed = Some(self.n);
    }

    fn prefix_compressed_items(&self) -> usize {
        self.prefix_compressed.unwrap_or(self.n)
    }

    fn total_separator_lengths(&self, prefix_len: usize) -> usize {
        match self.first_separator {
            Some((_, first_len)) => node::compressed_separator_range_size(
                first_len,
                self.prefix_compressed.unwrap_or(self.n),
                self.sum_separator_lengths,
                prefix_len,
            ),
            None => 0,
        }
    }

    fn body_size_after(&mut self, key: Key, len: usize) -> usize {
        let p;
        let t;
        if let Some((ref first, first_len)) = self.first_separator {
            if self.prefix_compressed.is_none() {
                p = prefix_len(first, &key);
            } else {
                p = self.prefix_len;
            }
            t = node::compressed_separator_range_size(
                first_len,
                self.prefix_compressed.unwrap_or(self.n + 1),
                self.sum_separator_lengths + len,
                p,
            );
        } else {
            t = 0;
            p = len;
        }

        branch_node::body_size(p, t, self.n + 1)
    }

    fn body_size_after_chunk(&self, base: &BaseBranch, chunk: &KeepChunk) -> usize {
        let p;
        let t;
        if let Some((ref first, first_len)) = self.first_separator {
            if self.prefix_compressed.is_none() {
                let chunk_last_key = base.key(chunk.end - 1);
                p = prefix_len(first, &chunk_last_key);
            } else {
                p = self.prefix_len;
            }
            t = node::compressed_separator_range_size(
                first_len,
                self.prefix_compressed.unwrap_or(self.n + chunk.len()),
                self.sum_separator_lengths + chunk.sum_separator_lengths,
                p,
            );
        } else {
            let chunk_first_key = base.key(chunk.start);
            let chunk_last_key = base.key(chunk.end - 1);
            let first_len = separator_len(&chunk_first_key);

            p = prefix_len(&chunk_first_key, &chunk_last_key);
            t = node::compressed_separator_range_size(
                first_len,
                self.n + chunk.len(),
                chunk.sum_separator_lengths - first_len,
                p,
            );
        };

        branch_node::body_size(p, t, self.n + chunk.len())
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

        gauge.ingest_key([0; 32], 0);

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

            gauge.ingest_key(item, len);
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
            gauge.ingest_key(next_key, s_len);
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
            gauge.ingest_key(next_key, s_len);
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                low: 0,
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
                gauge.ingest_key(key, separator_len(&key))
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
