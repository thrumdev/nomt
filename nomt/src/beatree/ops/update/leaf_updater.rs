use std::cmp::Ordering;

use crate::beatree::{
    leaf::node::{self as leaf_node, LeafBuilder, LeafNode, LEAF_NODE_BODY_SIZE},
    ops::bit_ops::separate,
    Key,
};
use crate::io::PagePool;

use super::{
    leaf_stage::LeavesTracker, LEAF_BULK_SPLIT_TARGET, LEAF_BULK_SPLIT_THRESHOLD,
    LEAF_MERGE_THRESHOLD,
};

pub struct BaseLeaf {
    pub node: LeafNode,
    pub separator: Key,
    low: usize,
}

impl BaseLeaf {
    pub fn new(node: LeafNode, separator: Key) -> Self {
        BaseLeaf {
            node,
            separator,
            low: 0,
        }
    }

    // Try to find the given key starting from `self.low` up to the end.
    // Returns whether the key is present or not and the index of the key
    // or the index containing the first key bigger then the one specified.
    fn find_key(&mut self, key: &Key) -> Option<(bool, usize)> {
        let mut high = self.node.n();

        if self.low == high {
            return None;
        }

        while self.low < high {
            let mid = self.low + (high - self.low) / 2;

            match key.cmp(&self.key(mid)) {
                // If the key at `mid` is smaller than the one we are looking for,
                // then we are sure to go to the right
                Ordering::Greater => self.low = mid + 1,
                // If the key is the same, then we return its position in the base leaf,
                // updating `self.low` to be the item just after `mid`
                Ordering::Equal => {
                    self.low = mid + 1;
                    return Some((true, mid));
                }
                Ordering::Less if mid == 0 => return Some((false, 0)),
                // If the key at `mid` is bigger, we need to check if
                // the previous one is smaller or equal
                Ordering::Less => match key.cmp(&self.key(mid - 1)) {
                    Ordering::Less => high = mid,
                    Ordering::Equal => {
                        self.low = mid;
                        return Some((true, mid - 1));
                    }
                    Ordering::Greater => {
                        self.low = mid;
                        return Some((false, mid));
                    }
                },
            }
        }

        self.low = self.node.n();
        Some((false, self.node.n()))
    }

    fn key(&self, i: usize) -> Key {
        self.node.key(i)
    }

    fn key_cell(&self, i: usize) -> (Key, &[u8], bool) {
        let (value, overflow) = self.node.value(i);
        (self.node.key(i), value, overflow)
    }

    fn cell(&self, i: usize) -> (&[u8], bool) {
        self.node.value(i)
    }
}

#[derive(Debug, PartialEq)]
enum LeafOp {
    // Key, Value, Overflow
    Insert(Key, Vec<u8>, bool),
    // From, To, Values size
    KeepChunk(usize, usize, usize),
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
    page_pool: PagePool,
}

impl LeafUpdater {
    pub fn new(page_pool: PagePool, base: Option<BaseLeaf>, cutoff: Option<Key>) -> Self {
        LeafUpdater {
            base,
            cutoff,
            separator_override: None,
            ops: Vec::new(),
            gauge: LeafGauge::default(),
            bulk_split: None,
            page_pool,
        }
    }

    pub fn is_in_scope(&self, key: &Key) -> bool {
        self.cutoff.map_or(true, |k| *key < k)
    }

    pub fn reset_base(&mut self, base: Option<BaseLeaf>, cutoff: Option<Key>) {
        self.base = base;
        self.cutoff = cutoff;
    }

    pub fn remove_cutoff(&mut self) {
        self.cutoff = None;
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
    pub fn digest(&mut self, leaves_tracker: &mut LeavesTracker) -> DigestResult {
        // no cells are going to be deleted from this point onwards - this keeps everything.
        self.keep_up_to(None, |_| {});

        // note: if we need a merge, it'd be more efficient to attempt to combine it with the last
        // leaf of the bulk split first rather than pushing the ops onwards. probably irrelevant
        // in practice; bulk splits are rare.
        let last_ops_start = self.build_bulk_splitter_leaves(leaves_tracker);

        if self.gauge.body_size() == 0 {
            self.ops.clear();
            self.separator_override = None;

            DigestResult::Finished
        } else if self.gauge.body_size() > LEAF_NODE_BODY_SIZE {
            assert_eq!(
                last_ops_start, 0,
                "normal split can only occur when not bulk splitting"
            );
            self.split(leaves_tracker)
        } else if self.gauge.body_size() >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let node = self.build_leaf(&self.ops[last_ops_start..]);
            let separator = self.separator();

            leaves_tracker.insert(separator, node, self.cutoff);

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
        let Some(base) = self.base.as_mut() else {
            // empty db
            return;
        };

        let from = base.low;
        let (found, to) = match up_to {
            // Nothing more to do, the end has already been reached
            None if base.low == base.node.n() => return,
            // Jump direcly to the end of the base node
            None => (false, base.node.n()),
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

        let values_size = base.node.values_size(from, to);
        self.ops.push(LeafOp::KeepChunk(from, to, values_size));

        if found {
            let (val, overflow) = base.cell(to);
            if overflow {
                with_deleted_overflow(val);
            }
        }

        self.bulk_split_step(self.ops.len() - 1);
    }

    // check whether bulk split needs to start, and if so, start it.
    // if ongoing, check if we need to cut off.
    // returns the amount of operations consumed for the bulk creation
    fn bulk_split_step(&mut self, op_index: usize) -> usize {
        let (mut n_items, mut values_size) = match self.ops[op_index] {
            LeafOp::Insert(_, ref val, _) => (1, val.len()),
            LeafOp::KeepChunk(from, to, values_size) => (to - from, values_size),
        };

        let body_size_after = self.gauge.body_size_after(n_items, values_size);
        match self.bulk_split {
            None if body_size_after >= LEAF_BULK_SPLIT_THRESHOLD => {
                self.bulk_split = Some(LeafBulkSplitter::default());
                self.gauge = LeafGauge::default();
                let mut idx = 0;
                while idx < self.ops.len() {
                    let consumed = self.bulk_split_step(idx);
                    idx += consumed;
                }
                idx
            }
            Some(ref mut bulk_splitter) if body_size_after >= LEAF_BULK_SPLIT_TARGET => {
                let mut recursive_on_split = false;
                let accept_item = body_size_after <= LEAF_NODE_BODY_SIZE || {
                    // If we are here, it means that self.ops[op_index] goes from underefull to overfull.
                    // If it is a KeepChunk, there is a chance to split it into two parts,
                    // accept the previous one, and iterate recursively over the newly created one.

                    match self.ops[op_index] {
                        LeafOp::Insert(..) => {
                            if self.gauge.body_size() < LEAF_MERGE_THRESHOLD {
                                // super degenerate split! node grew from underfull to overfull in one
                                // item. only thing to do here is merge leftwards, unfortunately.
                                // save this for later to do another pass with.
                                todo!()
                            }
                            false
                        }
                        LeafOp::KeepChunk(..) => {
                            // UNWRAP: if the operation is a KeepChunk variant, then base must exist
                            let (left_n_items, left_values_size) = split_keep_chunk(
                                self.base.as_ref().unwrap(),
                                &self.gauge,
                                &mut self.ops,
                                op_index,
                                LEAF_NODE_BODY_SIZE,
                                LEAF_NODE_BODY_SIZE,
                            );

                            if left_n_items > 0 {
                                // KeepChunk has been successfully split, the left chunk is now able to fit
                                n_items = left_n_items;
                                values_size = left_values_size;
                                recursive_on_split = true;
                                true
                            } else {
                                // KeepChunk could not have been split,
                                // thus we end up in the same scenario as with `LeafOp::Insert`
                                if self.gauge.body_size() < LEAF_MERGE_THRESHOLD {
                                    todo!()
                                }
                                false
                            }
                        }
                    }
                };

                let n = if accept_item {
                    self.gauge.ingest(n_items, values_size);
                    op_index + 1 - bulk_splitter.total_count
                } else {
                    op_index - bulk_splitter.total_count
                };

                // push onto bulk splitter & restart gauge.
                self.gauge = LeafGauge::default();
                bulk_splitter.push(n);

                if !accept_item {
                    self.gauge.ingest(n_items, values_size);
                }

                if recursive_on_split {
                    1 + self.bulk_split_step(op_index + 1)
                } else {
                    1
                }
            }
            _ => {
                self.gauge.ingest(n_items, values_size);
                1
            }
        }
    }

    fn build_bulk_splitter_leaves(&mut self, leaves_tracker: &mut LeavesTracker) -> usize {
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
                let next = self.op_first_key(op);
                let last = new_node.key(new_node.n() - 1);
                self.separator_override = Some(separate(&last, &next));
            }

            leaves_tracker.insert(separator, new_node, self.separator_override.or(self.cutoff));
            start += item_count;
        }

        start
    }

    /// The separator of the next leaf that will be built.
    pub fn separator(&self) -> Key {
        // the first leaf always gets a separator of all 0.
        self.separator_override
            .or(self.base.as_ref().map(|b| b.separator))
            .unwrap_or([0u8; 32])
    }

    fn split(&mut self, leaves_tracker: &mut LeavesTracker) -> DigestResult {
        let midpoint = self.gauge.body_size() / 2;
        let mut split_point = 0;

        let mut left_gauge = LeafGauge::default();
        while left_gauge.body_size() < midpoint {
            let (n_items, values_size) = match self.ops[split_point] {
                LeafOp::Insert(_, ref val, _) => (1, val.len()),
                LeafOp::KeepChunk(from, to, values_size)
                    if left_gauge.body_size_after(to - from, values_size) <= midpoint =>
                {
                    (to - from, values_size)
                }
                LeafOp::KeepChunk(..) => {
                    // UNWRAP: if the operation is a KeepChunk variant, then base must exist
                    let (n_items, values_size) = split_keep_chunk(
                        self.base.as_ref().unwrap(),
                        &left_gauge,
                        &mut self.ops,
                        split_point,
                        midpoint,
                        LEAF_NODE_BODY_SIZE,
                    );
                    (n_items, values_size)
                }
            };

            // stop using ops for the left leaf if no more item is able to fit
            if n_items == 0
                || left_gauge.body_size_after(n_items, values_size) > LEAF_NODE_BODY_SIZE
            {
                if left_gauge.body_size() < LEAF_MERGE_THRESHOLD {
                    // super degenerate split! jumped from underfull to overfull in a single step.
                    todo!()
                }

                break;
            }

            left_gauge.ingest(n_items, values_size);
            split_point += 1;
        }

        let left_ops = &self.ops[..split_point];
        let right_ops = &self.ops[split_point..];

        let left_key = self.op_last_key(&self.ops[split_point - 1]);
        let right_key = self.op_first_key(&self.ops[split_point]);

        let left_separator = self.separator();
        let right_separator = separate(&left_key, &right_key);

        let left_leaf = self.build_leaf(left_ops);
        leaves_tracker.insert(left_separator, left_leaf, Some(right_separator));

        let mut right_gauge = LeafGauge::default();
        for op in &self.ops[split_point..] {
            let (n_items, values_size) = match op {
                LeafOp::Insert(_, ref val, _) => (1, val.len()),
                LeafOp::KeepChunk(from, to, values_size) => (to - from, *values_size),
            };

            right_gauge.ingest(n_items, values_size);
        }

        if right_gauge.body_size() > LEAF_NODE_BODY_SIZE {
            // This is a rare case left uncovered by the bulk split, the threshold to activate it
            // has not been reached by the sum of all left and right operations. Now the right
            // leaf is too big, and another split is required to be executed
            self.ops.drain(..split_point);
            self.separator_override = Some(right_separator);
            self.gauge = right_gauge;
            self.split(leaves_tracker)
        } else if right_gauge.body_size() >= LEAF_MERGE_THRESHOLD || self.cutoff.is_none() {
            let right_leaf = self.build_leaf(right_ops);
            leaves_tracker.insert(right_separator, right_leaf, self.cutoff);

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
        let mut new_insert = 0;
        for i in 0..self.ops.len() {
            match self.ops[new_insert + i] {
                LeafOp::Insert(..) => (),
                LeafOp::KeepChunk(from, to, ..) => {
                    self.ops.remove(new_insert + i);

                    for pos in (from..to).into_iter().rev() {
                        let (k, v, o) = base.key_cell(pos);
                        self.ops
                            .insert(new_insert + i, LeafOp::Insert(k, v.to_vec(), o));
                    }
                    new_insert += to - from - 1;
                }
            }
        }
    }

    fn op_first_key(&self, leaf_op: &LeafOp) -> Key {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, _, _) => *k,
            LeafOp::KeepChunk(from, _, _) => self.base.as_ref().unwrap().key(*from),
        }
    }

    fn op_last_key(&self, leaf_op: &LeafOp) -> Key {
        // UNWRAP: `Keep` leaf ops only exist when base is `Some`.
        match leaf_op {
            LeafOp::Insert(k, _, _) => *k,
            LeafOp::KeepChunk(_, to, _) => self.base.as_ref().unwrap().key(to - 1),
        }
    }

    fn build_leaf(&self, ops: &[LeafOp]) -> LeafNode {
        let (n_values, total_value_size) = ops
            .iter()
            .map(|op| match op {
                LeafOp::Insert(_, v, _) => (1, v.len()),
                LeafOp::KeepChunk(from, to, values_size) => (to - from, *values_size),
            })
            .fold((0, 0), |(acc_n, acc_size), (n, size)| {
                (acc_n + n, acc_size + size)
            });

        let mut leaf_builder = LeafBuilder::new(&self.page_pool, n_values, total_value_size);

        for op in ops {
            match op {
                LeafOp::Insert(k, v, o) => {
                    leaf_builder.push_cell(*k, v, *o);
                }
                LeafOp::KeepChunk(from, to, _) => {
                    // UNWRAP: if the operation is a KeepChunk variant, then base must exist
                    leaf_builder.push_chunk(&self.base.as_ref().unwrap().node, *from, *to)
                }
            }
        }
        leaf_builder.finish()
    }
}

// Given a vector of `LeafOp`, try to split the `index` operation,
// which is expected to be KeepChunk, into two halves,
// targeting a `target` size and and not exceeding a `limit`.
//
// `target` and `limit` are required to understand when to accept a split
// with a final size smaller than the target. Constraining the split to always
// be bigger than the target causes the update algorithm to frequently
// fall into underfull to overfull scenarios.
fn split_keep_chunk(
    base: &BaseLeaf,
    gauge: &LeafGauge,
    ops: &mut Vec<LeafOp>,
    index: usize,
    target: usize,
    limit: usize,
) -> (usize, usize) {
    let LeafOp::KeepChunk(from, to, values_size) = ops[index] else {
        panic!("Attempted to split non `LeafOp::KeepChunk` operation");
    };

    let mut left_chunk_n_items = 0;
    let mut left_chunk_values_size = 0;
    for pos in from..to {
        let size = base.cell(pos).0.len();

        left_chunk_values_size += size;
        left_chunk_n_items += 1;

        let body_size_after = gauge.body_size_after(left_chunk_n_items, left_chunk_values_size);
        if body_size_after >= target {
            // if an item jumps from below the target to bigger then the limit, do not use it
            if body_size_after > limit {
                left_chunk_values_size -= size;
                left_chunk_n_items -= 1;
            }
            break;
        }
    }

    // there must be at least one element taken from the chunk,
    // and if all elements are taken then nothing needs to be changed
    if left_chunk_n_items != 0 && to - from != left_chunk_n_items {
        ops.insert(
            index,
            LeafOp::KeepChunk(from, from + left_chunk_n_items, left_chunk_values_size),
        );

        ops[index + 1] = LeafOp::KeepChunk(
            from + left_chunk_n_items,
            to,
            values_size - left_chunk_values_size,
        );
    }

    (left_chunk_n_items, left_chunk_values_size)
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
    fn ingest(&mut self, n: usize, values_size: usize) {
        self.n += n;
        self.value_size_sum += values_size;
    }

    fn body_size_after(&self, n: usize, values_size: usize) -> usize {
        leaf_node::body_size(self.n + n, self.value_size_sum + values_size)
    }

    fn body_size(&self) -> usize {
        leaf_node::body_size(self.n, self.value_size_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        separate, BaseLeaf, DigestResult, Key, LeafBuilder, LeafNode, LeafOp, LeafUpdater,
        LeavesTracker, PagePool,
    };

    lazy_static::lazy_static! {
        static ref PAGE_POOL: PagePool = PagePool::new();
    }

    fn key(x: u8) -> Key {
        [x; 32]
    }

    fn make_leaf(vs: Vec<(Key, Vec<u8>, bool)>) -> LeafNode {
        let n = vs.len();
        let total_value_size = vs.iter().map(|(_, v, _)| v.len()).sum();

        let mut builder = LeafBuilder::new(&PAGE_POOL, n, total_value_size);
        for (k, v, overflow) in vs {
            builder.push_cell(k, &v, overflow);
        }

        builder.finish()
    }

    #[test]
    fn leaf_binary_search() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 500], false),
            (key(3), vec![1u8; 500], false),
            (key(5), vec![1u8; 500], false),
            (key(7), vec![1u8; 500], false),
            (key(9), vec![1u8; 500], false),
        ]);

        let mut base = BaseLeaf {
            node: leaf,
            low: 0,
            separator: key(1),
        };

        assert_eq!(base.find_key(&key(0)), Some((false, 0)));
        assert_eq!(base.find_key(&key(1)), Some((true, 0)));
        assert_eq!(base.find_key(&key(2)), Some((false, 1)));
        assert_eq!(base.find_key(&key(3)), Some((true, 1)));
        assert_eq!(base.find_key(&key(4)), Some((false, 2)));
        assert_eq!(base.find_key(&key(5)), Some((true, 2)));
        assert_eq!(base.find_key(&key(6)), Some((false, 3)));
        assert_eq!(base.find_key(&key(7)), Some((true, 3)));
        assert_eq!(base.find_key(&key(8)), Some((false, 4)));
        assert_eq!(base.find_key(&key(9)), Some((true, 4)));
        assert_eq!(base.find_key(&key(10)), None);
    }

    #[test]
    fn is_in_scope() {
        let mut updater = LeafUpdater::new(PAGE_POOL.clone(), None, None);
        assert!(updater.is_in_scope(&key(0xff)));

        updater.reset_base(None, Some(key(0xfe)));
        assert!(updater.is_in_scope(&key(0xf0)));
        assert!(updater.is_in_scope(&key(0xfd)));
        assert!(!updater.is_in_scope(&key(0xfe)));
        assert!(!updater.is_in_scope(&key(0xff)));
    }

    #[test]
    fn update() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1000], false),
            (key(2), vec![1u8; 1000], false),
            (key(3), vec![1u8; 1000], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(2), Some(vec![2u8; 1000]), false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();

        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 3);
        assert_eq!(new_leaf.get(&key(1)).unwrap().0, &[1u8; 1000]);
        assert_eq!(new_leaf.get(&key(2)).unwrap().0, &[2u8; 1000]);
        assert_eq!(new_leaf.get(&key(3)).unwrap().0, &[1u8; 1000]);
    }

    #[test]
    fn insert_rightsized() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 900], false),
            (key(2), vec![1u8; 900], false),
            (key(3), vec![1u8; 900], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(4), Some(vec![1u8; 900]), false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();

        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 4);
        assert_eq!(new_leaf.get(&key(1)).unwrap().0, &[1u8; 900]);
        assert_eq!(new_leaf.get(&key(2)).unwrap().0, &[1u8; 900]);
        assert_eq!(new_leaf.get(&key(3)).unwrap().0, &[1u8; 900]);
        assert_eq!(new_leaf.get(&key(4)).unwrap().0, &[1u8; 900]);
    }

    #[test]
    fn insert_overflowing() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1200], false),
            (key(2), vec![1u8; 1200], false),
            (key(3), vec![1u8; 1200], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(4), Some(vec![1u8; 1200]), false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        let new_leaf_entry_1 = leaves_tracker.get(key(1)).unwrap();
        let new_leaf_entry_2 = leaves_tracker.get(separate(&key(2), &key(3))).unwrap();

        let new_leaf_1 = new_leaf_entry_1.inserted.as_ref().unwrap();
        let new_leaf_2 = new_leaf_entry_2.inserted.as_ref().unwrap();

        assert_eq!(new_leaf_1.n(), 2);
        assert_eq!(new_leaf_2.n(), 2);

        assert_eq!(new_leaf_1.get(&key(1)).unwrap().0, &[1u8; 1200]);
        assert_eq!(new_leaf_1.get(&key(2)).unwrap().0, &[1u8; 1200]);
        assert_eq!(new_leaf_2.get(&key(3)).unwrap().0, &[1u8; 1200]);
        assert_eq!(new_leaf_2.get(&key(4)).unwrap().0, &[1u8; 1200]);
    }

    #[test]
    fn delete() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1200], false),
            (key(2), vec![1u8; 1200], false),
            (key(3), vec![1u8; 1200], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(2), None, false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();

        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 2);
        assert_eq!(new_leaf.get(&key(1)).unwrap().0, &[1u8; 1200]);
        assert_eq!(new_leaf.get(&key(3)).unwrap().0, &[1u8; 1200]);
    }

    #[test]
    fn delete_underflow_and_merge() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 800], false),
            (key(2), vec![1u8; 800], false),
            (key(3), vec![1u8; 800], false),
        ]);

        let leaf2 = make_leaf(vec![
            (key(4), vec![1u8; 1100], false),
            (key(5), vec![1u8; 1100], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            Some(key(4)),
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(2), None, false, |_| {});
        let DigestResult::NeedsMerge(merge_key) = updater.digest(&mut leaves_tracker) else {
            panic!()
        };
        assert_eq!(merge_key, key(4));

        assert!(leaves_tracker.get(key(1)).is_none());

        updater.reset_base(
            Some(BaseLeaf {
                node: leaf2,
                low: 0,
                separator: key(4),
            }),
            None,
        );

        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };
        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();

        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 4);
        assert_eq!(new_leaf.get(&key(1)).unwrap().0, &[1u8; 800]);
        assert_eq!(new_leaf.get(&key(3)).unwrap().0, &[1u8; 800]);
        assert_eq!(new_leaf.get(&key(4)).unwrap().0, &[1u8; 1100]);
        assert_eq!(new_leaf.get(&key(5)).unwrap().0, &[1u8; 1100]);
    }

    #[test]
    fn delete_calls_with_deleted_overflow() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1200], false),
            (key(2), vec![1u8; 1200], true),
            (key(3), vec![1u8; 1200], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        let mut called = false;
        updater.ingest(key(2), None, false, |_| called = true);
        assert!(called);
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };
    }

    #[test]
    fn delete_completely() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1200], false),
            (key(2), vec![1u8; 1200], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(1), None, false, |_| {});
        updater.ingest(key(2), None, false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        assert!(leaves_tracker.get(key(1)).is_none());
    }

    #[test]
    fn delete_underflow_rightmost() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1200], false),
            (key(2), vec![1u8; 1200], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            None,
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(1), None, false, |_| {});
        let DigestResult::Finished = updater.digest(&mut leaves_tracker) else {
            panic!()
        };

        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();
        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 1);
        assert_eq!(new_leaf.get(&key(2)).unwrap().0, &[1u8; 1200]);
    }

    #[test]
    fn split_with_underflow() {
        let leaf = make_leaf(vec![
            (key(1), vec![1u8; 1800], false),
            (key(2), vec![1u8; 1800], false),
            (key(3), vec![1u8; 300], false),
        ]);

        let mut updater = LeafUpdater::new(
            PAGE_POOL.clone(),
            Some(BaseLeaf {
                node: leaf,
                low: 0,
                separator: key(1),
            }),
            Some(key(5)),
        );
        let mut leaves_tracker = LeavesTracker::new();

        updater.ingest(key(4), Some(vec![1; 300]), false, |_| {});
        let DigestResult::NeedsMerge(merge_key) = updater.digest(&mut leaves_tracker) else {
            panic!()
        };
        assert_eq!(merge_key, key(5));

        let new_leaf_entry = leaves_tracker.get(key(1)).unwrap();
        let new_leaf = new_leaf_entry.inserted.as_ref().unwrap();
        assert_eq!(new_leaf.n(), 2);
        assert_eq!(new_leaf.get(&key(1)).unwrap().0, &[1u8; 1800]);
        assert_eq!(new_leaf.get(&key(2)).unwrap().0, &[1u8; 1800]);

        assert_eq!(updater.separator_override, Some(separate(&key(2), &key(3))));
        assert_eq!(
            updater.ops,
            vec![
                LeafOp::Insert(key(3), vec![1u8; 300], false),
                LeafOp::Insert(key(4), vec![1u8; 300], false),
            ]
        );
    }
}
