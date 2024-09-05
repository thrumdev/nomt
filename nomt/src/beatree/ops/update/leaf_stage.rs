use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crossbeam_channel::TryRecvError;
use dashmap::DashMap;
use threadpool::ThreadPool;

use crate::beatree::{
    allocator::PageNumber,
    index::Index,
    leaf::{node::LeafNode, store::LeafStoreReader},
    ops::search_branch,
    Key,
};
use crate::io::PagePool;

use super::{
    leaf_updater::{BaseLeaf, DigestResult as LeafDigestResult, LeafUpdater},
    get_key, SeparatorRange,
};

struct ExtendRangeResponse {
    changed: Vec<(Key, ChangedLeafEntry)>,
    new_right_neighbor: Option<Option<RightNeighbor>>,
}

type LeftNeighbor = super::LeftNeighbor<ExtendRangeResponse>;
type RightNeighbor = super::RightNeighbor<ExtendRangeResponse>;
type ExtendRangeRequest = super::ExtendRangeRequest<ExtendRangeResponse>;

pub struct ChangedLeafEntry {
    pub deleted: Option<PageNumber>,
    pub inserted: Option<LeafNode>,

    // the separator of the next node.
    pub next_separator: Option<Key>,
}

#[derive(Default)]
pub struct LeafChanges {
    // TODO: there's no real reason for this to be a BTreemap rather than a Vec. We push to the
    // end always and drain from the left.
    inner: BTreeMap<Key, ChangedLeafEntry>,
    overflow_deleted: Vec<Vec<u8>>,
}

impl LeafChanges {
    pub fn delete(&mut self, key: Key, pn: PageNumber, next_separator: Option<Key>) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedLeafEntry {
            deleted: None,
            inserted: None,
            next_separator: None,
        });

        entry.next_separator = next_separator;

        // we can only delete a leaf once.
        assert!(entry.deleted.is_none());

        entry.deleted = Some(pn);
    }

    pub fn insert(&mut self, key: Key, node: LeafNode, next_separator: Option<Key>) {
        let entry = self.inner.entry(key).or_insert_with(|| ChangedLeafEntry {
            deleted: None,
            inserted: None,
            next_separator: None,
        });

        entry.next_separator = next_separator;

        if let Some(_prev) = entry.inserted.replace(node) {
            // TODO: this is where we'd clean up.
        }
    }

    fn delete_overflow(&mut self, overflow_cell: &[u8]) {
        self.overflow_deleted.push(overflow_cell.to_vec());
    }
}

fn indexed_leaf(
    bbn_index: &Index,
    key: Key,
) -> Option<(Key, Option<Key>, PageNumber)> {
    let Some((_, branch)) = bbn_index.lookup(key) else {
        return None;
    };

    let Some((i, leaf_pn)) = search_branch(&branch, key) else {
        return None;
    };
    let separator = get_key(&branch, i);

    let cutoff = if i + 1 < branch.n() as usize {
        Some(get_key(&branch, i + 1))
    } else {
        bbn_index.next_after(key).map(|(cutoff, _)| cutoff)
    };

    Some((separator, cutoff, leaf_pn))
}

pub fn run(
    bbn_index: &Index,
    leaf_cache: DashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    page_pool: PagePool,
    changeset: Vec<(Key, Option<(Vec<u8>, bool)>)>,
    thread_pool: ThreadPool,
    num_workers: usize,
) -> (Vec<(Key, ChangedLeafEntry)>, Vec<Vec<u8>>) {
    assert!(num_workers >= 1);
    let workers = prepare_workers(bbn_index, &changeset, num_workers);
    assert!(!workers.is_empty());

    let leaf_cache = Arc::new(leaf_cache);
    let changeset = Arc::new(changeset);

    let num_workers = workers.len();
    let (worker_result_tx, worker_result_rx) = crossbeam_channel::bounded(num_workers);

    for worker_params in workers {
        let bbn_index = bbn_index.clone();
        let leaf_cache = leaf_cache.clone();
        let leaf_reader = leaf_reader.clone();
        let page_pool = page_pool.clone();
        let changeset = changeset.clone();

        let worker_result_tx = worker_result_tx.clone();
        thread_pool.execute(move || {
            // passing the large `Arc` values by reference ensures that they are dropped at the
            // end of this scope, not the end of `run_worker`.
            let res = run_worker(
                bbn_index,
                &*leaf_cache,
                leaf_reader,
                page_pool,
                &*changeset,
                worker_params,
            );

            let _ = worker_result_tx.send(res);
        });
    }

    // we don't want to block other sync steps on deallocating these memory regions.
    drop(changeset);
    drop(leaf_cache);

    let mut changes = Vec::new();
    let mut deleted_overflow = Vec::new();

    for _ in 0..num_workers {
        // UNWRAP: results are always sent unless worker panics.
        let (worker_changes, worker_deleted_overflow) = worker_result_rx.recv().unwrap();
        changes.extend(worker_changes);
        deleted_overflow.extend(worker_deleted_overflow);
    }

    (changes, deleted_overflow)
}

struct WorkerParams {
    left_neighbor: Option<LeftNeighbor>,
    right_neighbor: Option<RightNeighbor>,
    range: SeparatorRange,
    op_range: Range<usize>,
}

fn prepare_workers(
    bbn_index: &Index,
    changeset: &[(Key, Option<(Vec<u8>, bool)>)],
    worker_count: usize,
) -> Vec<WorkerParams> {
    let mut remaining_workers = worker_count;
    let mut changeset_remaining = changeset;

    let mut workers = Vec::with_capacity(worker_count);

    // first worker covers everything. it'll be adjusted in the next iteration of the loop,
    // if possible.
    workers.push(WorkerParams {
        left_neighbor: None,
        right_neighbor: None,
        range: SeparatorRange {
            low: None,
            high: None,
        },
        op_range: Range {
            start: 0,
            end: changeset.len(),
        },
    });
    remaining_workers -= 1;

    // iter: find endpoint for previous worker and push next worker.
    // look up the leaf for the (ops/N)th operation, find out how many ops it covers,
    // and set the end range accordingly.
    while remaining_workers > 0 && changeset_remaining.len() > 0 {
        let pivot_idx = changeset_remaining.len() / (remaining_workers + 1);

        // UNWRAP: first worker is pushed at the beginning of the range.
        let prev_worker = workers.last_mut().unwrap();

        match indexed_leaf(bbn_index, changeset_remaining[pivot_idx].0) {
            None => break,
            Some((_, None, _)) => break,
            Some((separator, Some(_), _)) => {
                // link this worker with the previous one.
                let (tx, rx) = crossbeam_channel::unbounded();

                // have previous worker cover everything, only up to the separator of this node.
                let prev_worker_ops = pivot_idx
                    - changeset_remaining[..pivot_idx]
                        .iter()
                        .rev()
                        .take_while(|(k, _)| k >= &separator)
                        .count();

                if prev_worker_ops == 0 {
                    changeset_remaining = &changeset_remaining[pivot_idx..];
                    continue
                }

                let op_partition_index =
                    (changeset.len() - changeset_remaining.len()) + prev_worker_ops;

                // previous worker now owns all nodes up to this one.
                prev_worker.range.high = Some(separator);
                prev_worker.right_neighbor = Some(RightNeighbor { tx });
                prev_worker.op_range.end = op_partition_index;

                workers.push(WorkerParams {
                    left_neighbor: Some(LeftNeighbor { rx }),
                    right_neighbor: None,
                    range: SeparatorRange {
                        low: Some(separator),
                        high: None,
                    },
                    op_range: Range {
                        start: op_partition_index,
                        end: changeset.len(),
                    },
                });
                remaining_workers -= 1;
                changeset_remaining = &changeset_remaining[pivot_idx..];
            }
        }
    }

    workers
}

fn try_answer_left_neighbor(
    pending_request: &mut Option<ExtendRangeRequest>,
    worker_params: &mut WorkerParams,
    leaf_changes: &mut LeafChanges,
) {
    let Some(ref left_neighbor) = worker_params.left_neighbor else {
        return;
    };

    let request = match pending_request.take() {
        Some(r) => r,
        None => match left_neighbor.rx.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Disconnected) => {
                worker_params.left_neighbor = None;
                return;
            }
            _ => return,
        },
    };

    // We send the left side information it needs to deduce the first node following the previous
    // high point of its range. note that `self.low` and `left.high` are kept equal.

    let mut found_next_node = false;
    let mut next_separator = worker_params.range.low.clone();

    // We take the first rewritten node in the range.
    let take = leaf_changes
        .inner
        .iter()
        .take_while(|(_, entry)| {
            if found_next_node {
                return false;
            }

            found_next_node = entry.inserted.is_some();
            next_separator = entry.next_separator.clone();
            true
        })
        .count();

    let consumed_whole_range = next_separator == worker_params.range.high;

    if !found_next_node && !consumed_whole_range {
        *pending_request = Some(request);
        return;
    }

    // update our low range to match.
    worker_params.range.low = next_separator;

    let changed = (0..take)
        .map(|_| leaf_changes.inner.pop_first().unwrap())
        .collect::<Vec<_>>();

    let new_right_neighbor = if consumed_whole_range {
        // left neighbor consumed our entire range. link them up with our right neighbor.
        worker_params.left_neighbor = None;
        Some(worker_params.right_neighbor.take())
    } else {
        None
    };

    // UNWRAP: neighbor waiting on request will never drop.
    request
        .tx
        .send(ExtendRangeResponse {
            changed,
            new_right_neighbor,
        })
        .unwrap();
}

fn request_range_extension(worker_params: &mut WorkerParams, leaf_changes: &mut LeafChanges) {
    // UNWRAP: we should only be requesting a range extension when we have a right neighbor.
    // workers with no right neighbor have no limit to their range.
    let right_neighbor = worker_params.right_neighbor.as_ref().unwrap();

    let (tx, rx) = crossbeam_channel::unbounded();
    let request = ExtendRangeRequest { tx };

    // UNWRAP: right neighbor never drops until left neighbor is done.
    right_neighbor.tx.send(request).unwrap();

    // UNWRAP: right neighbor never drops until left neighbor is done.
    let response = rx.recv().unwrap();

    // UNWRAP: answering an extend range request always returns at least one change.
    worker_params.range.high = response.changed.last().unwrap().1.next_separator;
    leaf_changes.inner.extend(response.changed);

    if let Some(new_right_neighbor) = response.new_right_neighbor {
        worker_params.right_neighbor = new_right_neighbor;
        if worker_params.right_neighbor.is_some() {
            // exhausting the range means we didn't get a node to merge with. try again with
            // the next right neighbor.
            request_range_extension(worker_params, leaf_changes);
        }
    }
}

fn reset_leaf_base(
    bbn_index: &Index,
    leaf_cache: &DashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    leaf_changes: &mut LeafChanges,
    leaf_updater: &mut LeafUpdater,
    has_extended_range: bool,
    key: Key,
) {
    let Some((separator, cutoff, leaf_pn)) = indexed_leaf(bbn_index, key) else {
        return;
    };

    // simple path for any time we haven't extended our range.
    if !has_extended_range {
        reset_leaf_base_fresh(
            separator,
            cutoff,
            leaf_pn,
            leaf_cache,
            leaf_reader,
            leaf_changes,
            leaf_updater,
        );
        return;
    }

    // UNWRAP: extending our range gave us at least one additional deleted/inserted leaf.
    let range = leaf_changes.inner.range_mut(key..);

    for (new_key, new_entry) in range {
        // Right worker's first mutated item starts beyond the current range.
        if new_key > &key {
            let cutoff = if cutoff.map_or(true, |c| &c > new_key) {
                Some(*new_key)
            } else {
                cutoff
            };

            reset_leaf_base_fresh(
                separator,
                cutoff,
                leaf_pn,
                leaf_cache,
                leaf_reader,
                leaf_changes,
                leaf_updater,
            );
            return;
        } else if new_entry.inserted.is_none() {
            continue;
        } else {
            // UNWRAP: right worker sent us a node that covered the next key we asked for.
            let base_node = new_entry.inserted.take().unwrap();
            let base = BaseLeaf {
                node: base_node,
                iter_pos: 0,
                separator: *new_key,
            };
            leaf_updater.reset_base(Some(base), new_entry.next_separator);
            break;
        }
    }

    // special case: all rightward workers deleted every last one of their nodes after the last one
    // we received from a range extension. We are now writing the new rightmost node, which
    // is permitted to be underfull.
    leaf_updater.remove_cutoff();
}

fn reset_leaf_base_fresh(
    separator: Key,
    cutoff: Option<Key>,
    leaf_pn: PageNumber,
    leaf_cache: &DashMap<PageNumber, LeafNode>,
    leaf_reader: &LeafStoreReader,
    leaf_changes: &mut LeafChanges,
    leaf_updater: &mut LeafUpdater,
) {
    // we intend to work on this leaf, therefore, we delete it. any new leaves produced by the
    // updater will replace it.
    leaf_changes.delete(separator, leaf_pn, cutoff);

    let base = BaseLeaf {
        node: leaf_cache
            .remove(&leaf_pn)
            .map(|(_, l)| l)
            .unwrap_or_else(|| LeafNode {
                inner: leaf_reader.query(leaf_pn),
            }),
        iter_pos: 0,
        separator,
    };

    leaf_updater.reset_base(Some(base), cutoff);
}

fn run_worker(
    bbn_index: Index,
    leaf_cache: &DashMap<PageNumber, LeafNode>,
    leaf_reader: LeafStoreReader,
    page_pool: PagePool,
    changeset: &[(Key, Option<(Vec<u8>, bool)>)],
    mut worker_params: WorkerParams,
) -> (BTreeMap<Key, ChangedLeafEntry>, Vec<Vec<u8>>) {
    if changeset.is_empty() {
        return (BTreeMap::new(), Vec::new());
    }
    let mut leaf_changes = LeafChanges::default();
    let mut leaf_updater = LeafUpdater::new(page_pool, None, None);
    let mut pending_left_request = None;
    let mut has_extended_range = false;

    // point leaf updater at first leaf.
    reset_leaf_base(
        &bbn_index,
        &leaf_cache,
        &leaf_reader,
        &mut leaf_changes,
        &mut leaf_updater,
        has_extended_range,
        changeset[worker_params.op_range.start].0,
    );

    for (key, op) in &changeset[worker_params.op_range.clone()] {
        // ensure key is in scope for leaf updater. if not, digest it. merge rightwards until
        //    done _or_ key is in scope.
        while !leaf_updater.is_in_scope(&key) {
            let k = if let LeafDigestResult::NeedsMerge(cutoff) =
                leaf_updater.digest(&mut leaf_changes)
            {
                cutoff
            } else {
                *key
            };

            try_answer_left_neighbor(
                &mut pending_left_request,
                &mut worker_params,
                &mut leaf_changes,
            );

            if worker_params.range.high.map_or(false, |high| k >= high) {
                has_extended_range = true;
                request_range_extension(&mut worker_params, &mut leaf_changes);
            }

            reset_leaf_base(
                &bbn_index,
                &leaf_cache,
                &leaf_reader,
                &mut leaf_changes,
                &mut leaf_updater,
                has_extended_range,
                k,
            );
        }

        let (value_change, overflow) = match op {
            None => (None, false),
            Some((v, overflow)) => (Some(v.clone()), *overflow),
        };

        let delete_overflow = |overflow_cell: &[u8]| leaf_changes.delete_overflow(overflow_cell);
        leaf_updater.ingest(*key, value_change, overflow, delete_overflow);
    }

    loop {
        let res = leaf_updater.digest(&mut leaf_changes);

        try_answer_left_neighbor(
            &mut pending_left_request,
            &mut worker_params,
            &mut leaf_changes,
        );

        if let LeafDigestResult::NeedsMerge(cutoff) = res {
            if worker_params
                .range
                .high
                .map_or(false, |high| cutoff >= high)
            {
                has_extended_range = true;
                request_range_extension(&mut worker_params, &mut leaf_changes);
            }

            reset_leaf_base(
                &bbn_index,
                &leaf_cache,
                &leaf_reader,
                &mut leaf_changes,
                &mut leaf_updater,
                has_extended_range,
                cutoff,
            );

            continue;
        }
        break;
    }

    // answer any outstanding pending request
    try_answer_left_neighbor(
        &mut pending_left_request,
        &mut worker_params,
        &mut leaf_changes,
    );
    assert!(pending_left_request.is_none());

    // wait until left worker concludes or exhausts our range.
    while worker_params.left_neighbor.is_some() {
        match worker_params.left_neighbor.as_ref().map(|l| l.rx.recv()) {
            None => continue,
            Some(Ok(item)) => {
                try_answer_left_neighbor(&mut Some(item), &mut worker_params, &mut leaf_changes);
            }
            Some(Err(_)) => {
                worker_params.left_neighbor = None;
            }
        }
    }

    (leaf_changes.inner, leaf_changes.overflow_deleted)
}
