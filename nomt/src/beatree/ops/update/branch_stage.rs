use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use threadpool::ThreadPool;

use crate::beatree::{allocator::PageNumber, branch::node::BranchNode, index::Index, Key};

use crate::io::page_pool::{Page, PagePool, UnsafePageView, UnsafePageViewMut};

use super::branch_updater::{BaseBranch, BranchUpdater, DigestResult as BranchDigestResult};
use super::extend_range_protocol::{
    request_range_extension, try_answer_left_neighbor, LeftNeighbor, RightNeighbor, SeparatorRange,
    WorkerParams,
};

/// Tracker of all changes that happen to branch nodes during an update
pub type BranchesTracker = super::NodesTracker<BranchNode<UnsafePageViewMut>>;
type ChangedBranchEntry = super::ChangedNodeEntry<BranchNode<UnsafePageViewMut>>;

/// Change the btree's branch nodes in the specified way
pub fn run(
    bbn_index: &Index,
    page_pool: PagePool,
    changeset: Vec<(Key, Option<PageNumber>)>,
    thread_pool: ThreadPool,
    num_workers: usize,
) -> (BTreeMap<Key, ChangedBranchEntry>, Vec<Vec<Page>>) {
    if changeset.is_empty() {
        return (BTreeMap::new(), Vec::new());
    }

    assert!(num_workers >= 1);
    let workers = prepare_workers(bbn_index, &changeset, num_workers);
    assert!(!workers.is_empty());

    let changeset = Arc::new(changeset);

    let num_workers = workers.len();
    let (worker_result_tx, worker_result_rx) = crossbeam_channel::bounded(num_workers);

    for worker_params in workers {
        let bbn_index = bbn_index.clone();
        let page_pool = page_pool.clone();
        let changeset = changeset.clone();

        let worker_result_tx = worker_result_tx.clone();
        thread_pool.execute(move || {
            // passing the large `Arc` values by reference ensures that they are dropped at the
            // end of this scope, not the end of `run_worker`.
            let res = run_worker(bbn_index, page_pool, &*changeset, worker_params);
            let _ = worker_result_tx.send(res);
        });
    }

    // we don't want to block other sync steps on deallocating these memory regions.
    drop(changeset);

    let mut changes = BTreeMap::new();
    let mut bbn_outdated_pages = Vec::new();

    for _ in 0..num_workers {
        // UNWRAP: results are always sent unless worker panics.
        let (worker_changes, worker_outdated) = worker_result_rx.recv().unwrap();
        changes.extend(worker_changes);
        bbn_outdated_pages.push(worker_outdated);
    }

    (changes, bbn_outdated_pages)
}

fn prepare_workers(
    bbn_index: &Index,
    changeset: &[(Key, Option<PageNumber>)],
    worker_count: usize,
) -> Vec<WorkerParams<BranchNode<UnsafePageViewMut>>> {
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
    // look up the branch for the (ops/N)th operation, find out how many ops it covers,
    // and set the end range accordingly.
    while remaining_workers > 0 && changeset_remaining.len() > 0 {
        let pivot_idx = changeset_remaining.len() / (remaining_workers + 1);

        // If pivot_idx == 0 the number of changeset_remaining
        // is less than the number of remaining workers.
        // Everything remaining will be covered by the previous worker.
        if pivot_idx == 0 {
            break;
        }

        // UNWRAP: first worker is pushed at the beginning of the range.
        let prev_worker = workers.last_mut().unwrap();

        let Some((_, branch)) = bbn_index.lookup(changeset_remaining[pivot_idx].0) else {
            // This could happen only on empty db, let one worker handle everything
            break;
        };

        // SAFETY: page pool is alive, pages in index are live and frozen.
        let view = unsafe { UnsafePageView::new(branch) };
        let branch = BranchNode::new(view);

        let separator = super::get_key(&branch, 0);

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
            continue;
        }

        let op_partition_index = (changeset.len() - changeset_remaining.len()) + prev_worker_ops;

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
        changeset_remaining = &changeset_remaining[prev_worker_ops..];
    }

    workers
}

fn reset_branch_base(
    bbn_index: &Index,
    branches_tracker: &mut BranchesTracker,
    bbn_outdated_pages: &mut Vec<Page>,
    branch_updater: &mut BranchUpdater,
    has_extended_range: bool,
    mut key: Key,
) {
    if !has_extended_range {
        reset_branch_base_fresh(
            bbn_index,
            branches_tracker,
            bbn_outdated_pages,
            branch_updater,
            key,
        );
        return;
    }

    if let Some((_, node, next_separator)) = branches_tracker.pending_base.take() {
        // We are reusing a branch created by another worker. we need to schedule clean-up
        // of that branch to avoid a memory leak.
        let page = node.into_inner().into_shared();
        bbn_outdated_pages.push(page.clone().into_inner());

        let base = BaseBranch {
            node: BranchNode::new(page),
            iter_pos: 0,
        };
        branch_updater.reset_base(Some(base), next_separator);
    } else {
        if let Some(separator) = branches_tracker
            .inner
            .last_key_value()
            .and_then(|(_, entry)| entry.next_separator)
        {
            if separator > key {
                key = separator;
            }
            reset_branch_base_fresh(
                bbn_index,
                branches_tracker,
                bbn_outdated_pages,
                branch_updater,
                key,
            );
        } else {
            // special case: all rightward workers deleted every last one of their nodes after the last one
            // we received from a range extension. We are now writing the new rightmost node, which
            // is permitted to be underfull.
            branch_updater.remove_cutoff();
        }
    }
}

fn reset_branch_base_fresh(
    bbn_index: &Index,
    branches_tracker: &mut BranchesTracker,
    bbn_outdated_pages: &mut Vec<Page>,
    branch_updater: &mut BranchUpdater,
    key: Key,
) {
    let Some((separator, branch_page)) = bbn_index.lookup(key) else {
        return;
    };

    // SAFETY: page pool is alive, pages in index are live and frozen.
    let view = unsafe { UnsafePageView::new(branch_page.clone()) };
    let branch = BranchNode::new(view);
    let cutoff = bbn_index.next_after(key).map(|(k, _)| k);

    branches_tracker.delete(separator, branch.bbn_pn().into(), cutoff);
    bbn_outdated_pages.push(branch_page);

    let base = BaseBranch {
        node: branch,
        iter_pos: 0,
    };
    branch_updater.reset_base(Some(base), cutoff);
}

fn run_worker(
    bbn_index: Index,
    page_pool: PagePool,
    changeset: &[(Key, Option<PageNumber>)],
    mut worker_params: WorkerParams<BranchNode<UnsafePageViewMut>>,
) -> (BTreeMap<Key, ChangedBranchEntry>, Vec<Page>) {
    let mut branches_tracker = BranchesTracker::new();
    let mut branch_updater = BranchUpdater::new(page_pool, None, None);
    let mut bbn_outdated_pages = Vec::new();
    let mut pending_left_request = None;
    let mut has_extended_range = false;
    let mut has_finished_workload = false;

    // point branch updater at first branch.
    reset_branch_base(
        &bbn_index,
        &mut branches_tracker,
        &mut bbn_outdated_pages,
        &mut branch_updater,
        has_extended_range,
        changeset[worker_params.op_range.start].0,
    );

    for (key, op) in &changeset[worker_params.op_range.clone()] {
        // ensure key is in scope for branch updater. if not, digest it. merge rightwards until
        //    done _or_ key is in scope.
        while !branch_updater.is_in_scope(&key) {
            let k = if let BranchDigestResult::NeedsMerge(cutoff) =
                branch_updater.digest(&mut branches_tracker)
            {
                cutoff
            } else {
                *key
            };

            try_answer_left_neighbor(
                &mut pending_left_request,
                &mut worker_params,
                &mut branches_tracker,
                has_finished_workload,
            );

            has_extended_range = false;
            if worker_params.range.high.map_or(false, |high| k >= high) {
                has_extended_range = true;
                request_range_extension(&mut worker_params, &mut branches_tracker);
            }

            reset_branch_base(
                &bbn_index,
                &mut branches_tracker,
                &mut bbn_outdated_pages,
                &mut branch_updater,
                has_extended_range,
                k,
            );
        }

        branch_updater.ingest(*key, *op);
    }

    while let BranchDigestResult::NeedsMerge(cutoff) = branch_updater.digest(&mut branches_tracker)
    {
        try_answer_left_neighbor(
            &mut pending_left_request,
            &mut worker_params,
            &mut branches_tracker,
            has_finished_workload,
        );

        has_extended_range = false;
        if worker_params
            .range
            .high
            .map_or(false, |high| cutoff >= high)
        {
            has_extended_range = true;
            request_range_extension(&mut worker_params, &mut branches_tracker);
        }

        reset_branch_base(
            &bbn_index,
            &mut branches_tracker,
            &mut bbn_outdated_pages,
            &mut branch_updater,
            has_extended_range,
            cutoff,
        );
    }

    // Now we are safe to send over our right neighbor to the left one
    // because we're sure to have finished interacting with it
    has_finished_workload = true;

    // wait until left worker concludes or exhausts our range.
    while worker_params.left_neighbor.is_some() {
        // answer any outstanding pending request
        try_answer_left_neighbor(
            &mut pending_left_request,
            &mut worker_params,
            &mut branches_tracker,
            has_finished_workload,
        );
        // we're done with the right worker so we're sure
        // we are able to respond to each request
        assert!(pending_left_request.is_none());

        match worker_params.left_neighbor.as_ref().map(|l| l.rx.recv()) {
            None => continue,
            Some(Ok(item)) => {
                pending_left_request = Some(item);
            }
            Some(Err(_)) => {
                worker_params.left_neighbor = None;
            }
        }
    }

    (branches_tracker.inner, bbn_outdated_pages)
}
