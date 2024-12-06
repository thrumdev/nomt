use std::ops::Range;
use std::sync::Arc;
use threadpool::ThreadPool;

use crate::beatree::{
    allocator::{PageNumber, SyncAllocator},
    branch::node::BranchNode,
    index::Index,
    Key,
};
use crate::io::{IoCommand, IoHandle, IoKind, PagePool};

use super::branch_updater::{BaseBranch, BranchUpdater, DigestResult as BranchDigestResult};
use super::extend_range_protocol::{
    request_range_extension, try_answer_left_neighbor, LeftNeighbor, RightNeighbor, SeparatorRange,
    WorkerParams,
};

/// Tracker of all changes that happen to branch nodes during an update
pub type BranchesTracker = super::NodesTracker<BranchNode>;

/// Data, including buffers, which may be dropped only once I/O has certainly concluded.
#[derive(Default)]
pub struct PostIoDrop {
    deferred_drop_pages: Vec<Vec<Arc<BranchNode>>>,
}

/// Outputs of the branch stage.
#[derive(Default)]
pub struct BranchStageOutput {
    /// The page numbers of all freed pages.
    pub freed_pages: Vec<PageNumber>,
    /// The number of submitted I/Os.
    pub submitted_io: usize,
    /// Data which should be dropped after all submitted I/Os have concluded.
    pub post_io_drop: PostIoDrop,
}

/// Change the btree's branch nodes in the specified way.
pub fn run(
    bbn_index: &mut Index,
    bbn_writer: SyncAllocator,
    page_pool: PagePool,
    io_handle: IoHandle,
    changeset: Vec<(Key, Option<PageNumber>)>,
    thread_pool: ThreadPool,
    num_workers: usize,
) -> anyhow::Result<BranchStageOutput> {
    if changeset.is_empty() {
        return Ok(BranchStageOutput::default());
    }

    assert!(num_workers >= 1);
    let workers = prepare_workers(&*bbn_index, &changeset, num_workers);
    assert!(!workers.is_empty());

    let changeset = Arc::new(changeset);

    let num_workers = workers.len();
    let (worker_result_tx, worker_result_rx) = crossbeam_channel::bounded(num_workers);

    for worker_params in workers {
        let bbn_index = bbn_index.clone();
        let bbn_writer = bbn_writer.clone();
        let page_pool = page_pool.clone();
        let io_handle = io_handle.clone();
        let changeset = changeset.clone();

        let worker_result_tx = worker_result_tx.clone();
        thread_pool.execute(move || {
            // passing the large `Arc` values by reference ensures that they are dropped at the
            // end of this scope, not the end of `run_worker`.
            let res = run_worker(
                bbn_index,
                bbn_writer,
                page_pool,
                io_handle,
                &*changeset,
                worker_params,
            );
            let _ = worker_result_tx.send(res);
        });
    }

    // we don't want to block other sync steps on deallocating these memory regions.
    drop(changeset);

    let mut output = BranchStageOutput::default();

    for _ in 0..num_workers {
        // UNWRAP: results are always sent unless worker panics.
        let worker_output = worker_result_rx.recv().unwrap();
        apply_bbn_changes(bbn_index, &mut output, worker_output);
    }

    Ok(output)
}

fn apply_bbn_changes(
    bbn_index: &mut Index,
    output: &mut BranchStageOutput,
    mut worker_output: BranchWorkerOutput,
) {
    for (key, changed_branch) in worker_output.branches_tracker.inner {
        match changed_branch.inserted {
            Some((node, _pn)) => {
                bbn_index.insert(key, node);
                output.submitted_io += 1;
            }
            None => {
                bbn_index.remove(&key);
            }
        }

        if let Some(deleted_pn) = changed_branch.deleted {
            output.freed_pages.push(deleted_pn);
        }
    }

    output.submitted_io += worker_output.branches_tracker.extra_freed.len();
    output
        .freed_pages
        .extend(worker_output.branches_tracker.extra_freed.drain(..));
    output.post_io_drop.deferred_drop_pages.push(std::mem::take(
        &mut worker_output.branches_tracker.deferred_drop_pages,
    ));
}

fn prepare_workers(
    bbn_index: &Index,
    changeset: &[(Key, Option<PageNumber>)],
    worker_count: usize,
) -> Vec<WorkerParams<BranchNode>> {
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
    branch_updater: &mut BranchUpdater,
    has_extended_range: bool,
    mut key: Key,
) {
    if !has_extended_range {
        reset_branch_base_fresh(bbn_index, branches_tracker, branch_updater, key);
        return;
    }

    if let Some((_, node, next_separator)) = branches_tracker.pending_base.take() {
        let base = BaseBranch::new(node);
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
            reset_branch_base_fresh(bbn_index, branches_tracker, branch_updater, key);
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
    branch_updater: &mut BranchUpdater,
    key: Key,
) {
    let Some((separator, branch)) = bbn_index.lookup(key) else {
        return;
    };

    let cutoff = bbn_index.next_key(key);

    branches_tracker.delete(separator, branch.bbn_pn().into(), cutoff);

    let base = BaseBranch::new(branch);
    branch_updater.reset_base(Some(base), cutoff);
}

struct BranchWorkerOutput {
    branches_tracker: BranchesTracker,
}

fn run_worker(
    bbn_index: Index,
    bbn_writer: SyncAllocator,
    page_pool: PagePool,
    io_handle: IoHandle,
    changeset: &[(Key, Option<PageNumber>)],
    mut worker_params: WorkerParams<BranchNode>,
) -> BranchWorkerOutput {
    let mut branch_updater = BranchUpdater::new(page_pool, None, None);
    let mut pending_left_request = None;
    let mut has_extended_range = false;
    let mut has_finished_workload = false;

    let mut new_branch_state = NewBranchHandler {
        bbn_writer,
        branches_tracker: BranchesTracker::new(),
        io_handle,
    };

    // point branch updater at first branch.
    reset_branch_base(
        &bbn_index,
        &mut new_branch_state.branches_tracker,
        &mut branch_updater,
        has_extended_range,
        changeset[worker_params.op_range.start].0,
    );

    for (key, op) in &changeset[worker_params.op_range.clone()] {
        // ensure key is in scope for branch updater. if not, digest it. merge rightwards until
        //    done _or_ key is in scope.
        while !branch_updater.is_in_scope(&key) {
            let k = if let BranchDigestResult::NeedsMerge(cutoff) =
                branch_updater.digest(&mut new_branch_state)
            {
                cutoff
            } else {
                *key
            };

            try_answer_left_neighbor(
                &mut pending_left_request,
                &mut worker_params,
                &mut new_branch_state.branches_tracker,
                has_finished_workload,
            );

            has_extended_range = false;
            if worker_params.range.high.map_or(false, |high| k >= high) {
                has_extended_range = true;
                request_range_extension(&mut worker_params, &mut new_branch_state.branches_tracker);
            }

            reset_branch_base(
                &bbn_index,
                &mut new_branch_state.branches_tracker,
                &mut branch_updater,
                has_extended_range,
                k,
            );
        }

        branch_updater.ingest(*key, *op);
    }

    while let BranchDigestResult::NeedsMerge(cutoff) = branch_updater.digest(&mut new_branch_state)
    {
        try_answer_left_neighbor(
            &mut pending_left_request,
            &mut worker_params,
            &mut new_branch_state.branches_tracker,
            has_finished_workload,
        );

        has_extended_range = false;
        if worker_params
            .range
            .high
            .map_or(false, |high| cutoff >= high)
        {
            has_extended_range = true;
            request_range_extension(&mut worker_params, &mut new_branch_state.branches_tracker);
        }

        reset_branch_base(
            &bbn_index,
            &mut new_branch_state.branches_tracker,
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
            &mut new_branch_state.branches_tracker,
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

    BranchWorkerOutput {
        branches_tracker: new_branch_state.branches_tracker,
    }
}

struct NewBranchHandler {
    bbn_writer: SyncAllocator,
    branches_tracker: BranchesTracker,
    io_handle: IoHandle,
}

impl super::branch_updater::HandleNewBranch for NewBranchHandler {
    fn handle_new_branch(&mut self, key: Key, mut bbn: BranchNode, cutoff: Option<Key>) {
        let fd = self.bbn_writer.store_fd();

        // TODO: handle error
        let page_number = self.bbn_writer.allocate().unwrap();

        bbn.set_bbn_pn(page_number.0);
        let bbn = Arc::new(bbn);

        // TODO: handle error
        let page = bbn.page();
        self.io_handle
            .send(IoCommand {
                kind: IoKind::WriteRaw(fd, page_number.0 as u64, page),
                user_data: 0,
            })
            .expect("I/O Pool Down");

        self.branches_tracker.insert(key, bbn, cutoff, page_number);
    }
}
