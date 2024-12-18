//! This module implements the rollback log.
//!
//! The rollback log maintains a list of reverse deltas. A reverse delta contains the prior value
//! for every key that was modified or deleted.
//!
//! The deltas are stored in an in-memory ring buffer. When the buffer size reaches the limit, the
//! oldest deltas are discarded to make space for new ones.
//!
//! The deltas are also persisted on disk in a [`seglog`].

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fs::File,
    io::Cursor,
    path::PathBuf,
    sync::Arc,
};

use crossbeam::channel::{Receiver, RecvError, Sender};
use dashmap::DashMap;
use nomt_core::trie::KeyPath;
use parking_lot::{Condvar, Mutex};
use threadpool::ThreadPool;

use self::delta::Delta;
use crate::{
    beatree::{self, AsyncLookup, OverflowMetadata, ReadTransaction},
    io::{FatPage, IoHandle},
    seglog::{self, RecordId, SegmentedLog},
    KeyReadWrite,
};

mod delta;
#[cfg(test)]
mod tests;

const MAX_SEGMENT_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB

struct InMemory {
    /// The log of deltas that we have accumulated so far.
    ///
    /// The items are pushed onto the back and popped from the front. When the log reaches
    /// [`Shared::max_rollback_log_len`], the oldest delta is discarded.
    ///
    /// The deltas are stored in-memory even after they are dumped on disk. Upon restart, the deltas
    /// are re-read from disk and stored here.
    log: VecDeque<(RecordId, Delta)>,

    /// If this is set, then the next writeout will truncate the log at this offset.
    pending_truncate: Option<u64>,
}

struct Shared {
    worker_tp: ThreadPool,
    sync_tp: ThreadPool,
    in_memory: Mutex<InMemory>,
    seglog: Mutex<SegmentedLog>,
    /// The number of items that we should keep in the log. Deltas that are past this limit are
    /// discarded.
    max_rollback_log_len: usize,
}

impl InMemory {
    fn new() -> Self {
        Self {
            log: VecDeque::new(),
            pending_truncate: None,
        }
    }

    /// Push a delta into the in-memory cache.
    fn push_back(&mut self, record_id: RecordId, delta: Delta) {
        self.log.push_back((record_id, delta));
    }

    fn pop_back(&mut self) -> Option<(RecordId, Delta)> {
        self.log.pop_back()
    }

    // Returns the total number of deltas, including the staged one.
    fn total_len(&self) -> usize {
        self.log.len()
    }
}

const ROLLBACK_TP_SIZE: usize = 2;

/// This structure manages the rollback log. Modifications to the rollback log are made using
/// [`ReverseDeltaBuilder`] supplied to [`Rollback::commit`].
#[derive(Clone)]
pub struct Rollback {
    shared: Arc<Shared>,
}

impl Rollback {
    pub fn read(
        max_rollback_log_len: u32,
        db_dir_path: PathBuf,
        db_dir_fd: Arc<File>,
        rollback_start_active: u64,
        rollback_end_active: u64,
    ) -> anyhow::Result<Self> {
        let mut in_memory = InMemory::new();
        let seglog = seglog::open(
            db_dir_path,
            db_dir_fd,
            "rollback".to_string(),
            MAX_SEGMENT_SIZE,
            rollback_start_active.into(),
            rollback_end_active.into(),
            |record_id, payload| {
                let mut cursor = Cursor::new(payload);
                let delta = Delta::decode(&mut cursor)?;
                in_memory.push_back(record_id, delta);
                Ok(())
            },
        )?;
        let shared = Arc::new(Shared {
            worker_tp: ThreadPool::new(ROLLBACK_TP_SIZE),
            sync_tp: ThreadPool::new(1),
            in_memory: Mutex::new(in_memory),
            seglog: Mutex::new(seglog),
            max_rollback_log_len: max_rollback_log_len as usize,
        });
        Ok(Self { shared })
    }

    /// Begin a rollback delta.
    pub fn delta_builder(&self, store: &crate::Store) -> ReverseDeltaBuilder {
        self.delta_builder_inner(StoreLoadValueAsync::new(store))
    }

    // generality is primarily for testing.
    fn delta_builder_inner(&self, store: impl LoadValueAsync) -> ReverseDeltaBuilder {
        let (command_tx, command_rx) = crossbeam::channel::unbounded();
        let (completion_tx, completion_rx) = crossbeam::channel::unbounded();

        let priors = Arc::new(DashMap::new());
        let next_fn = store.build_next();
        self.shared.worker_tp.execute({
            let priors = priors.clone();
            move || reverse_delta_worker(store, command_rx, completion_rx, priors)
        });
        self.shared
            .worker_tp
            .execute(move || reverse_delta_completion_worker(completion_tx, next_fn));
        ReverseDeltaBuilder {
            command_tx,
            tp: self.shared.worker_tp.clone(),
            priors,
        }
    }

    /// Saves the delta into the log.
    ///
    /// This function accepts the final list of operations that should be performed sorted by the
    /// key paths in ascending order.
    pub fn commit(
        &self,
        actuals: &[(KeyPath, KeyReadWrite)],
        delta: ReverseDeltaBuilder,
    ) -> anyhow::Result<()> {
        let delta = delta.finalize(actuals);
        let delta_bytes = delta.encode();

        let mut in_memory = self.shared.in_memory.lock();
        let mut seglog = self.shared.seglog.lock();

        let record_id = seglog.append(&delta_bytes)?;
        in_memory.push_back(record_id, delta);
        Ok(())
    }

    /// Truncates the rollback log by removing the last `n` deltas.
    ///
    /// This function returns the keys and values that we should apply to the database to restore
    /// the state as it was before the last `n` deltas were applied.
    ///
    /// This function is destructive and consumes the rollback log.
    pub fn truncate(
        &self,
        mut n: usize,
    ) -> anyhow::Result<Option<BTreeMap<KeyPath, Option<Vec<u8>>>>> {
        assert!(n > 0);
        let mut in_memory = self.shared.in_memory.lock();
        if n > in_memory.total_len() {
            return Ok(None);
        }

        let mut traceback = BTreeMap::new();
        let mut earliest_record_id = None;
        while n > 0 {
            // Pop the most recent delta from the log and add its original values to the traceback,
            // potentially overwriting some of the values that were added in previous iterations.
            //
            // UNWRAP: we checked above that `n` is greater or equal to the total number of deltas
            //         and `n` is strictly decreasing.
            let (record_id, delta) = in_memory.pop_back().unwrap();
            earliest_record_id = Some(record_id);
            for (key, value) in delta.priors {
                traceback.insert(key, value);
            }
            n -= 1;
        }
        // UNWRAP: we checked above that `n` is greater than 0 and that means that there is at
        //         least one delta that we will remove.
        let earliest_record_id = earliest_record_id.unwrap();

        // We got the earliest record ID that we will remove. To obtain the new live range start
        // we need to get the ID of the element prior to that, hence `prev`.
        //
        // UNWRAP: `prev` returns `None` iff the record ID is nil. We know that `earliest_record_id`
        //         can't be nil because there were some elements in the log that we removed.
        let new_end_live = earliest_record_id.prev().unwrap();

        // Set pending truncate to the new end live.
        //
        // We cannot prune the log right away. If we did, and the process crashed, the log could
        // get truncated but the manifest would still reference the old, longer, range.
        in_memory.pending_truncate = Some(new_end_live.0);

        Ok(Some(traceback))
    }

    /// Returns a controller for the sync process.
    pub fn sync(&self) -> SyncController {
        SyncController::new(self.clone())
    }

    /// Dumps the contents of the staging to the rollback.
    fn writeout_start(&self) -> WriteoutData {
        let mut in_memory = self.shared.in_memory.lock();
        let seglog = self.shared.seglog.lock();

        let pending_truncate = in_memory.pending_truncate.take();

        // NOTE: for now, if there is a pending truncate, we ignore everything else.
        if let Some(pending_truncate) = pending_truncate {
            return WriteoutData {
                rollback_start_live: seglog.live_range().0 .0,
                rollback_end_live: pending_truncate,
                prune_to_new_start_live: None,
                prune_to_new_end_live: Some(pending_truncate),
            };
        }

        let prune_to_new_start_live = if in_memory.total_len() > self.shared.max_rollback_log_len {
            Some(in_memory.pop_back().unwrap().0.next().0)
        } else {
            None
        };

        let (rollback_start_live, rollback_end_live) = seglog.live_range();

        WriteoutData {
            rollback_start_live: rollback_start_live.0,
            rollback_end_live: rollback_end_live.0,
            prune_to_new_start_live,
            prune_to_new_end_live: None,
        }
    }

    /// Finalizes the writeout.
    ///
    /// This should be called after the manifest has been updated with the new rollback log live
    /// range.
    ///
    /// We use this point in time to prune the rollback log, by removing the deltas that are no
    /// longer needed.
    fn writeout_end(
        &self,
        new_start_live: Option<u64>,
        new_end_live: Option<u64>,
    ) -> anyhow::Result<()> {
        if let Some(new_start_live) = new_start_live {
            let mut seglog = self.shared.seglog.lock();
            seglog.prune_back(new_start_live.into())?;
        }
        if let Some(new_end_live) = new_end_live {
            let mut seglog = self.shared.seglog.lock();
            seglog.prune_front(new_end_live.into())?;
        }
        Ok(())
    }
}

pub struct SyncController {
    rollback: Rollback,
    wd: Arc<Mutex<Option<WriteoutData>>>,
    wd_cv: Arc<Condvar>,
    post_meta: Arc<Mutex<Option<anyhow::Result<()>>>>,
    post_meta_cv: Arc<Condvar>,
}

impl SyncController {
    fn new(rollback: Rollback) -> Self {
        let wd = Arc::new(Mutex::new(None));
        let wd_cv = Arc::new(Condvar::new());
        let post_meta = Arc::new(Mutex::new(None));
        let post_meta_cv = Arc::new(Condvar::new());
        Self {
            rollback,
            wd,
            wd_cv,
            post_meta,
            post_meta_cv,
        }
    }

    /// Begins the sync process.
    ///
    /// This function doesn't block.
    pub fn begin_sync(&mut self) {
        let tp = self.rollback.shared.sync_tp.clone();
        let rollback = self.rollback.clone();
        let wd = self.wd.clone();
        let wd_cv = self.wd_cv.clone();
        tp.execute(move || {
            let writeout_data = rollback.writeout_start();
            let _ = wd.lock().replace(writeout_data);
            wd_cv.notify_one();
        });
    }

    /// Wait for the rollback writeout to complete. Returns the new rollback live range
    /// `(start_live, end_live)`.
    ///
    /// This should be called by the sync thread. Blocking.
    pub fn wait_pre_meta(&self) -> (u64, u64) {
        let mut wd = self.wd.lock();
        self.wd_cv.wait_while(&mut wd, |wd| wd.is_none());
        // UNWRAP: we checked above that `wd` is not `None`.
        let wd = wd.as_ref().unwrap();
        (wd.rollback_start_live, wd.rollback_end_live)
    }

    /// This should be called after the meta has been updated.
    ///
    /// This function doesn't block.
    pub fn post_meta(&self) {
        let tp = self.rollback.shared.sync_tp.clone();
        let wd = self.wd.lock().take().unwrap();
        let post_meta = self.post_meta.clone();
        let post_meta_cv = self.post_meta_cv.clone();
        let rollback = self.rollback.clone();
        tp.execute(move || {
            let result =
                rollback.writeout_end(wd.prune_to_new_start_live, wd.prune_to_new_end_live);
            let _ = post_meta.lock().replace(result);
            post_meta_cv.notify_one();
        });
    }

    /// Wait until the post-meta writeout completes.
    pub fn wait_post_meta(&self) -> anyhow::Result<()> {
        let mut post_meta = self.post_meta.lock();
        self.post_meta_cv
            .wait_while(&mut post_meta, |post_meta| post_meta.is_none());
        // UNWRAP: we checked above that `post_meta` is not `None`.
        post_meta.take().unwrap()
    }
}

struct WriteoutData {
    rollback_start_live: u64,
    rollback_end_live: u64,
    /// If this is `Some`, then the [`Rollback::writeout_end`] should be called with this value.
    prune_to_new_start_live: Option<u64>,
    /// If this is `Some`, then the [`Rollback::writeout_end`] should be called with this value.
    prune_to_new_end_live: Option<u64>,
}

pub struct ReverseDeltaBuilder {
    command_tx: Sender<DeltaBuilderCommand>,
    tp: ThreadPool,
    /// The values of the keys that should be preserved at commit time for this delta.
    ///
    /// Before the commit takes place, the set contains tentative values.
    priors: Arc<DashMap<KeyPath, Option<Vec<u8>>>>,
}

/// A trait for asynchronously loading values from the store.
///
/// This family of traits allows us to mock the store in tests.
///
/// Dropping a value implementing this trait should end its corresponding completion stream.
pub trait LoadValueAsync: Send + Sync + Sized + 'static {
    type Pending: AsyncPending<Store = Self>;
    type Completion: Send;

    /// Start a load. This may complete eagerly or return a pending result.
    fn start_load(
        &self,
        key_path: KeyPath,
        user_data: u64,
    ) -> Result<Option<Vec<u8>>, Self::Pending>;

    /// Create the completion function.
    fn build_next(
        &self,
    ) -> impl FnMut() -> Option<(u64, anyhow::Result<Self::Completion>)> + Send + 'static;
}

/// A trait for a pending asynchronous value load.
///
/// This family of traits allows us to mock the store in tests.
pub trait AsyncPending {
    type Store: LoadValueAsync;
    type OverflowMetadata;

    fn submit(&mut self, store: &Self::Store, user_data: u64) -> Option<Self::OverflowMetadata>;
    fn try_complete(
        &mut self,
        completion: <Self::Store as LoadValueAsync>::Completion,
        meta: Option<Self::OverflowMetadata>,
    ) -> Option<Option<Vec<u8>>>;
}

struct LoadValueCompletion<T>(u64, anyhow::Result<T>);

enum DeltaBuilderCommand {
    /// Start lookup on a key path.
    Lookup(KeyPath),
    /// Join all outstanding lookups, then send a signal over the channel.
    /// Replace the priors with the given map.
    Join(Sender<()>, Arc<DashMap<KeyPath, Option<Vec<u8>>>>),
}

// This worker manages all reverse value requests asynchronously.
fn reverse_delta_worker<Store: LoadValueAsync>(
    store: Store,
    command_rx: Receiver<DeltaBuilderCommand>,
    completion_rx: Receiver<LoadValueCompletion<Store::Completion>>,
    mut priors: Arc<DashMap<KeyPath, Option<Vec<u8>>>>,
) {
    const TARGET_OVERFLOW_REQUESTS: usize = 128;

    enum LiveRequest<Req: AsyncPending> {
        Main(Req, KeyPath),
        OverflowPage(u64, Req::OverflowMetadata),
    }

    let mut requests = HashMap::new();

    // regular request count starts at 0 and moves upwards.
    let mut request_index = 0u64;
    // overflow requests start at max and move downwards.
    let mut overflow_request_index = u64::MAX;

    // INVARIANT: store is always `Some` until `joining` is true and requests are empty.
    let mut store = Some(store);
    let mut joining = false;

    let mut handle_lookup = |key_path,
                             store: &mut Option<Store>,
                             priors: &DashMap<_, _>,
                             requests: &mut HashMap<_, _>| {
        // UNWRAP: `Load` commands never come after finish.
        let store = store.as_ref().unwrap();
        match store.start_load(key_path, request_index) {
            Ok(val) => {
                priors.insert(key_path, val);
            }
            Err(pending) => {
                requests.insert(request_index, LiveRequest::Main(pending, key_path));
                request_index += 1;
            }
        }
    };

    // the number of requests which are "dormant", i.e. they are used only for dispatching
    // overflow requests.
    let mut dormant_request_count = 0;

    let mut handle_completion =
        |user_data,
         store: &mut Option<Store>,
         joining: bool,
         maybe_completion: anyhow::Result<_>,
         priors: &DashMap<_, _>,
         requests: &mut HashMap<_, LiveRequest<Store::Pending>>| {
            // TODO handle error properly
            let completion: Store::Completion = maybe_completion.unwrap();

            // UNWRAP: completions come for occupied entries only.
            let request = requests.remove(&user_data).unwrap();
            let resubmit = match request {
                LiveRequest::Main(mut request, key_path) => {
                    match request.try_complete(completion, None) {
                        Some(value) => {
                            priors.insert(key_path, value);
                            None
                        }
                        None => {
                            // this is now an overflow request. count it as dormant and insert
                            // back.
                            dormant_request_count += 1;
                            requests.insert(user_data, LiveRequest::Main(request, key_path));
                            Some(user_data)
                        }
                    }
                }
                LiveRequest::OverflowPage(request_id, overflow_meta) => {
                    // UNWRAP: initial request always exists
                    let overflow_request = requests.get_mut(&request_id).unwrap();
                    // PANIC: ...and has kind initial.
                    let LiveRequest::Main(ref mut pending, ref key_path) = overflow_request else {
                        unreachable!()
                    };

                    // If we get a value, insert it in priors and stop tracking the request.
                    if let Some(value) = pending.try_complete(completion, Some(overflow_meta)) {
                        dormant_request_count -= 1;
                        priors.insert(*key_path, value);
                        requests.remove(&request_id);
                        None
                    } else {
                        Some(request_id)
                    }
                }
            };

            if joining && requests.is_empty() {
                *store = None;
            } else if let Some(resubmit_id) = resubmit {
                // we received an overflow continuation. submit and continue.

                let submitted = {
                    let non_dormant_count = requests.len() - dormant_request_count;

                    // UNWRAP: resubmit request always exists.
                    let overflow_request = requests.get_mut(&resubmit_id).unwrap();
                    // PANIC: ...and has kind initial.
                    let LiveRequest::Main(ref mut pending, _) = overflow_request else {
                        unreachable!()
                    };
                    // UNWRAP: ...and the `Store` must be live.
                    let store = store.as_ref().unwrap();

                    let mut submitted = Vec::new();

                    // Always submit at least one, but never more than the maximum.
                    let submit_count = std::cmp::max(
                        1,
                        TARGET_OVERFLOW_REQUESTS.saturating_sub(non_dormant_count),
                    );
                    for _ in 0..submit_count {
                        let Some(meta) = pending.submit(store, overflow_request_index) else {
                            break;
                        };
                        submitted.push((
                            overflow_request_index,
                            LiveRequest::OverflowPage(resubmit_id, meta),
                        ));
                        overflow_request_index -= 1;
                    }

                    submitted
                };

                requests.extend(submitted);
            }
        };

    loop {
        let join = crossbeam::select! {
            recv(command_rx) -> command => match command {
                Err(RecvError) => {
                    joining = true;
                    if requests.is_empty() {
                        store = None;
                    }
                    break
                }
                Ok(DeltaBuilderCommand::Lookup(key_path)) => {
                    handle_lookup(key_path, &mut store, &priors, &mut requests);
                    None
                },
                Ok(DeltaBuilderCommand::Join(done_joining, new_priors)) => Some((done_joining, new_priors)),
            },
            recv(completion_rx) -> completion => match completion {
                // PANIC: completions never disconnect before commands.
                Err(RecvError) => unreachable!(),
                Ok(LoadValueCompletion(user_data, completion)) => {
                    handle_completion(user_data, &mut store, joining, completion, &priors, &mut requests);
                    None
                },
            },
        };

        if let Some((done_joining, new_priors)) = join {
            while !requests.is_empty() {
                // UNWRAP: completions never disconnect before `commands` does.
                let LoadValueCompletion(user_data, completion) = completion_rx.recv().unwrap();
                handle_completion(
                    user_data,
                    &mut store,
                    joining,
                    completion,
                    &priors,
                    &mut requests,
                );
            }

            priors = new_priors;
            let _ = done_joining.send(());
        }
    }

    // wait on outstanding completions.
    for LoadValueCompletion(user_data, completion) in completion_rx {
        handle_completion(
            user_data,
            &mut store,
            joining,
            completion,
            &priors,
            &mut requests,
        );
    }
}

fn reverse_delta_completion_worker<F, T>(completion_tx: Sender<LoadValueCompletion<T>>, mut next: F)
where
    F: FnMut() -> Option<(u64, anyhow::Result<T>)>,
{
    while let Some((user_data, result)) = next() {
        let _ = completion_tx.send(LoadValueCompletion(user_data, result));
    }
}

impl ReverseDeltaBuilder {
    /// Note that a write might be made to a key and that the rollback should preserve the prior
    /// value. This function is speculative; the rollback delta may later be committed with a
    /// different set of operations, and some of the tentative operations may be discarded.
    ///
    /// This function doesn't block.
    pub fn tentative_preserve_prior(&self, key_path: KeyPath) {
        let _ = self.command_tx.send(DeltaBuilderCommand::Lookup(key_path));
    }

    /// Finalize the delta.
    ///
    /// This function is expected to be called before the store is modified.
    fn finalize(self, actuals: &[(KeyPath, KeyReadWrite)]) -> Delta {
        // wait for all submitted requests to finish.
        let fresh_priors = Arc::new(DashMap::new());
        let (join_tx, join_rx) = crossbeam::channel::bounded(1);
        let _ = self
            .command_tx
            .send(DeltaBuilderCommand::Join(join_tx, fresh_priors.clone()));
        let _ = join_rx.recv();

        // At this point, `tentative_priors` is unique, because the worker has swapped
        // with `fresh_priors`.
        let tentative_priors = self.priors;
        let mut final_priors = HashMap::with_capacity(tentative_priors.len() * 2);

        for (path, read_write) in actuals {
            match read_write {
                KeyReadWrite::Read(_) => {
                    // The path was read. We don't need to preserve anything.
                }
                KeyReadWrite::Write(_) => {
                    // The path was written. If we have a tentative value, keep it. Otherwise, fetch
                    // the current value from the store.
                    if let Some((path, value)) = tentative_priors.remove(path) {
                        // Tentative speculation was a hit. Keep the entry.
                        final_priors.insert(path, value);
                    } else {
                        // The delta builder was not aware of this write. Initiate a fetch from the store
                        // and record the result as a prior.
                        let _ = self.command_tx.send(DeltaBuilderCommand::Lookup(*path));
                    }
                }
                KeyReadWrite::ReadThenWrite(prior, _) => {
                    // The path was read and then written. We could just keep the prior value.
                    final_priors.insert(*path, prior.clone());
                }
            }
        }

        // Wait for the load worker to join. After this point, priors contains the final set of
        // values to be preserved.
        drop(self.command_tx);
        let _ = self.tp.join();

        // UNWRAP: At this point, `fresh_priors` is unique because the worker thread has joined.
        // At this point, fresh_priors is fully populated with all lookups submitted in the loop.
        let fresh_priors = Arc::into_inner(fresh_priors).unwrap().into_iter();
        final_priors.extend(fresh_priors);
        Delta {
            priors: final_priors,
        }
    }
}

pub struct StoreLoadValueAsync {
    read_tx: ReadTransaction,
    io_handle: IoHandle,
}

impl StoreLoadValueAsync {
    /// Create a new asynchronous value loader from the store.
    pub fn new(store: &crate::store::Store) -> Self {
        let read_tx = store.read_transaction();
        let io_handle = store.io_pool().make_handle();

        StoreLoadValueAsync { read_tx, io_handle }
    }
}

impl AsyncPending for AsyncLookup {
    type Store = StoreLoadValueAsync;
    type OverflowMetadata = beatree::OverflowMetadata;

    fn submit(
        &mut self,
        store: &StoreLoadValueAsync,
        user_data: u64,
    ) -> Option<beatree::OverflowMetadata> {
        AsyncLookup::submit(self, &store.io_handle, user_data)
    }

    fn try_complete(
        &mut self,
        completion: FatPage,
        overflow_meta: Option<OverflowMetadata>,
    ) -> Option<Option<Vec<u8>>> {
        self.try_finish(completion, overflow_meta)
    }
}

impl LoadValueAsync for StoreLoadValueAsync {
    type Completion = FatPage;
    type Pending = AsyncLookup;

    fn start_load(
        &self,
        key_path: KeyPath,
        user_data: u64,
    ) -> Result<Option<Vec<u8>>, Self::Pending> {
        self.read_tx
            .lookup_async(key_path, &self.io_handle, user_data)
    }

    fn build_next(
        &self,
    ) -> impl FnMut() -> Option<(u64, anyhow::Result<Self::Completion>)> + Send + 'static {
        // using just the receiver ensures that when `self` is dropped, the completion stream ends.
        let mut receiver = self.io_handle.receiver().clone().into_iter();
        move || {
            receiver.next().map(|complete_io| {
                let user_data = complete_io.command.user_data;
                let res = complete_io
                    .result
                    .map(|()| complete_io.command.kind.unwrap_buf());
                (user_data, res.map_err(Into::into))
            })
        }
    }
}
