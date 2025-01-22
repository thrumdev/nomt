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

use crate::overlay::LiveOverlay;
use crossbeam::channel::Sender;
use dashmap::DashMap;
use nomt_core::trie::KeyPath;
use parking_lot::{Condvar, Mutex};
use threadpool::ThreadPool;

use self::reverse_delta_worker::{DeltaBuilderCommand, LoadValueAsync, StoreLoadValueAsync};
use crate::{
    seglog::{self, RecordId, SegmentedLog},
    KeyReadWrite,
};

pub use self::delta::Delta;

mod delta;
mod reverse_delta_worker;
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
    pub fn delta_builder(
        &self,
        store: &crate::Store,
        overlay: &LiveOverlay,
    ) -> ReverseDeltaBuilder {
        self.delta_builder_inner(StoreLoadValueAsync::new(store, overlay.clone()))
    }

    // generality is primarily for testing.
    fn delta_builder_inner(&self, store: impl LoadValueAsync) -> ReverseDeltaBuilder {
        let priors = Arc::new(DashMap::new());
        let command_tx = reverse_delta_worker::start(store, &self.shared.worker_tp, priors.clone());
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
    pub fn commit(&self, delta: Delta) -> anyhow::Result<()> {
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
    pub fn finalize(self, actuals: &[(KeyPath, KeyReadWrite)]) -> Delta {
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
