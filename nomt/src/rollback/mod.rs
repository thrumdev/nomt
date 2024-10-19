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
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::Cursor,
    path::PathBuf,
    sync::Arc,
};

use dashmap::DashMap;
use nomt_core::trie::KeyPath;
use parking_lot::Mutex;
use threadpool::ThreadPool;

use self::delta::Delta;
use crate::{
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

/// This structure manages the rollback log. Modifications to the rollback log are made using
/// [`ReverseDeltaBuilder`] supplied to [`Rollback::commit`].
#[derive(Clone)]
pub struct Rollback {
    shared: Arc<Shared>,
}

impl Rollback {
    pub fn read(
        max_rollback_log_len: u32,
        rollback_tp_size: usize,
        db_dir_path: PathBuf,
        db_dir_fd: File,
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
            worker_tp: ThreadPool::new(rollback_tp_size),
            in_memory: Mutex::new(in_memory),
            seglog: Mutex::new(seglog),
            max_rollback_log_len: max_rollback_log_len as usize,
        });
        Ok(Self { shared })
    }

    /// Begin a rollback delta.
    pub fn delta_builder(&self) -> ReverseDeltaBuilder {
        ReverseDeltaBuilder {
            tp: self.shared.worker_tp.clone(),
            priors: Arc::new(DashMap::new()),
        }
    }

    /// Saves the delta into the log.
    ///
    /// This function accepts the final list of operations that should be performed sorted by the
    /// key paths in ascending order.
    pub fn commit(
        &self,
        store: impl LoadValue,
        actuals: &[(KeyPath, KeyReadWrite)],
        delta: ReverseDeltaBuilder,
    ) -> anyhow::Result<()> {
        let delta = delta.finalize(store, actuals);
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

    /// Dumps the contents of the staging to the rollback.
    pub fn writeout_start(&self) -> anyhow::Result<WriteoutData> {
        let mut in_memory = self.shared.in_memory.lock();
        let seglog = self.shared.seglog.lock();

        let pending_truncate = in_memory.pending_truncate.take();

        // NOTE: for now, if there is a pending truncate, we ignore everything else.
        if let Some(pending_truncate) = pending_truncate {
            return Ok(WriteoutData {
                rollback_start_live: seglog.live_range().0 .0,
                rollback_end_live: pending_truncate,
                prune_to_new_start_live: None,
                prune_to_new_end_live: Some(pending_truncate),
            });
        }

        let prune_to_new_start_live = if in_memory.total_len() > self.shared.max_rollback_log_len {
            Some(in_memory.pop_back().unwrap().0.next().0)
        } else {
            None
        };

        let (rollback_start_live, rollback_end_live) = seglog.live_range();

        Ok(WriteoutData {
            rollback_start_live: rollback_start_live.0,
            rollback_end_live: rollback_end_live.0,
            prune_to_new_start_live,
            prune_to_new_end_live: None,
        })
    }

    /// Finalizes the writeout.
    ///
    /// This should be called after the manifest has been updated with the new rollback log live
    /// range.
    ///
    /// We use this point in time to prune the rollback log, by removing the deltas that are no
    /// longer needed.
    pub fn writeout_end(
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

pub struct WriteoutData {
    pub rollback_start_live: u64,
    pub rollback_end_live: u64,
    /// If this is `Some`, then the [`Rollback::writeout_end`] should be called with this value.
    pub prune_to_new_start_live: Option<u64>,
    /// If this is `Some`, then the [`Rollback::writeout_end`] should be called with this value.
    pub prune_to_new_end_live: Option<u64>,
}

pub struct ReverseDeltaBuilder {
    tp: ThreadPool,
    /// The values of the keys that should be preserved at commit time for this delta.
    ///
    /// Before the commit takes place, the set contains tentative values.
    priors: Arc<DashMap<KeyPath, Option<Vec<u8>>>>,
}

/// A trait for loading values from the store.
///
/// This seam allows us to mock the store in tests.
pub trait LoadValue: Clone + Send + Sync + 'static {
    fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>>;
}

impl LoadValue for crate::store::Store {
    fn load_value(&self, key_path: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        self.load_value(key_path)
    }
}

impl ReverseDeltaBuilder {
    /// Note that a write might be made to a key and that the rollback should preserve the prior
    /// value. This function is speculative; the rollback delta may later be committed with a
    /// different set of operations, and some of the tentative operations may be discarded.
    ///
    /// This function doesn't block.
    pub fn tentative_preserve_prior(&self, store: impl LoadValue, key_path: KeyPath) {
        self.tp.execute({
            let priors = self.priors.clone();
            move || {
                let value = store.load_value(key_path).unwrap();
                priors.insert(key_path, value);
            }
        });
    }

    /// Finalize the delta.
    ///
    /// This function is expected to be called before the store is modified.
    fn finalize(self, store: impl LoadValue, actuals: &[(KeyPath, KeyReadWrite)]) -> Delta {
        // Wait for all tentative writes issued so far to complete.
        //
        // NB: This doesn't take into account other users of `tp`. If there are any, we will be
        // needlessly blocking on them.
        self.tp.join();

        let tentative_priors = Arc::clone(&self.priors);
        let final_priors = Arc::new(DashMap::with_capacity(tentative_priors.len() * 2));

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
                        let store = store.clone();
                        let path = path.clone();
                        let final_priors = final_priors.clone();
                        self.tp.execute(move || {
                            let value = store.load_value(path).unwrap();
                            final_priors.insert(path, value);
                        });
                    }
                }
                KeyReadWrite::ReadThenWrite(prior, _) => {
                    // The path was read and then written. We could just keep the prior value.
                    final_priors.insert(*path, prior.clone());
                }
            }
        }

        // Wait for all the fetches to complete. After this point, priors contains the final set of
        // values to be preserved.
        self.tp.join();

        Delta {
            priors: Arc::into_inner(final_priors).unwrap().into_iter().collect(),
        }
    }
}
