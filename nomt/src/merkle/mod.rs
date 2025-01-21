//! Parallel, pipelined merkle trie updates.
//!
//! This splits the work of warming-up and performing the trie update across worker threads.

use anyhow::Context;
use crossbeam::channel::{self, Receiver, Sender};
use page_set::FrozenSharedPageSet;
use parking_lot::Mutex;

use nomt_core::{
    page_id::PageId,
    trie::{self, KeyPath, Node, ValueHash},
    trie_pos::TriePosition,
};
use seek::Seek;

use std::{collections::HashMap, sync::Arc};

use crate::{
    io::PagePool,
    overlay::LiveOverlay,
    page_cache::{Page, PageCache, ShardIndex},
    rw_pass_cell::WritePassEnvelope,
    store::{BucketInfo, DirtyPage, Store},
    HashAlgorithm, Witness, WitnessedOperations, WitnessedPath, WitnessedRead, WitnessedWrite,
};
use threadpool::ThreadPool;

mod page_set;
mod page_walker;
mod seek;
mod worker;

pub use page_walker::UpdatedPage;

/// Updated pages produced by update workers.
pub struct UpdatedPages(Vec<Vec<UpdatedPage>>);

impl UpdatedPages {
    /// Freeze, label, and iterate all the pages.
    ///
    /// Pages are 'labeled' by placing the page ID into the page data itself prior to freezing.
    pub fn into_frozen_iter(self) -> impl Iterator<Item = (PageId, DirtyPage)> {
        self.0.into_iter().flatten().map(|updated_page| {
            let page = updated_page.page.freeze();
            let dirty = DirtyPage {
                page,
                diff: updated_page.diff,
                bucket: updated_page.bucket_info,
            };
            (updated_page.page_id, dirty)
        })
    }
}

/// Whether a key was read or written.
#[derive(Debug, Clone)]
pub enum KeyReadWrite {
    /// The key was read.
    Read,
    /// The key was written. Contains the written value or `None` if deleted.
    Write(Option<ValueHash>),
    /// The key was both read and written. Contains the written value of `None` if deleted.
    ReadThenWrite(Option<ValueHash>),
}

impl KeyReadWrite {
    fn is_write(&self) -> bool {
        match self {
            KeyReadWrite::Read => false,
            KeyReadWrite::Write(_) | KeyReadWrite::ReadThenWrite(_) => true,
        }
    }

    fn is_read(&self) -> bool {
        match self {
            KeyReadWrite::Read | KeyReadWrite::ReadThenWrite(_) => true,
            KeyReadWrite::Write(_) => false,
        }
    }

    fn written_value(&self) -> Option<Option<ValueHash>> {
        match self {
            KeyReadWrite::Read => None,
            KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(v) => {
                Some(v.as_ref().map(|vh| *vh))
            }
        }
    }
}

/// The update worker pool.
pub struct UpdatePool {
    worker_tp: ThreadPool,
    do_warm_up: bool,
}

impl UpdatePool {
    /// Create a new `UpdatePool`.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is zero.
    pub fn new(num_workers: usize, do_warm_up: bool) -> Self {
        UpdatePool {
            worker_tp: threadpool::Builder::new()
                .num_threads(num_workers)
                .thread_name("nomt-commit".to_string())
                .build(),
            do_warm_up,
        }
    }

    /// Create a `Updater` that uses the underlying pool.
    ///
    /// # Deadlocks
    ///
    /// The Updater expects to have exclusive access to the page cache, so if there
    /// are outstanding read passes, write passes, or threads waiting on write passes,
    /// deadlocks are practically guaranteed at some point during the lifecycle of the Updater.
    pub fn begin<H: HashAlgorithm>(
        &self,
        page_cache: PageCache,
        page_pool: PagePool,
        store: Store,
        overlay: LiveOverlay,
        root: Node,
    ) -> Updater {
        let params = worker::WarmUpParams {
            page_cache: page_cache.clone(),
            overlay: overlay.clone(),
            store: store.clone(),
            root,
        };

        let warm_up = if self.do_warm_up {
            Some(spawn_warm_up::<H>(&self.worker_tp, params))
        } else {
            None
        };

        Updater {
            worker_tp: self.worker_tp.clone(),
            warm_up,
            page_cache,
            root,
            store,
            page_pool,
            overlay,
        }
    }
}

/// Parallel commit handler.
///
/// The expected usage is to call `warm_up` repeatedly and conclude with `commit`.
pub struct Updater {
    worker_tp: ThreadPool,
    page_cache: PageCache,
    warm_up: Option<WarmUpHandle>,
    root: Node,
    store: Store,
    page_pool: PagePool,
    overlay: LiveOverlay,
}

impl Updater {
    /// Warm up the given key-path by pre-fetching the relevant pages.
    pub fn warm_up(&self, key_path: KeyPath) {
        if let Some(ref warm_up) = self.warm_up {
            let _ = warm_up.warmup_tx.send(WarmUpCommand { key_path });
        }
    }

    /// Update the trie with the given key-value read/write operations.
    ///
    /// Key-paths should be in sorted order
    /// and should appear at most once within the vector. Witness specifies whether or not
    /// to collect the witness of the operation.
    /// `into_overlay` specifies whether the results of this will be committed into an overlay,
    /// disabling certain optimizations.
    pub fn update_and_prove<H: HashAlgorithm>(
        self,
        read_write: Vec<(KeyPath, KeyReadWrite)>,
        witness: bool,
        into_overlay: bool,
    ) -> UpdateHandle {
        if let Some(ref warm_up) = self.warm_up {
            let _ = warm_up.finish_tx.send(());
        }
        let shared = Arc::new(UpdateShared {
            witness,
            into_overlay,
            overlay: self.overlay.clone(),
            read_write,
            root_page_pending: Mutex::new(Vec::with_capacity(64)),
        });

        let num_workers = self.page_cache.shard_count();
        let shard_regions = (0..num_workers).map(ShardIndex::Shard).collect::<Vec<_>>();

        // receive warm-ups from worker.
        // TODO: handle error better.
        let (warm_ups, warm_page_set) = if let Some(ref warm_up) = self.warm_up {
            let output = warm_up.output_rx.recv().unwrap();
            (output.paths, Some(output.pages))
        } else {
            (HashMap::new(), None)
        };
        let warm_ups = Arc::new(warm_ups);

        let write_pass = self.page_cache.new_write_pass();
        let worker_passes = write_pass.split_n(shard_regions);

        let (worker_tx, worker_rx) = crossbeam_channel::bounded(num_workers);

        for (worker_id, write_pass) in worker_passes.into_iter().enumerate() {
            let command = UpdateCommand {
                shared: shared.clone(),
                write_pass: write_pass.into_envelope(),
            };

            let params = worker::UpdateParams {
                page_cache: self.page_cache.clone(),
                page_pool: self.page_pool.clone(),
                store: self.store.clone(),
                root: self.root,
                warm_ups: warm_ups.clone(),
                warm_page_set: warm_page_set.clone(),
                command,
                worker_id,
            };
            spawn_updater::<H>(&self.worker_tp, params, worker_tx.clone());
        }

        UpdateHandle {
            shared,
            worker_rx,
            num_workers,
        }
    }
}

/// A handle for waiting on the results of a commit operation.
pub struct UpdateHandle {
    shared: Arc<UpdateShared>,
    worker_rx: Receiver<anyhow::Result<WorkerOutput>>,
    num_workers: usize,
}

impl UpdateHandle {
    /// Wait on the results of the commit operation.
    pub fn join(self) -> Output {
        let mut new_root = None;

        let mut maybe_witness = self.shared.witness.then_some(Witness {
            path_proofs: Vec::new(),
        });

        let mut maybe_witnessed_ops = self.shared.witness.then_some(WitnessedOperations {
            reads: Vec::new(),
            writes: Vec::new(),
        });

        let mut updated_pages = Vec::new();

        let mut path_proof_offset = 0;
        let mut witnessed_start = 0;

        let mut received_outputs = 0;
        for output in self.worker_rx.into_iter() {
            // TODO: handle error better.
            let output = output.unwrap();

            received_outputs += 1;
            if let Some(root) = output.root {
                assert!(new_root.is_none());
                new_root = Some(root);
            }

            updated_pages.push(output.updated_pages);

            // if the Commit worker collected the witnessed paths
            // then we need to aggregate them
            if let Some(witnessed_paths) = output.witnessed_paths {
                // UNWRAP: the same `UpdateShared` object is used to decide whether
                // to collect witnesses or not. If the commit worker did so,
                // `maybe_witness` and `maybe_witnessed_ops` must be initialized to contain
                // all witnesses from all workers.
                let witness = maybe_witness.as_mut().unwrap();
                let witnessed_ops = maybe_witnessed_ops.as_mut().unwrap();

                let path_proof_count = witnessed_paths.len();
                witness.path_proofs.reserve(witnessed_paths.len());
                for (path_index, (path, leaf_data, batch_size)) in
                    witnessed_paths.into_iter().enumerate()
                {
                    witness.path_proofs.push(path);
                    let witnessed_end = witnessed_start + batch_size;
                    for (k, v) in &self.shared.read_write[witnessed_start..witnessed_end] {
                        if v.is_read() {
                            let value_hash = leaf_data.as_ref().and_then(|leaf_data| {
                                if &leaf_data.key_path == k {
                                    Some(leaf_data.value_hash)
                                } else {
                                    None
                                }
                            });

                            witnessed_ops.reads.push(WitnessedRead {
                                key: *k,
                                value: value_hash,
                                path_index: path_index + path_proof_offset,
                            });
                        }
                        if let Some(written) = v.written_value() {
                            witnessed_ops.writes.push(WitnessedWrite {
                                key: *k,
                                value: written,
                                path_index: path_index + path_proof_offset,
                            });
                        }
                    }
                    witnessed_start = witnessed_end;
                }

                path_proof_offset += path_proof_count;
            }
        }

        // TODO: handle error when a worker dies unexpectedly.
        assert_eq!(self.num_workers, received_outputs);

        // UNWRAP: one thread always produces the root.
        Output {
            root: new_root.unwrap(),
            updated_pages: UpdatedPages(updated_pages),
            witness: maybe_witness,
            witnessed_operations: maybe_witnessed_ops,
        }
    }
}

/// The output of a commit operation.
pub struct Output {
    /// The new root.
    pub root: Node,
    /// All updated pages from all worker threads. The covered sets of pages are disjoint.
    pub updated_pages: UpdatedPages,
    /// Optional witness
    pub witness: Option<Witness>,
    /// Optional list of all witnessed operations.
    pub witnessed_operations: Option<WitnessedOperations>,
}

struct UpdateCommand {
    shared: Arc<UpdateShared>,
    write_pass: WritePassEnvelope<ShardIndex>,
}

struct WarmUpCommand {
    key_path: KeyPath,
}

struct WarmUpOutput {
    pages: FrozenSharedPageSet,
    paths: HashMap<KeyPath, Seek>,
}

enum RootPagePending {
    SubTrie {
        range_start: usize,
        range_end: usize,
        prev_terminal: Option<trie::LeafData>,
    },
    Node(Node),
}

struct WorkerOutput {
    root: Option<Node>,
    witnessed_paths: Option<Vec<(WitnessedPath, Option<trie::LeafData>, usize)>>,
    updated_pages: Vec<UpdatedPage>,
}

impl WorkerOutput {
    fn new(witness: bool) -> Self {
        WorkerOutput {
            root: None,
            witnessed_paths: if witness { Some(Vec::new()) } else { None },
            updated_pages: Vec::new(),
        }
    }
}

// Shared data used in committing.
struct UpdateShared {
    read_write: Vec<(KeyPath, KeyReadWrite)>,
    // nodes needing to be written to pages above a shard.
    root_page_pending: Mutex<Vec<(TriePosition, RootPagePending)>>,
    overlay: LiveOverlay,
    witness: bool,
    into_overlay: bool,
}

impl UpdateShared {
    fn push_pending_root_nodes(&self, nodes: Vec<(TriePosition, Node)>) {
        let mut pending = self.root_page_pending.lock();
        for (trie_pos, node) in nodes {
            pending.push((trie_pos, RootPagePending::Node(node)));
        }
    }

    fn push_pending_subtrie(
        &self,
        trie_pos: TriePosition,
        range_start: usize,
        range_end: usize,
        prev_terminal: Option<trie::LeafData>,
    ) {
        self.root_page_pending.lock().push((
            trie_pos,
            RootPagePending::SubTrie {
                range_start,
                range_end,
                prev_terminal,
            },
        ));
    }

    // Takes all pending root page operations, sorted and deduplicated. Note that duplicates are
    // only expected when a pending range was encountered by multiple workers.
    fn take_root_pending(&self) -> Vec<(TriePosition, RootPagePending)> {
        let mut ops = std::mem::take(&mut *self.root_page_pending.lock());
        ops.sort_unstable_by(|a, b| a.0.path().cmp(b.0.path()));
        ops
    }
}

struct WarmUpHandle {
    finish_tx: Sender<()>,
    warmup_tx: Sender<WarmUpCommand>,
    output_rx: Receiver<WarmUpOutput>,
}

fn spawn_warm_up<H: HashAlgorithm>(
    worker_tp: &ThreadPool,
    params: worker::WarmUpParams,
) -> WarmUpHandle {
    let (warmup_tx, warmup_rx) = channel::unbounded();
    let (output_tx, output_rx) = channel::bounded(1);
    let (finish_tx, finish_rx) = channel::bounded(1);

    worker_tp.execute(move || worker::run_warm_up::<H>(params, warmup_rx, finish_rx, output_tx));

    WarmUpHandle {
        warmup_tx,
        finish_tx,
        output_rx,
    }
}

fn spawn_updater<H: HashAlgorithm>(
    worker_tp: &ThreadPool,
    params: worker::UpdateParams,
    output_tx: Sender<anyhow::Result<WorkerOutput>>,
) {
    let worker_id = params.worker_id;
    worker_tp.execute(move || {
        let output_or_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            worker::run_update::<H>(params)
        }));
        let output = output_or_panic
            .unwrap_or_else(|e| Err(anyhow::anyhow!("panic in updater: {:?}", e)))
            .with_context(|| format!("worker {} erred out", worker_id));
        let _ = output_tx.send(output);
    });
}

fn get_in_memory_page(
    overlay: &LiveOverlay,
    page_cache: &PageCache,
    page_id: &PageId,
) -> Option<(Page, BucketInfo)> {
    overlay
        .page(page_id)
        .map(|page| {
            let bucket_info = match page.bucket {
                BucketInfo::Known(ref b) => BucketInfo::Known(*b),
                BucketInfo::FreshOrDependent(ref maybe) => {
                    if let Some(bucket) = maybe.get() {
                        BucketInfo::Known(bucket)
                    } else {
                        BucketInfo::FreshOrDependent(maybe.clone())
                    }
                }
                // PANIC: overlays are never created with `FreshWithNoDependents`.
                BucketInfo::FreshWithNoDependents => panic!(),
            };

            (page.page.clone(), bucket_info)
        })
        .or_else(|| {
            page_cache
                .get(page_id.clone())
                .map(|(page, bucket)| (page, BucketInfo::Known(bucket)))
        })
}
