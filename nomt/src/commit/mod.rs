//! Parallel, pipelined merkle trie commits.
//!
//! This splits the work of warming-up and performing the trie update across worker threads.

use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::Mutex;

use nomt_core::{
    page_id::PageId,
    trie::{self, KeyPath, Node, NodeHasher, ValueHash},
    trie_pos::TriePosition,
};

use std::sync::{Arc, Barrier};

use crate::{
    page_cache::{PageCache, ShardIndex},
    page_diff::PageDiff,
    rw_pass_cell::WritePassEnvelope,
    store::Store,
    Witness, WitnessedOperations, WitnessedPath, WitnessedRead, WitnessedWrite,
};
use threadpool::ThreadPool;

mod worker;

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

/// The commit worker pool.
pub struct CommitPool {
    worker_tp: ThreadPool,
}

impl CommitPool {
    /// Create a new `CommitPool`.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is zero.
    pub fn new(num_workers: usize) -> Self {
        CommitPool {
            worker_tp: threadpool::Builder::new()
                .num_threads(num_workers)
                .thread_name("nomt-commit".to_string())
                .build(),
        }
    }

    /// Create a `Committer` that uses the underlying pool.
    ///
    /// # Deadlocks
    ///
    /// The Committer expects to have exclusive access to the page cache, so if there
    /// are outstanding read passes, write passes, or threads waiting on write passes,
    /// deadlocks are practically guaranteed at some point during the lifecycle of the Committer.
    pub fn begin<H: NodeHasher>(
        &self,
        page_cache: PageCache,
        store: Store,
        root: Node,
    ) -> Committer {
        let num_workers = page_cache.shard_count();

        let barrier = Arc::new(Barrier::new(num_workers + 1));

        let workers: Vec<WorkerHandle> = (0..num_workers)
            .map(|_| {
                let params = worker::Params {
                    page_cache: page_cache.clone(),
                    store: store.clone(),
                    root,
                    barrier: barrier.clone(),
                };
                spawn_worker::<H>(&self.worker_tp, params)
            })
            .collect();

        // wait until all workers are spawned and have their read pass, otherwise this can race
        // with `commit` being called.
        let _ = barrier.wait();

        Committer {
            worker_tp: self.worker_tp.clone(),
            workers,
            page_cache,
            worker_round_robin: 0,
        }
    }
}

/// Parallel commit handler.
///
/// The expected usage is to call `warm_up` repeatedly and conclude with `commit`.
pub struct Committer {
    worker_tp: ThreadPool,
    page_cache: PageCache,
    workers: Vec<WorkerHandle>,
    worker_round_robin: usize,
}

impl Committer {
    /// Warm up the given key-path by pre-fetching the relevant pages.
    pub fn warm_up(&mut self, key_path: KeyPath) {
        let worker = self.worker_round_robin;
        self.worker_round_robin += 1;
        self.worker_round_robin %= self.workers.len();

        let _ = self.workers[worker]
            .warmup_tx
            .send(WarmUpCommand { key_path });
    }

    /// Commit the given key-value read/write operations. Key-paths should be in sorted order
    /// and should appear at most once within the vector. Witness specify whether or not
    /// collecting the witness of the commit operation.
    pub fn commit(
        mut self,
        read_write: Vec<(KeyPath, KeyReadWrite)>,
        witness: bool,
    ) -> CommitHandle {
        let shared = Arc::new(CommitShared {
            witness,
            read_write,
            root_page_pending: Mutex::new(Vec::with_capacity(64)),
        });

        for worker in &self.workers {
            let _ = worker.commit_tx.send(ToWorker::Prepare);
        }

        let write_pass = self.page_cache.new_write_pass();
        let shard_regions = (0..self.workers.len())
            .map(ShardIndex::Shard)
            .collect::<Vec<_>>();
        let worker_passes = write_pass.split_n(shard_regions);

        for (worker, write_pass) in self.workers.iter().zip(worker_passes) {
            // TODO: handle error better
            worker
                .commit_tx
                .send(ToWorker::Commit(CommitCommand {
                    shared: shared.clone(),
                    write_pass: write_pass.into_envelope(),
                }))
                .unwrap();
        }

        CommitHandle {
            shared,
            workers: std::mem::take(&mut self.workers),
        }
    }
}

impl Drop for Committer {
    fn drop(&mut self) {
        if self.workers.is_empty() {
            return;
        }
        self.workers.clear();

        // hack: we need to do this to avoid the store from being dropped in a worker thread,
        // which RocksDB really doesn't like and is not really thread-safe despite being "safe"
        // code. remove this line to get a free unlimited supply of SIGSEGV and SIGABRT on shutdown
        // whenever a `Session` is live.
        self.worker_tp.join();
    }
}

/// A handle for waiting on the results of a commit operation.
pub struct CommitHandle {
    shared: Arc<CommitShared>,
    workers: Vec<WorkerHandle>,
}

impl CommitHandle {
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

        let mut page_diffs = Vec::new();

        let mut path_proof_offset = 0;
        let mut witnessed_start = 0;

        for output in self.workers.into_iter().map(|w| {
            w.output_rx
                .recv()
                .expect("couldn't await output from worker thread. panicked?")
        }) {
            if let Some(root) = output.root {
                assert!(new_root.is_none());
                new_root = Some(root);
            }

            page_diffs.push(output.page_diffs);

            // if the Commit worker collected the witnessed paths
            // then we need to aggregate them
            if let Some(witnessed_paths) = output.witnessed_paths {
                // UNWRAP: the same `CommitShared` object is used to decide whether
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

        // UNWRAP: one thread always produces the root.
        Output {
            root: new_root.unwrap(),
            page_diffs,
            witness: maybe_witness,
            witnessed_operations: maybe_witnessed_ops,
        }
    }
}

/// The output of a commit operation.
pub struct Output {
    /// The new root.
    pub root: Node,
    /// All page-diffs from all worker threads. The covered sets of pages are disjoint.
    pub page_diffs: Vec<Vec<(PageId, PageDiff)>>,
    /// Optional witness
    pub witness: Option<Witness>,
    /// Optional list of all witnessed operations.
    pub witnessed_operations: Option<WitnessedOperations>,
}

enum ToWorker {
    // Prepare to commit. Drop any existing read-pass.
    Prepare,
    // Shard provided. Load pages and commit upwards.
    Commit(CommitCommand),
}

struct CommitCommand {
    shared: Arc<CommitShared>,
    write_pass: WritePassEnvelope<ShardIndex>,
}

struct WarmUpCommand {
    key_path: KeyPath,
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
    page_diffs: Vec<(PageId, PageDiff)>,
}

impl WorkerOutput {
    fn new(witness: bool) -> Self {
        WorkerOutput {
            root: None,
            witnessed_paths: if witness { Some(Vec::new()) } else { None },
            page_diffs: Vec::new(),
        }
    }
}

// Shared data used in committing.
struct CommitShared {
    read_write: Vec<(KeyPath, KeyReadWrite)>,
    // nodes needing to be written to pages above a shard.
    root_page_pending: Mutex<Vec<(TriePosition, RootPagePending)>>,
    witness: bool,
}

impl CommitShared {
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

struct WorkerHandle {
    output_rx: Receiver<WorkerOutput>,
    commit_tx: Sender<ToWorker>,
    warmup_tx: Sender<WarmUpCommand>,
}

fn spawn_worker<H: NodeHasher>(worker_tp: &ThreadPool, params: worker::Params) -> WorkerHandle {
    let (commit_tx, commit_rx) = channel::unbounded();
    let (warmup_tx, warmup_rx) = channel::unbounded();
    let (output_tx, output_rx) = channel::bounded(1);

    let worker_comms = worker::Comms {
        commit_rx,
        warmup_rx,
        output_tx,
    };

    worker_tp.execute(move || worker::run::<H>(worker_comms, params));

    WorkerHandle {
        commit_tx,
        warmup_tx,
        output_rx,
    }
}
