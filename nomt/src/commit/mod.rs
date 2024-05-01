//! Parallel, pipelined merkle trie commits.
//!
//! This splits the work of warming-up and performing the trie update across worker threads.

use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::Mutex;

use nomt_core::{
    page_id::{ChildPageIndex, PageId, NUM_CHILDREN, ROOT_PAGE_ID},
    trie::{self, KeyPath, Node, NodeHasher, ValueHash},
    trie_pos::TriePosition,
};

use std::collections::HashMap;
use std::sync::{Arc, Barrier};

use threadpool::ThreadPool;

use crate::{
    page_cache::{PageCache, PageDiff},
    page_region::PageRegion,
    rw_pass_cell::WritePassEnvelope,
    Witness, WitnessedOperations, WitnessedPath, WitnessedRead, WitnessedWrite,
};

mod worker;

const MAX_WORKERS: usize = 64;

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

    fn is_delete(&self) -> bool {
        match self {
            KeyReadWrite::Read => false,
            KeyReadWrite::Write(val) | KeyReadWrite::ReadThenWrite(val) => val.is_none(),
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
    /// If `num_workers` is greater than 64, 64 workers will be used.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is zero.
    pub fn new(num_workers: usize) -> Self {
        let num_workers = std::cmp::min(MAX_WORKERS, num_workers);

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
    pub fn begin<H: NodeHasher>(&self, page_cache: PageCache, root: Node) -> Committer {
        let num_workers = self.worker_tp.max_count();

        let barrier = Arc::new(Barrier::new(num_workers + 1));
        let workers: Vec<WorkerHandle> = (0..num_workers)
            .map(|_| {
                let params = worker::Params {
                    page_cache: page_cache.clone(),
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
    ///
    /// Set `delete` to true when the key path may be deleted.
    pub fn warm_up(&mut self, key_path: KeyPath, delete: bool) {
        let worker = self.worker_round_robin;
        self.worker_round_robin += 1;
        self.worker_round_robin %= self.workers.len();

        let _ = self.workers[worker]
            .warmup_tx
            .send(WarmUpCommand { key_path, delete });
    }

    /// Commit the given key-value read/write operations. Key-paths should be in sorted order
    /// and should appear at most once within the vector.
    pub fn commit(mut self, read_write: Vec<(KeyPath, KeyReadWrite)>) -> CommitHandle {
        let shared = Arc::new(CommitShared {
            read_write,
            root_page_pending: Mutex::new(Vec::with_capacity(64)),
        });

        // We apply a simple strategy that assumes keys are uniformly distributed, and give
        // each worker an approximately even number of root child pages. This scales well up to
        // 64 worker threads.
        // The first `remainder` workers get `part + 1` children and the rest get `part`.
        let part = NUM_CHILDREN / self.workers.len();
        let remainder = NUM_CHILDREN % self.workers.len();

        let mut regions = Vec::with_capacity(self.workers.len());
        for (worker_index, worker) in self.workers.iter().enumerate() {
            let _ = worker.commit_tx.send(ToWorker::Prepare);

            let (start, count) = if worker_index >= remainder {
                (part * worker_index + remainder, part)
            } else {
                (part * worker_index + worker_index, part + 1)
            };

            // UNWRAP: start / start + count are both less than the number of children.
            let start_child = ChildPageIndex::new(start as u8).unwrap();
            let end_child = ChildPageIndex::new((start + count - 1) as u8).unwrap();
            let region = PageRegion::from_page_id_descendants(ROOT_PAGE_ID, start_child, end_child);
            regions.push(region);
        }

        let write_pass = self.page_cache.new_write_pass();
        let worker_passes = write_pass.split_n(regions);

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
        let mut witness = Witness {
            path_proofs: Vec::new(),
        };
        let mut witnessed_ops = WitnessedOperations {
            reads: Vec::new(),
            writes: Vec::new(),
        };
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

            let path_proof_count = output.witnessed_paths.len();
            witness.path_proofs.reserve(output.witnessed_paths.len());
            for (path_index, (path, leaf_data, batch_size)) in
                output.witnessed_paths.into_iter().enumerate()
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

        // UNWRAP: one thread always produces the root.
        Output {
            root: new_root.unwrap(),
            page_diffs,
            witness,
            witnessed_operations: witnessed_ops,
        }
    }
}

/// The output of a commit operation.
pub struct Output {
    /// The new root.
    pub root: Node,
    /// All page-diffs from all worker threads. The covered sets of pages are disjoint.
    pub page_diffs: Vec<HashMap<PageId, PageDiff>>,
    /// The witness.
    pub witness: Witness,
    /// All witnessed operations.
    pub witnessed_operations: WitnessedOperations,
}

enum ToWorker {
    // Prepare to commit. Drop any existing read-pass.
    Prepare,
    // Shard provided. Load pages and commit upwards.
    Commit(CommitCommand),
}

struct CommitCommand {
    shared: Arc<CommitShared>,
    write_pass: WritePassEnvelope<PageRegion>,
}

struct WarmUpCommand {
    key_path: KeyPath,
    delete: bool,
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
    witnessed_paths: Vec<(WitnessedPath, Option<trie::LeafData>, usize)>,
    page_diffs: HashMap<PageId, PageDiff>,
}

impl Default for WorkerOutput {
    fn default() -> Self {
        WorkerOutput {
            root: None,
            witnessed_paths: Vec::new(),
            page_diffs: HashMap::new(),
        }
    }
}

// Shared data used in committing.
struct CommitShared {
    read_write: Vec<(KeyPath, KeyReadWrite)>,
    // nodes needing to be written to pages above a shard.
    root_page_pending: Mutex<Vec<(TriePosition, RootPagePending)>>,
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
