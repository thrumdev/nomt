#![warn(missing_docs)]

//! A Nearly-Optimal Merkle Trie Database.

use bitvec::prelude::*;
use metrics::{Metric, Metrics};
use std::{
    mem,
    rc::Rc,
    sync::{atomic::AtomicUsize, Arc},
};

use commit::{CommitPool, Committer};
use nomt_core::{
    page_id::ROOT_PAGE_ID,
    proof::PathProof,
    trie::{InternalData, NodeHasher, NodeHasherExt, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::Store;

// CARGO HACK: silence lint; this is used in integration tests

pub use nomt_core::proof;
pub use nomt_core::trie::{KeyPath, LeafData, Node};
pub use options::Options;

// beatree module needs to be exposed to be benchmarked
#[cfg(feature = "benchmarks")]
#[allow(missing_docs)]
pub mod beatree;
#[cfg(not(feature = "benchmarks"))]
mod beatree;

mod bitbox;
mod commit;
mod metrics;
mod options;
mod page_cache;
mod page_diff;
mod page_region;
mod page_walker;
mod rw_pass_cell;
mod seek;
mod store;

mod io;

const MAX_COMMIT_CONCURRENCY: usize = 64;

/// A full value stored within the trie.
pub type Value = Rc<Vec<u8>>;

struct Shared {
    /// The current root of the trie.
    root: Node,
}

/// A witness that can be used to prove the correctness of state trie retrievals and updates.
///
/// Expected to be serializable.
pub struct Witness {
    /// Various paths down the trie used as part of this witness.
    pub path_proofs: Vec<WitnessedPath>,
}

/// Operations provable by a corresponding witness.
// TODO: the format of this structure depends heavily on how it'd be used with the path proofs.
pub struct WitnessedOperations {
    /// Read operations.
    pub reads: Vec<WitnessedRead>,
    /// Write operations.
    pub writes: Vec<WitnessedWrite>,
}

/// A path observed in the witness.
pub struct WitnessedPath {
    /// Proof of a query path along the trie.
    pub inner: PathProof,
    /// The query path itself.
    pub path: TriePosition,
}

/// A witness of a read value.
pub struct WitnessedRead {
    /// The key of the read value.
    pub key: KeyPath,
    /// The hash of the value witnessed. None means no value.
    pub value: Option<ValueHash>,
    /// The index of the path in the corresponding witness.
    pub path_index: usize,
}

/// A witness of a write operation.
pub struct WitnessedWrite {
    /// The key of the written value.
    pub key: KeyPath,
    /// The hash of the written value. `None` means "delete".
    pub value: Option<ValueHash>,
    /// The index of the path in the corresponding witness.
    pub path_index: usize,
}

/// Whether a key was read, written, or both, along with old and new values.
#[derive(Debug, Clone)]
pub enum KeyReadWrite {
    /// The key was read. Contains the read value.
    Read(Option<Value>),
    /// The key was written. Contains the written value.
    Write(Option<Value>),
    /// The key was both read and written. Contains the previous value and the new value.
    ReadThenWrite(Option<Value>, Option<Value>),
}

impl KeyReadWrite {
    /// Returns the last recorded value for the slot.
    pub fn last_value(&self) -> Option<&Value> {
        match self {
            KeyReadWrite::Read(v) | KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) => {
                v.as_ref()
            }
        }
    }

    /// Updates the state of the given slot.
    ///
    /// If the slot was read, it becomes read-then-write. If it was written, the value is updated.
    pub fn write(&mut self, new_value: Option<Value>) {
        match *self {
            KeyReadWrite::Read(ref mut value) => {
                *self = KeyReadWrite::ReadThenWrite(mem::take(value), new_value);
            }
            KeyReadWrite::Write(ref mut value) => {
                *value = new_value;
            }
            KeyReadWrite::ReadThenWrite(_, ref mut value) => {
                *value = new_value;
            }
        }
    }

    fn to_compact(&self) -> crate::commit::KeyReadWrite {
        let hash = |v: &Value| *blake3::hash(v).as_bytes();
        match self {
            KeyReadWrite::Read(_) => crate::commit::KeyReadWrite::Read,
            KeyReadWrite::Write(val) => crate::commit::KeyReadWrite::Write(val.as_ref().map(hash)),
            KeyReadWrite::ReadThenWrite(_, val) => {
                crate::commit::KeyReadWrite::ReadThenWrite(val.as_ref().map(hash))
            }
        }
    }
}

/// Hash nodes with blake3.
pub struct Blake3Hasher;

impl NodeHasher for Blake3Hasher {
    fn hash_node(data: &nomt_core::trie::NodePreimage) -> [u8; 32] {
        blake3::hash(data).into()
    }
}

/// An instance of the Nearly-Optimal Merkle Trie Database.
pub struct Nomt {
    commit_pool: CommitPool,
    /// The handle to the page cache.
    page_cache: PageCache,
    store: Store,
    shared: Arc<Mutex<Shared>>,
    /// The number of active sessions. Expected to be either 0 or 1.
    session_cnt: Arc<AtomicUsize>,
    metrics: Metrics,
}

impl Nomt {
    /// Open the database with the given options.
    pub fn open(mut o: Options) -> anyhow::Result<Self> {
        if o.commit_concurrency == 0 {
            anyhow::bail!("commit concurrency must be greater than zero".to_string());
        }

        if o.commit_concurrency > MAX_COMMIT_CONCURRENCY {
            o.commit_concurrency = MAX_COMMIT_CONCURRENCY;
        }

        let metrics = Metrics::new(o.metrics);

        let store = Store::open(&o)?;
        let root_page = store.load_page(ROOT_PAGE_ID)?;
        let page_cache = PageCache::new(root_page, &o, metrics.clone());
        let root = compute_root_node::<Blake3Hasher>(&page_cache);
        Ok(Self {
            commit_pool: CommitPool::new(o.commit_concurrency),
            page_cache,
            store,
            shared: Arc::new(Mutex::new(Shared { root })),
            session_cnt: Arc::new(AtomicUsize::new(0)),
            metrics,
        })
    }

    /// Returns a recent root of the trie.
    pub fn root(&self) -> Node {
        self.shared.lock().root.clone()
    }

    /// Returns true if the trie has not been modified after the creation.
    pub fn is_empty(&self) -> bool {
        self.root() == TERMINATOR
    }

    /// Creates a new [`Session`] object, that serves a purpose of capturing the reads and writes
    /// performed by the application, updating the trie and creating a [`Witness`], allowing to
    /// re-execute the same operations without having access to the full trie.
    ///
    /// Only a single session could be created at a time.
    pub fn begin_session(&self) -> Session {
        let prev = self
            .session_cnt
            .swap(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(prev, 0, "only one session could be active at a time");
        Session {
            store: self.store.clone(),
            committer: Some(self.commit_pool.begin::<Blake3Hasher>(
                self.page_cache.clone(),
                self.store.clone(),
                self.root(),
            )),
            session_cnt: self.session_cnt.clone(),
            metrics: self.metrics.clone(),
        }
    }

    /// Commit the transaction and returns the new root.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique. For every
    /// key in the actuals, the function [`Session::tentative_read_slot`] or
    /// [`Session::tentative_write_slot`] must be called before committing.
    pub fn commit(
        &self,
        session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<Node> {
        match self.commit_inner(session, actuals, false)? {
            (node, None, None) => Ok(node),
            // UNWRAP: witness specified to false
            _ => unreachable!(),
        }
    }

    /// Commit the transaction and create a proof for the given session. Also, returns the new root.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique. For every
    /// key in the actuals, the function [`Session::tentative_read_slot`] or
    /// [`Session::tentative_write_slot`] must be called before committing.
    pub fn commit_and_prove(
        &self,
        session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<(Node, Witness, WitnessedOperations)> {
        match self.commit_inner(session, actuals, true)? {
            (node, Some(witness), Some(witnessed_ops)) => Ok((node, witness, witnessed_ops)),
            // UNWRAP: witness specified to true
            _ => unreachable!(),
        }
    }

    // Effectively commit the transaction.
    // If 'witness' is set to true, it collects the witness and
    // returns `(Node, Some(Witness), Some(WitnessedOperations))`
    // Otherwise, it solely returns the new root node, returning
    // `(Node, None, None)`
    fn commit_inner(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
        witness: bool,
    ) -> anyhow::Result<(Node, Option<Witness>, Option<WitnessedOperations>)> {
        let mut compact_actuals = Vec::with_capacity(actuals.len());
        for (path, read_write) in &actuals {
            compact_actuals.push((path.clone(), read_write.to_compact()));
        }

        // UNWRAP: committer always `Some` during lifecycle.
        let commit_handle = session
            .committer
            .take()
            .unwrap()
            .commit(compact_actuals, witness);

        let mut tx = self.store.new_tx();
        for (path, read_write) in actuals {
            if let KeyReadWrite::Write(ref value) | KeyReadWrite::ReadThenWrite(_, ref value) =
                read_write
            {
                tx.write_value(path, value.as_ref().map(|x| &x[..]));
            }
        }

        let commit = commit_handle.join();

        let new_root = commit.root;
        self.page_cache
            .commit(commit.page_diffs.into_iter().flatten(), &mut tx);
        self.shared.lock().root = new_root;
        self.store.commit(tx)?;

        Ok((new_root, commit.witness, commit.witnessed_operations))
    }

    /// Return Nomt's metrics.
    /// To collect them, they need to be activated at [`Nomt`] creation
    pub fn metrics(&self) -> Metrics {
        self.metrics.clone()
    }
}

/// A session presents a way of interaction with the trie.
///
/// During a session the application is assumed to perform a zero or more reads and writes. When
/// the session is finished, the application can [commit][`Nomt::commit_and_prove`] the changes
/// and create a [`Witness`] that can be used to prove the correctness of replaying the same
/// operations.
pub struct Session {
    store: Store,
    committer: Option<Committer>, // always `Some` during lifecycle.
    session_cnt: Arc<AtomicUsize>,
    metrics: Metrics,
}

impl Session {
    /// Synchronously read the value stored at the given key.
    ///
    /// Returns `None` if the value is not stored under the given key. Fails only if I/O fails.
    pub fn tentative_read_slot(&mut self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        // UNWRAP: committer always `Some` during lifecycle.
        self.committer.as_mut().unwrap().warm_up(path);

        let _maybe_guard = self.metrics.record(Metric::ValueFetchTime);

        let value = self.store.load_value(path)?.map(Rc::new);
        Ok(value)
    }

    /// Signals that the given key is going to be written to.
    pub fn tentative_write_slot(&mut self, path: KeyPath) {
        // UNWRAP: committer always `Some` during lifecycle.
        self.committer.as_mut().unwrap().warm_up(path);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let prev = self
            .session_cnt
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(prev, 1, "expected one active session at commit time");
    }
}

fn compute_root_node<H: NodeHasher>(page_cache: &PageCache) -> Node {
    let Some(root_page) = page_cache.get(ROOT_PAGE_ID) else {
        return TERMINATOR;
    };
    let read_pass = page_cache.new_read_pass();

    // 3 cases.
    // 1: root page is empty. in this case, root is the TERMINATOR.
    // 2: root page has top two slots filled, but _their_ children are empty. root is a leaf.
    //    this is because internal nodes and leaf nodes would have items below.
    // 3: root is an internal node.
    let is_empty = |node_index| root_page.node(&read_pass, node_index) == TERMINATOR;

    let left = root_page.node(&read_pass, 0);
    let right = root_page.node(&read_pass, 1);

    if is_empty(0) && is_empty(1) {
        TERMINATOR
    } else if (2..6usize).all(is_empty) {
        H::hash_leaf(&LeafData {
            key_path: left,
            value_hash: right,
        })
    } else {
        H::hash_internal(&InternalData { left, right })
    }
}
