#![warn(missing_docs)]

//! A Nearly-Optimal Merkle Trie Database.

use bitvec::prelude::*;
use io::PagePool;
use metrics::{Metric, Metrics};
use std::{
    mem,
    sync::{atomic::AtomicUsize, Arc},
};

use merkle::{UpdatePool, Updater};
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
pub use nomt_core::trie::{KeyPath, LeafData, Node, NodePreimage};
pub use options::Options;

// beatree module needs to be exposed to be benchmarked
#[cfg(feature = "benchmarks")]
#[allow(missing_docs)]
pub mod beatree;
#[cfg(not(feature = "benchmarks"))]
mod beatree;

mod bitbox;
mod merkle;
mod metrics;
mod options;
mod page_cache;
mod page_diff;
mod page_region;
mod rollback;
mod rw_pass_cell;
mod seglog;
mod store;
mod sys;

mod io;

const MAX_COMMIT_CONCURRENCY: usize = 64;

/// A full value stored within the trie.
pub type Value = Vec<u8>;

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
    pub fn last_value(&self) -> Option<&[u8]> {
        match self {
            KeyReadWrite::Read(v) | KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) => {
                v.as_deref()
            }
        }
    }

    /// Returns true if the key was written to.
    pub fn is_write(&self) -> bool {
        matches!(
            self,
            KeyReadWrite::Write(_) | KeyReadWrite::ReadThenWrite(_, _)
        )
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

    /// Updates the state of the given slot.
    ///
    /// If the slot was written, it becomes read-then-write.
    pub fn read(&mut self, read_value: Option<Value>) {
        match *self {
            KeyReadWrite::Read(_) | KeyReadWrite::ReadThenWrite(_, _) => {}
            KeyReadWrite::Write(ref mut value) => {
                *self = KeyReadWrite::ReadThenWrite(read_value, mem::take(value));
            }
        }
    }

    fn to_compact<T: HashAlgorithm>(&self) -> crate::merkle::KeyReadWrite {
        let hash = |v: &Value| T::hash_value(v);
        match self {
            KeyReadWrite::Read(_) => crate::merkle::KeyReadWrite::Read,
            KeyReadWrite::Write(val) => crate::merkle::KeyReadWrite::Write(val.as_ref().map(hash)),
            KeyReadWrite::ReadThenWrite(_, val) => {
                crate::merkle::KeyReadWrite::ReadThenWrite(val.as_ref().map(hash))
            }
        }
    }
}

/// An instance of the Nearly-Optimal Merkle Trie Database.
pub struct Nomt<T: HashAlgorithm> {
    merkle_update_pool: UpdatePool,
    /// The handle to the page cache.
    page_cache: PageCache,
    page_pool: PagePool,
    store: Store,
    shared: Arc<Mutex<Shared>>,
    /// The number of active sessions. Expected to be either 0 or 1.
    session_cnt: Arc<AtomicUsize>,
    metrics: Metrics,
    _marker: std::marker::PhantomData<T>,
}

impl<T: HashAlgorithm> Nomt<T> {
    /// Open the database with the given options.
    pub fn open(mut o: Options) -> anyhow::Result<Self> {
        if o.commit_concurrency == 0 {
            anyhow::bail!("commit concurrency must be greater than zero".to_string());
        }

        if o.commit_concurrency > MAX_COMMIT_CONCURRENCY {
            o.commit_concurrency = MAX_COMMIT_CONCURRENCY;
        }

        let metrics = Metrics::new(o.metrics);

        let page_pool = PagePool::new();
        let store = Store::open(&o, page_pool.clone())?;
        let root_page = store.load_page(ROOT_PAGE_ID)?;
        let page_cache = PageCache::new(root_page, &o, metrics.clone());
        let root = compute_root_node::<T>(&page_cache);
        Ok(Self {
            merkle_update_pool: UpdatePool::new(o.commit_concurrency, o.warm_up),
            page_cache,
            page_pool,
            store,
            shared: Arc::new(Mutex::new(Shared { root })),
            session_cnt: Arc::new(AtomicUsize::new(0)),
            metrics,
            _marker: std::marker::PhantomData,
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

    /// Returns the value stored under the given key.
    ///
    /// Returns `None` if the value is not stored under the given key. Fails only if I/O fails.
    ///
    /// This is used for testing for now.
    #[doc(hidden)]
    pub fn read(&self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        self.store.load_value(path)
    }

    /// Creates a new [`Session`] object, that serves a purpose of capturing the reads and writes
    /// performed by the application, updating the trie and creating a [`Witness`], allowing to
    /// re-execute the same operations without having access to the full trie.
    ///
    /// Only a single session may be created at a time. Creating a new session without dropping or
    /// committing an existing open session will lead to a panic.
    pub fn begin_session(&self) -> Session {
        self.begin_session_inner(/* allow_rollback */ true)
    }

    fn begin_session_inner(&self, allow_rollback: bool) -> Session {
        let prev = self
            .session_cnt
            .swap(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(prev, 0, "only one session could be active at a time");
        let store = self.store.clone();
        let rollback_delta = if allow_rollback {
            self.store.rollback().map(|r| r.delta_builder())
        } else {
            None
        };
        Session {
            store,
            merkle_updater: Some(self.merkle_update_pool.begin::<T>(
                self.page_cache.clone(),
                self.page_pool.clone(),
                self.store.clone(),
                self.root(),
            )),
            session_cnt: self.session_cnt.clone(),
            metrics: self.metrics.clone(),
            rollback_delta,
        }
    }

    /// Commit the transaction and returns the new root.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
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
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
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
        if cfg!(debug_assertions) {
            // Check that the actuals are sorted by key path.
            for i in 1..actuals.len() {
                assert!(
                    actuals[i].0 > actuals[i - 1].0,
                    "actuals are not sorted at index {}",
                    i
                );
            }
        }
        if let Some(delta_builder) = session.rollback_delta.take() {
            // UNWRAP: if rollback_delta is `Some``, then rollback must be also `Some`.
            let rollback = self.store.rollback().unwrap();
            rollback.commit(self.store.clone(), &actuals, delta_builder)?;
        }

        let mut compact_actuals = Vec::with_capacity(actuals.len());
        for (path, read_write) in &actuals {
            compact_actuals.push((path.clone(), read_write.to_compact::<T>()));
        }

        // UNWRAP: merkle_updater always `Some` during lifecycle.
        let merkle_update_handle = session
            .merkle_updater
            .take()
            .unwrap()
            .update_and_prove::<T>(compact_actuals, witness);

        let mut tx = self.store.new_value_tx();
        for (path, read_write) in actuals {
            if let KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) = read_write {
                tx.write_value::<T>(path, value);
            }
        }

        let merkle_update = merkle_update_handle.join();

        let new_root = merkle_update.root;
        self.shared.lock().root = new_root;
        self.store
            .commit(tx, self.page_cache.clone(), merkle_update.page_diffs)?;

        Ok((
            new_root,
            merkle_update.witness,
            merkle_update.witnessed_operations,
        ))
    }

    /// Perform a rollback of the last `n` commits.
    ///
    /// This function assumes no sessions are active and panics otherwise.
    pub fn rollback(&self, n: usize) -> anyhow::Result<()> {
        if n == 0 {
            return Ok(());
        }
        let Some(rollback) = self.store.rollback() else {
            anyhow::bail!("rollback: not enabled");
        };
        let Some(traceback) = rollback.truncate(n)? else {
            anyhow::bail!("rollback: not enough logged for rolling back");
        };

        // Begin a new session. We do not allow rollback for this operation because that would
        // interfere with the rollback log: if another rollback were to be issued, it must rollback
        // the changes in the rollback log and not the changes performed by the current rollback.
        let sess = self.begin_session_inner(/* allow_rollback */ false);

        // Convert the traceback into a series of write commands.
        let mut actuals = Vec::new();
        for (key, value) in traceback {
            sess.warm_up(key);
            let value = KeyReadWrite::Write(value);
            actuals.push((key, value));
        }

        self.commit_inner(sess, actuals, /* witness */ false)?;

        Ok(())
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
    merkle_updater: Option<Updater>, // always `Some` during lifecycle.
    session_cnt: Arc<AtomicUsize>,
    metrics: Metrics,
    rollback_delta: Option<rollback::ReverseDeltaBuilder>,
}

impl Session {
    /// Signal to the backend to warm up the merkle paths and b-tree pages for a key, so they are
    /// ready by the time you commit the session.
    ///
    /// This should be called for every logical write within the session, as well as every logical
    /// read if you expect to generate a merkle proof for the session. If you do not expect to
    /// prove this session, you can skip calling this for reads, but still need to warm up logical
    /// writes.
    ///
    /// The purpose of warming up is to move I/O out of the critical path of committing a
    /// session to maximize throughput.
    /// There is no correctness issue with doing too many warm-ups, but there is a cost for I/O.
    pub fn warm_up(&self, path: KeyPath) {
        // UNWRAP: merkle_updater always `Some` during lifecycle.
        self.merkle_updater.as_ref().unwrap().warm_up(path);
    }

    /// Synchronously read the value stored under the given key.
    ///
    /// Returns `None` if the value is not stored under the given key. Fails only if I/O fails.
    pub fn read(&self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        let _maybe_guard = self.metrics.record(Metric::ValueFetchTime);
        self.store.load_value(path)
    }

    /// Signals that the given key is going to be written to. Relevant only if rollback is enabled.
    ///
    /// This function initiates an I/O load operation to fetch and preserve the prior value of the key.
    /// It serves as a hint to reduce I/O operations during the commit process by pre-loading values
    /// that are likely to be needed.
    ///
    /// Important considerations:
    /// 1. This function does not perform deduplication. Calling it multiple times for the same key
    ///    will result in multiple load operations, which can be wasteful.
    /// 2. The definitive set of values to be committed is determined by the [`Nomt::commit`] call.
    ///    It's safe to call this function for keys that may not ultimately be written, and keys
    ///    not marked here but included in the final set will still be preserved.
    /// 3. While this function helps optimize I/O, it's not strictly necessary for correctness.
    ///    The commit process will ensure all required prior values are preserved.
    /// 4. If the path is given to [`Nomt::commit`] with the `ReadThenWrite` operation, calling
    ///    this function is not needed as the prior value will be taken from there.
    ///
    /// For best performance, call this function once for each key you expect to write during the
    /// session, except for those that will be part of a `ReadThenWrite` operation. The earlier
    /// this call is issued, the better for efficiency.
    pub fn preserve_prior_value(&self, path: KeyPath) {
        if let Some(rollback) = &self.rollback_delta {
            rollback.tentative_preserve_prior(self.store.clone(), path);
        }
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

/// A hasher for arbitrary-length values.
pub trait ValueHasher {
    /// Hash an arbitrary-length value.
    fn hash_value(value: &[u8]) -> [u8; 32];
}

/// A hash algorithm that uses Blake3 for both nodes and values.
pub struct Blake3Hasher;

impl NodeHasher for Blake3Hasher {
    fn hash_node(data: &NodePreimage) -> [u8; 32] {
        blake3::hash(data).into()
    }
}

impl ValueHasher for Blake3Hasher {
    fn hash_value(data: &[u8]) -> [u8; 32] {
        blake3::hash(data).into()
    }
}

/// A marker trait for hash functions usable with NOMT. The type must support both hashing nodes as
/// well as values.
pub trait HashAlgorithm: ValueHasher + NodeHasher {}

impl<T: ValueHasher + NodeHasher> HashAlgorithm for T {}

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

#[cfg(test)]
mod tests {

    #[test]
    fn session_is_sync() {
        fn is_sync<T: Sync>() {}

        is_sync::<crate::Session>();
    }
}
