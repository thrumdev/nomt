#![warn(missing_docs)]

//! A Nearly-Optimal Merkle Trie Database.

use bitvec::prelude::*;
use io::PagePool;
use metrics::{Metric, Metrics};
use std::{mem, sync::Arc};

use merkle::{UpdatePool, Updater};
use nomt_core::{
    page_id::ROOT_PAGE_ID,
    proof::PathProof,
    trie::{self, InternalData, NodeHasher, NodeHasherExt, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
};
use overlay::{LiveOverlay, OverlayMarker};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::{Store, ValueTransaction};

// CARGO HACK: silence lint; this is used in integration tests

pub use nomt_core::proof;
pub use nomt_core::trie::{KeyPath, LeafData, Node, NodeKind, NodePreimage};
pub use options::{Options, PanicOnSyncMode};
pub use overlay::{InvalidAncestors, Overlay};
pub use store::HashTableUtilization;

// beatree module needs to be exposed to be benchmarked and fuzzed
#[cfg(any(feature = "benchmarks", feature = "fuzz"))]
#[allow(missing_docs)]
pub mod beatree;
#[cfg(not(any(feature = "benchmarks", feature = "fuzz")))]
mod beatree;

mod bitbox;
mod merkle;
mod metrics;
mod options;
mod overlay;
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
    root: Root,
    /// The marker of the last committed overlay. `None` if the last commit was not an overlay.
    last_commit_marker: Option<OverlayMarker>,
}

/// A witness that can be used to prove the correctness of state trie retrievals and updates.
///
/// Expected to be serializable.
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
pub struct Witness {
    /// Various paths down the trie used as part of this witness.
    pub path_proofs: Vec<WitnessedPath>,
    /// The operations witnessed by the paths.
    pub operations: WitnessedOperations,
}

/// Operations provable by a corresponding witness.
// TODO: the format of this structure depends heavily on how it'd be used with the path proofs.
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
pub struct WitnessedOperations {
    /// Read operations.
    pub reads: Vec<WitnessedRead>,
    /// Write operations.
    pub writes: Vec<WitnessedWrite>,
}

/// A path observed in the witness.
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
pub struct WitnessedPath {
    /// Proof of a query path along the trie.
    pub inner: PathProof,
    /// The query path itself.
    pub path: TriePosition,
}

/// A witness of a read value.
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
pub struct WitnessedRead {
    /// The key of the read value.
    pub key: KeyPath,
    /// The hash of the value witnessed. None means no value.
    pub value: Option<ValueHash>,
    /// The index of the path in the corresponding witness.
    pub path_index: usize,
}

/// A witness of a write operation.
#[cfg_attr(
    feature = "borsh",
    derive(borsh::BorshDeserialize, borsh::BorshSerialize)
)]
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

/// The root of the Merkle Trie.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Root([u8; 32]);

impl Root {
    /// Whether the root represents an empty trie.
    pub fn is_empty(&self) -> bool {
        self.0 == trie::TERMINATOR
    }

    /// Get the underlying bytes of the root.
    pub fn into_inner(self) -> [u8; 32] {
        self.0
    }
}

impl std::fmt::Display for Root {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for byte in &self.0[0..4] {
            write!(f, "{:02x}", byte)?;
        }

        write!(f, "...")?;

        for byte in &self.0[28..32] {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Root {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Root(")?;
        for byte in &self.0 {
            write!(f, "{:02x}", byte)?;
        }
        write!(f, ")")?;
        Ok(())
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
        let root = compute_root_node::<T>(&page_cache, &store);
        Ok(Self {
            merkle_update_pool: UpdatePool::new(o.commit_concurrency, o.warm_up),
            page_cache,
            page_pool,
            store,
            shared: Arc::new(Mutex::new(Shared {
                root: Root(root),
                last_commit_marker: None,
            })),
            metrics,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns a recent root of the trie.
    pub fn root(&self) -> Root {
        self.shared.lock().root.clone()
    }

    /// Returns true if the trie has no items in it.
    pub fn is_empty(&self) -> bool {
        self.root().is_empty()
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

    /// Returns the current sync sequence number.
    #[doc(hidden)]
    pub fn sync_seqn(&self) -> u32 {
        self.store.sync_seqn()
    }

    /// Create a new [`Session`] object with the given parameters.
    ///
    /// The [`Session`] is a read-only handle on the database and is used to create a changeset to
    /// be applied to the database. Sessions provide read interfaces and additionally coordinate
    /// work such as proving and rollback preimages which make committing more efficient.
    ///
    /// There may be multiple sessions live, though the existence of an outstanding session will
    /// prevent writes to the database. Sessions are the main way of reading to the database,
    /// and permit a changeset to be committed either directly to the database or into an
    /// in-memory [`Overlay`].
    pub fn begin_session(&self, params: SessionParams) -> Session {
        let live_overlay = params.overlay;

        let store = self.store.clone();
        let rollback_delta = if params.record_rollback_delta {
            self.store
                .rollback()
                .map(|r| r.delta_builder(&store, &live_overlay))
        } else {
            None
        };
        Session {
            store,
            merkle_updater: self.merkle_update_pool.begin::<T>(
                self.page_cache.clone(),
                self.page_pool.clone(),
                self.store.clone(),
                live_overlay.clone(),
                live_overlay
                    .parent_root()
                    .unwrap_or_else(|| self.root().into_inner()),
            ),
            metrics: self.metrics.clone(),
            rollback_delta,
            overlay: live_overlay,
            witness_mode: params.witness,
        }
    }

    /// Finish the provided [`Session`]. Provide the actual reads and writes (in sorted order)
    /// that are to be considered within the finished session.
    pub fn finish_session(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> FinishedSession {
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
        let rollback_delta = session
            .rollback_delta
            .take()
            .map(|delta_builder| delta_builder.finalize(&actuals));

        let mut compact_actuals = Vec::with_capacity(actuals.len());
        for (path, read_write) in &actuals {
            compact_actuals.push((path.clone(), read_write.to_compact::<T>()));
        }

        // UNWRAP: merkle_updater always `Some` during lifecycle.
        let merkle_update_handle = session
            .merkle_updater
            .update_and_prove::<T>(compact_actuals, session.witness_mode.0);

        let mut tx = self.store.new_value_tx();
        for (path, read_write) in actuals {
            if let KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) = read_write {
                tx.write_value::<T>(path, value);
            }
        }

        let merkle_output = merkle_update_handle.join();
        FinishedSession {
            value_transaction: tx,
            merkle_output,
            rollback_delta,
            // UNWRAP: session overlay is always `Some` during lifecycle.
            parent_overlay: session.overlay,
        }
    }

    /// Commit a finished session to disk directly.
    pub fn commit_finished(&self, session: FinishedSession) -> Result<(), anyhow::Error> {
        // TODO: do a prev_root check or something to ensure continuity?

        {
            let mut shared = self.shared.lock();
            shared.root = Root(session.merkle_output.root);
            shared.last_commit_marker = None;
        }

        if let Some(rollback_delta) = session.rollback_delta {
            // UNWRAP: if rollback_delta is `Some`, then rollback must be also `Some`.
            let rollback = self.store.rollback().unwrap();
            rollback.commit(rollback_delta)?;
        }

        self.store.commit(
            session.value_transaction.into_iter(),
            self.page_cache.clone(),
            session
                .merkle_output
                .updated_pages
                .into_frozen_iter(/* into_overlay */ false),
        )
    }

    /// Commit the changes from this overlay to the underlying database.
    ///
    /// This assumes no sessions are active and will panic otherwise.
    ///
    /// This call will fail if the overlay has a parent which has not been committed.
    pub fn commit_overlay(&self, overlay: Overlay) -> anyhow::Result<()> {
        if !overlay.parent_matches_marker(self.shared.lock().last_commit_marker.as_ref()) {
            anyhow::bail!("Overlay parent not committed");
        }

        let root = overlay.root();
        let page_changes: Vec<_> = overlay
            .page_changes()
            .into_iter()
            .map(|(page_id, dirty_page)| (page_id.clone(), dirty_page.clone()))
            .collect();
        let values: Vec<_> = overlay
            .value_changes()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let rollback_delta = overlay.rollback_delta().map(|delta| delta.clone());

        let marker = overlay.commit();

        {
            let mut shared = self.shared.lock();
            shared.root = root;
            shared.last_commit_marker = Some(marker);
        }

        if let Some(rollback_delta) = rollback_delta {
            // UNWRAP: if rollback_delta is `Some`, then rollback must be also `Some`.
            let rollback = self.store.rollback().unwrap();
            rollback.commit(rollback_delta)?;
        }

        self.store
            .commit(values, self.page_cache.clone(), page_changes)
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
        // UNWRAP: `None` ancestors are always valid.
        let mut session_params = SessionParams::default();
        session_params.record_rollback_delta = false;
        let sess = self.begin_session(session_params);

        // Convert the traceback into a series of write commands.
        let mut actuals = Vec::new();
        for (key, value) in traceback {
            sess.warm_up(key);
            let value = KeyReadWrite::Write(value);
            actuals.push((key, value));
        }

        let finished = self.finish_session(sess, actuals);
        self.commit_finished(finished)?;

        Ok(())
    }

    /// Return Nomt's metrics.
    /// To collect them, they need to be activated at [`Nomt`] creation
    #[doc(hidden)]
    pub fn metrics(&self) -> Metrics {
        self.metrics.clone()
    }

    /// Get the hash-table space utilization.
    pub fn hash_table_utilization(&self) -> HashTableUtilization {
        self.store.hash_table_utilization()
    }
}

/// A configuration type used to inform NOMT whether to generate witnesses of accessed data.
pub struct WitnessMode(bool);

impl WitnessMode {
    /// Witness all reads and writes to the trie.
    pub fn read_write() -> Self {
        WitnessMode(true)
    }

    /// Do not generate a witness.
    pub fn disabled() -> Self {
        WitnessMode(false)
    }
}

/// Parameters for instantiating a session.
pub struct SessionParams {
    record_rollback_delta: bool,
    witness: WitnessMode,
    overlay: LiveOverlay,
}

impl Default for SessionParams {
    fn default() -> Self {
        SessionParams {
            record_rollback_delta: true,
            witness: WitnessMode::disabled(),
            // UNWRAP: empty live overlay always valid.
            overlay: LiveOverlay::new(None).unwrap(),
        }
    }
}

impl SessionParams {
    /// Whether to generate a witness of the read and written keys. Default: disabled
    ///
    /// If `WitnessMode::read_write()` is provided, then when this session has concluded it will be
    /// possible to use [`FinishedSession::take_witness`] to get the recorded witness.
    pub fn witness_mode(mut self, witness: WitnessMode) -> Self {
        self.witness = witness;
        self
    }

    /// Use a set of live overlays (ancestors, in descending order) as a parent. Default: None
    ///
    /// Errors are returned if the set of ancestor overlays provided are not _sound_ or _complete_.
    ///
    /// An error is returned if the overlays do not represent a chain, starting from the first.
    /// i.e. the second overlay must be an ancestor of the first, and the third must be an ancestor
    /// of the second, and so on. This can be thought of as soundness.
    ///
    /// An error is returned if the final overlay's parent has not yet been committed. The complete
    /// set of all uncommitted ancestors must be provided.
    pub fn overlay<'a>(
        mut self,
        ancestors: impl IntoIterator<Item = &'a Overlay>,
    ) -> Result<Self, InvalidAncestors> {
        self.overlay = LiveOverlay::new(ancestors)?;
        Ok(self)
    }
}

/// A session presents a way of interaction with the trie.
///
/// The session enables the application to perform reads and prepare writes.
///
/// When the session is finished, the application can confirm the changes by calling
/// [`Nomt::finish_session`] or others and create a [`Witness`] that can be used to prove the
/// correctness of replaying the same operations.
pub struct Session {
    store: Store,
    merkle_updater: Updater,
    metrics: Metrics,
    rollback_delta: Option<rollback::ReverseDeltaBuilder>,
    overlay: LiveOverlay,
    witness_mode: WitnessMode,
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
        self.merkle_updater.warm_up(path);
    }

    /// Synchronously read the value stored under the given key.
    ///
    /// Returns `None` if the value is not stored under the given key. Fails only if I/O fails.
    pub fn read(&self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        let _maybe_guard = self.metrics.record(Metric::ValueFetchTime);
        if let Some(value_change) = self.overlay.value(&path) {
            return Ok(value_change.as_option().map(|v| v.to_vec()));
        }
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
    /// 2. The definitive set of values to be updated is determined by the update (e.g.
    ///    [`Nomt::finish_session`]) call.
    ///
    ///    It's safe to call this function for keys that may not ultimately be written, and keys
    ///    not marked here but included in the final set will still be preserved.
    /// 3. While this function helps optimize I/O, it's not strictly necessary for correctness.
    ///    The commit process will ensure all required prior values are preserved.
    /// 4. If the path is given to [`Nomt::finish_session`] (and others) with the `ReadThenWrite`
    ///    operation, calling this function is not needed as the prior value will be taken from
    ///    there.
    ///
    /// For best performance, call this function once for each key you expect to write during the
    /// session, except for those that will be part of a `ReadThenWrite` operation. The earlier
    /// this call is issued, the better for efficiency.
    pub fn preserve_prior_value(&self, path: KeyPath) {
        if let Some(rollback) = &self.rollback_delta {
            rollback.tentative_preserve_prior(path);
        }
    }
}

/// A finished session.
///
/// This is the result of completing a session and computing the merkle root and merkle DB changes,
/// but which has not yet been applied to the underlying store.
///
/// It may either be committed directly to the database or transformed into an in-memory [`Overlay`]
/// to be used as a base for further in-memory sessions.
pub struct FinishedSession {
    value_transaction: ValueTransaction,
    merkle_output: merkle::Output,
    rollback_delta: Option<rollback::Delta>,
    parent_overlay: LiveOverlay,
}

impl FinishedSession {
    /// Get the root as-of this session.
    pub fn root(&self) -> Root {
        Root(self.merkle_output.root)
    }

    /// Take the witness, if any.
    ///
    /// If this session was configured with proving  (see [`SessionParams::witness_mode`]),
    /// this will be `Some` on the first call and `None` thereafter.
    pub fn take_witness(&mut self) -> Option<Witness> {
        self.merkle_output.witness.take()
    }

    /// Transform this into an overlay that can be queried in memory and used as the base for
    /// further in-memory [`Session`]s.
    pub fn into_overlay(self) -> Overlay {
        let updated_pages = self
            .merkle_output
            .updated_pages
            .into_frozen_iter(/* into_overlay */ true)
            .collect();
        let values = self.value_transaction.into_iter().collect();

        self.parent_overlay.finish(
            self.merkle_output.root,
            updated_pages,
            values,
            self.rollback_delta,
        )
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

fn compute_root_node<H: HashAlgorithm>(page_cache: &PageCache, store: &Store) -> Node {
    // 3 cases.
    // 1: root page is empty and beatree is empty. in this case, root is the TERMINATOR.
    // 2: root page is empty and beatree has a single item. in this case, root is a leaf.
    // 3: root is an internal node.

    if let Some((root_page, _)) = page_cache.get(ROOT_PAGE_ID) {
        let left = root_page.node(0);
        let right = root_page.node(1);

        if left != TERMINATOR || right != TERMINATOR {
            // case 3
            return H::hash_internal(&InternalData { left, right });
        }
    }

    // cases 1/2
    let read_tx = store.read_transaction();
    let mut iterator = read_tx.iterator(beatree::Key::default(), None);

    let io_handle = store.io_pool().make_handle();

    loop {
        match iterator.next() {
            None => return TERMINATOR, // case 1
            Some(beatree::iterator::IterOutput::Blocked) => {
                // UNWRAP: when blocked, needed leaf always exists.
                let leaf = match read_tx.load_leaf_async(
                    iterator.needed_leaves().next().unwrap(),
                    &io_handle,
                    0,
                ) {
                    Ok(leaf_node) => leaf_node,
                    Err(leaf_load) => {
                        // UNWRAP: `Err` indicates a request was sent.
                        let complete_io = io_handle.recv().unwrap();

                        // UNWRAP: the I/O command submitted by `load_leaf_async` is always a `Read`
                        leaf_load.finish(complete_io.command.kind.unwrap_buf())
                    }
                };

                iterator.provide_leaf(leaf);
            }
            Some(beatree::iterator::IterOutput::Item(key_path, value)) => {
                // case 2
                return H::hash_leaf(&LeafData {
                    key_path,
                    value_hash: H::hash_value(value),
                });
            }
            Some(beatree::iterator::IterOutput::OverflowItem(key_path, value_hash, _)) => {
                // case 2
                return H::hash_leaf(&LeafData {
                    key_path,
                    value_hash,
                });
            }
        }
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
