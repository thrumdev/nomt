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
    page_id::{PageId, ROOT_PAGE_ID},
    proof::PathProof,
    trie::{InternalData, NodeHasher, NodeHasherExt, ValueHash, TERMINATOR},
    trie_pos::TriePosition,
};
use overlay::{LiveOverlay, OverlayMarker};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::{DirtyPage, Store, ValueTransaction};

// CARGO HACK: silence lint; this is used in integration tests

pub use nomt_core::proof;
pub use nomt_core::trie::{KeyPath, LeafData, Node, NodeKind, NodePreimage};
pub use options::{Options, PanicOnSyncMode};
pub use overlay::{InvalidAncestors, Overlay};
pub use store::HashTableUtilization;

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
    root: Node,
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
        let root = compute_root_node::<T>(&page_cache, &store);
        Ok(Self {
            merkle_update_pool: UpdatePool::new(o.commit_concurrency, o.warm_up),
            page_cache,
            page_pool,
            store,
            shared: Arc::new(Mutex::new(Shared {
                root,
                last_commit_marker: None,
            })),
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

    /// Returns the current sync sequence number.
    #[doc(hidden)]
    pub fn sync_seqn(&self) -> u32 {
        self.store.sync_seqn()
    }

    /// Creates a new [`Session`] object, that serves a purpose of capturing the reads and writes
    /// performed by the application, updating the trie and creating a [`Witness`], allowing to
    /// re-execute the same operations without having access to the full trie.
    ///
    /// Only a single session may be created at a time. Creating a new session without dropping or
    /// committing an existing open session will lead to a panic.
    pub fn begin_session(&self) -> Session {
        // UNWRAP: empty ancestors are always valid.
        self.begin_session_inner(/* allow_rollback */ true, None)
            .unwrap()
    }

    /// Creates a new [`Session`] object on top of a series of in-memory overlays.
    ///
    /// This will fail if the first overlay was not created on top of the previous overlays.
    /// Providing an empty iterator will give the same result as `begin_session`.
    ///
    /// It is the user's responsibility to ensure that all uncommitted overlays are provided.
    /// Failing to provide all uncommitted overlays or providing overlays which do not form a
    /// sequence will lead to an error.
    pub fn begin_session_with_overlay<'a>(
        &self,
        overlays: impl IntoIterator<Item = &'a Overlay>,
    ) -> Result<Session, InvalidAncestors> {
        self.begin_session_inner(/* allow_rollback */ true, overlays)
    }

    fn begin_session_inner<'a>(
        &self,
        allow_rollback: bool,
        overlays: impl IntoIterator<Item = &'a Overlay>,
    ) -> Result<Session, InvalidAncestors> {
        let live_overlay = LiveOverlay::new(overlays)?;

        let prev = self
            .session_cnt
            .swap(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(prev, 0, "only one session could be active at a time");
        let store = self.store.clone();
        let rollback_delta = if allow_rollback {
            self.store
                .rollback()
                .map(|r| r.delta_builder(&store, &live_overlay))
        } else {
            None
        };
        Ok(Session {
            store,
            merkle_updater: Some(self.merkle_update_pool.begin::<T>(
                self.page_cache.clone(),
                self.page_pool.clone(),
                self.store.clone(),
                live_overlay.clone(),
                live_overlay.parent_root().unwrap_or_else(|| self.root()),
            )),
            session_cnt: self.session_cnt.clone(),
            metrics: self.metrics.clone(),
            rollback_delta,
            overlay: Some(live_overlay),
        })
    }

    /// Update the merkle tree and commit the changes to an in-memory [`Overlay`].
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
    pub fn update(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<Overlay> {
        let (value_tx, merkle_update, rollback_delta) = self.update_inner(
            &mut session,
            actuals,
            /* witness */ false,
            /* into_overlay */ true,
        )?;
        let updated_pages = merkle_update.updated_pages.into_frozen_iter().collect();

        let values = value_tx.into_iter().collect();

        // UNWRAP: overlay is always some during session lifecycle.
        Ok(session.overlay.take().unwrap().finish(
            merkle_update.root,
            updated_pages,
            values,
            rollback_delta,
        ))
    }

    /// Update the merkle tree and commit the changes to an in-memory [`Overlay`] and create a
    /// proof of all updated values.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
    pub fn update_and_prove(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<(Overlay, Witness, WitnessedOperations)> {
        let (value_tx, merkle_update, rollback_delta) = self.update_inner(
            &mut session,
            actuals,
            /* witness */ true,
            /* into_overlay */ true,
        )?;
        let updated_pages = merkle_update.updated_pages.into_frozen_iter().collect();

        let values = value_tx.into_iter().collect();

        // UNWRAP: overlay is always some during session lifecycle.
        let overlay = session.overlay.take().unwrap().finish(
            merkle_update.root,
            updated_pages,
            values,
            rollback_delta,
        );

        // UNWRAP: witness specified as true.
        let witness = merkle_update.witness.unwrap();
        let witnessed_ops = merkle_update.witnessed_operations.unwrap();
        Ok((overlay, witness, witnessed_ops))
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

        self.commit_inner(root, page_changes, values, rollback_delta, Some(marker))
    }

    /// Update the merkle tree and commit the changes to the underlying database.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
    pub fn update_and_commit(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<Node> {
        let (value_tx, merkle_update, rollback_delta) = self.update_inner(
            &mut session,
            actuals,
            /* witness */ false,
            /* into_overlay */ false,
        )?;
        self.commit_inner(
            merkle_update.root,
            merkle_update.updated_pages.into_frozen_iter(),
            value_tx.into_iter(),
            rollback_delta,
            None,
        )?;
        Ok(merkle_update.root)
    }

    /// Commit the transaction and create a proof for the given session. Also, returns the new root.
    ///
    /// The actuals are a list of key paths and the corresponding read/write operations. The list
    /// must be sorted by the key paths in ascending order. The key paths must be unique.
    pub fn update_commit_and_prove(
        &self,
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<(Node, Witness, WitnessedOperations)> {
        let (value_tx, merkle_update, rollback_delta) = self.update_inner(
            &mut session,
            actuals,
            /* witness */ true,
            /* into_overlay */ false,
        )?;

        // UNWRAP: witness specified as true.
        let witness = merkle_update.witness.unwrap();
        let witnessed_ops = merkle_update.witnessed_operations.unwrap();
        self.commit_inner(
            merkle_update.root,
            merkle_update.updated_pages.into_frozen_iter(),
            value_tx.into_iter(),
            rollback_delta,
            None,
        )?;
        Ok((merkle_update.root, witness, witnessed_ops))
    }

    // Perform updates and merklization, but just into memory. If `witness` is true, the returned
    // witness fields will be `Some`
    fn update_inner(
        &self,
        session: &mut Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
        witness: bool,
        into_overlay: bool,
    ) -> anyhow::Result<(ValueTransaction, merkle::Output, Option<rollback::Delta>)> {
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
            .take()
            .unwrap()
            .update_and_prove::<T>(compact_actuals, witness, into_overlay);

        let mut tx = self.store.new_value_tx();
        for (path, read_write) in actuals {
            if let KeyReadWrite::Write(value) | KeyReadWrite::ReadThenWrite(_, value) = read_write {
                tx.write_value::<T>(path, value);
            }
        }

        let merkle_update = merkle_update_handle.join();
        Ok((tx, merkle_update, rollback_delta))
    }

    // Commit the transaction to disk.
    fn commit_inner(
        &self,
        new_root: Node,
        updated_pages: impl IntoIterator<Item = (PageId, DirtyPage)> + Send + 'static,
        value_tx: impl IntoIterator<Item = (beatree::Key, beatree::ValueChange)> + Send + 'static,
        rollback_delta: Option<rollback::Delta>,
        overlay_marker: Option<OverlayMarker>,
    ) -> anyhow::Result<()> {
        {
            let mut shared = self.shared.lock();
            shared.root = new_root;
            shared.last_commit_marker = overlay_marker;
        }

        if let Some(rollback_delta) = rollback_delta {
            // UNWRAP: if rollback_delta is `Some`, then rollback must be also `Some`.
            let rollback = self.store.rollback().unwrap();
            rollback.commit(rollback_delta)?;
        }

        self.store
            .commit(value_tx, self.page_cache.clone(), updated_pages)?;

        Ok(())
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
        let mut sess = self
            .begin_session_inner(/* allow_rollback */ false, None)
            .unwrap();

        // Convert the traceback into a series of write commands.
        let mut actuals = Vec::new();
        for (key, value) in traceback {
            sess.warm_up(key);
            let value = KeyReadWrite::Write(value);
            actuals.push((key, value));
        }

        let (value_tx, merkle_update, rollback_delta) = self.update_inner(
            &mut sess, actuals, /* witness */ false, /* into_overlay */ false,
        )?;
        self.commit_inner(
            merkle_update.root,
            merkle_update.updated_pages.into_frozen_iter(),
            value_tx.into_iter(),
            rollback_delta,
            None,
        )?;

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

/// A session presents a way of interaction with the trie.
///
/// The session enables the application to perform reads and prepare writes.
///
/// When the session is finished, the application can confirm the changes by calling
/// [`Nomt::update`] or others and create a [`Witness`] that can be used to prove the correctness
/// of replaying the same operations.
pub struct Session {
    store: Store,
    merkle_updater: Option<Updater>, // always `Some` during lifecycle.
    session_cnt: Arc<AtomicUsize>,
    metrics: Metrics,
    rollback_delta: Option<rollback::ReverseDeltaBuilder>,
    overlay: Option<LiveOverlay>, // always `Some` during lifecycle.
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
        if let Some(value_change) = self
            .overlay
            .as_ref()
            .and_then(|overlay| overlay.value(&path))
        {
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
    ///    [`Nomt::update`]) call.
    ///
    ///    It's safe to call this function for keys that may not ultimately be written, and keys
    ///    not marked here but included in the final set will still be preserved.
    /// 3. While this function helps optimize I/O, it's not strictly necessary for correctness.
    ///    The commit process will ensure all required prior values are preserved.
    /// 4. If the path is given to [`Nomt::update`] (and others) with the `ReadThenWrite` operation,
    ///    calling this function is not needed as the prior value will be taken from there.
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
