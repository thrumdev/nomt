use bitvec::prelude::*;
use fxhash::FxHashMap;
use std::{
    collections::hash_map::Entry,
    mem,
    path::PathBuf,
    rc::Rc,
    sync::{atomic::AtomicUsize, Arc},
};

use cursor::PageCacheCursor;
use nomt_core::{
    proof::PathProof,
    trie::{NodeHasher, ValueHash, TERMINATOR},
};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::Store;
use threadpool::ThreadPool;

pub use nomt_core::trie::{KeyPath, LeafData, Node};

mod cursor;
mod page_cache;
mod store;

pub type Value = Rc<Vec<u8>>;

pub struct Options {
    /// The path to the directory where the trie is stored.
    pub path: PathBuf,
    /// The maximum number of concurrent page fetches.
    pub fetch_concurrency: usize,
    /// The maximum number of concurrent background page fetches.
    pub traversal_concurrency: usize,
}

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
#[derive(Default)]
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
    pub path: BitVec<u8, Msb0>,
}

/// A witness of a read value.
pub struct WitnessedRead {
    /// The key of the read value.
    pub key: KeyPath,
    /// The value itself.
    pub value: Option<Value>,
    /// The index of the path in the corresponding witness.
    pub path_index: usize,
}

/// A witness of a write operation.
pub struct WitnessedWrite {
    /// The key of the written value.
    pub key: KeyPath,
    /// The value itelf. `None` means "delete".
    pub value: Option<Value>,
    /// The index of the path in the corresponding witness.
    pub path_index: usize,
}

/// Information about an encountered terminal during the lookup of some key.
#[derive(Clone, PartialEq)]
struct TerminalInfo {
    /// A leaf, if this is a leaf. `None` indicates this is a terminator node.
    pub leaf: Option<LeafData>,
    /// The depth
    pub depth: u8,
}

/// Whether a key was read, written, or both, along with old and new values.
enum KeyReadWrite {
    /// The key was read. Contains the read value.
    Read(Option<Value>),
    /// The key was written. Contains the written value.
    Write(Option<Value>),
    /// The key was both read and written. Contains the previous value and the new value.
    ReadThenWrite(Option<Value>, Option<Value>),
}

impl KeyReadWrite {
    fn last_value(&self) -> Option<&Value> {
        match self {
            KeyReadWrite::Read(v) | KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) => {
                v.as_ref()
            }
        }
    }

    fn write(&mut self, new_value: Option<Value>) {
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
}

struct Blake3Hasher;

impl NodeHasher for Blake3Hasher {
    fn hash_node(data: &nomt_core::trie::NodePreimage) -> [u8; 32] {
        blake3::hash(data).into()
    }
}

/// An instance of the Nearly-Optimal Merkle Trie Database.
pub struct Nomt {
    /// The handle to the page cache.
    page_cache: PageCache,
    store: Store,
    warmup_tp: ThreadPool,
    shared: Arc<Mutex<Shared>>,
    /// The number of active sessions. Expected to be either 0 or 1.
    session_cnt: AtomicUsize,
}

impl Nomt {
    /// Open the database with the given options.
    pub fn open(o: Options) -> anyhow::Result<Self> {
        let store = Store::open(&o)?;
        let page_cache = PageCache::new(store.clone(), &o);
        let root = store.load_root()?;
        Ok(Self {
            page_cache,
            store,
            warmup_tp: threadpool::Builder::new()
                .num_threads(o.fetch_concurrency)
                .thread_name("nomt-warmup".to_string())
                .build(),
            shared: Arc::new(Mutex::new(Shared { root })),
            session_cnt: AtomicUsize::new(0),
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
            prev_root: self.root(),
            page_cache: self.page_cache.clone(),
            store: self.store.clone(),
            warmup_tp: self.warmup_tp.clone(),
            access: FxHashMap::default(),
            terminals: Arc::new(Mutex::new(FxHashMap::default())),
        }
    }

    /// Commit the transaction and create a proof for the given session.
    pub fn commit_and_prove(&self, session: Session) -> anyhow::Result<Witness> {
        let prev = self
            .session_cnt
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(prev, 1, "expected one active session at commit time");

        let Session {
            prev_root,
            page_cache,
            store: _,
            warmup_tp,
            access,
            terminals,
        } = session;

        // Wait for all warmup tasks to finish. That way, we can be sure that all terminal
        // information is available and that `terminals` would be the only reference.
        warmup_tp.join();

        // unwrap: since we waited for all warmup tasks to finish, there should be no other
        // references to `terminals`.
        let terminals = Arc::into_inner(terminals).unwrap();
        let terminals = terminals.into_inner();

        let mut tx = self.store.new_tx();

        let mut witness_builder = WitnessBuilder::default();
        let (access, terminals) = {
            let mut access = access.into_iter().collect::<Vec<_>>();
            let mut terminals = terminals.into_iter().collect::<Vec<_>>();
            access.sort_unstable_by_key(|(k, _)| *k);
            terminals.sort_unstable_by_key(|(k, _)| *k);
            (access, terminals)
        };

        for ((path, read_write), (t_path, terminal_info)) in access.into_iter().zip(terminals) {
            assert_eq!(path, t_path, "unexpected terminal path");

            if let KeyReadWrite::Write(ref value) | KeyReadWrite::ReadThenWrite(_, ref value) =
                read_write
            {
                let value_hash = value.as_ref().map(|v| *blake3::hash(v).as_bytes());
                let prev_value = match terminal_info.leaf.as_ref() {
                    None => None,
                    Some(l) if l.key_path == path => Some(l.value_hash),
                    Some(_) => None,
                };
                tx.write_value::<Blake3Hasher>(
                    path,
                    prev_value,
                    value_hash.zip(value.as_ref().map(|v| &v[..])),
                );
            }
            witness_builder.push(path, terminal_info.clone(), &read_write);
        }

        let (witness, root) = {
            let cache_write = page_cache.acquire_writer();
            let mut cursor =
                PageCacheCursor::at_root(prev_root, crate::cursor::Backend::Unique(cache_write));

            let (witness, _witnessed) = witness_builder.update_and_witness(&mut cursor);
            cursor.rewind();
            (witness, cursor.node())
        };

        self.shared.lock().root = root;

        self.page_cache.commit(&mut tx);
        self.store.commit(tx)?;
        Ok(witness)
    }
}

/// A session presents a way of interaction with the trie.
///
/// During a session the application is assumed to perform a zero or more reads and writes. When
/// the session is finished, the application can [commit][`Nomt::commit_and_prove`] the changes
/// and create a [`Witness`] that can be used to prove the correctness of replaying the same
/// operations.
pub struct Session {
    prev_root: Node,
    page_cache: PageCache,
    store: Store,
    warmup_tp: ThreadPool,
    access: FxHashMap<KeyPath, KeyReadWrite>,
    terminals: Arc<Mutex<FxHashMap<KeyPath, TerminalInfo>>>,
}

impl Session {
    /// Synchronously read the value stored at the given key. Returns `None` if the value is not
    /// stored under the given key. Fails only if I/O fails.
    pub fn read_slot(&mut self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        if let Some(read_write) = self.access.get(&path) {
            Ok(read_write.last_value().cloned())
        } else {
            self.warmup(path);
            let value = self.store.load_value(path)?.map(Rc::new);
            self.access.insert(path, KeyReadWrite::Read(value.clone()));
            Ok(value)
        }
    }

    /// Writes a value at the given key path. If `None` is passed, the key is deleted.
    pub fn write_slot(&mut self, path: KeyPath, new_value: Option<Value>) {
        match self.access.entry(path) {
            Entry::Occupied(mut o) => o.get_mut().write(new_value),
            Entry::Vacant(v) => {
                v.insert(KeyReadWrite::Write(new_value));
                self.warmup(path);
            }
        }
    }

    fn warmup(&self, path: KeyPath) {
        let page_cache = self.page_cache.clone();
        let store = self.store.clone();
        let terminals = self.terminals.clone();
        let root = self.prev_root;
        let f = move || {
            let mut cur =
                PageCacheCursor::at_root(root, crate::cursor::Backend::Shared(page_cache));
            cur.seek(path);
            let (_, depth) = cur.position();
            let node = cur.node();
            let terminal_info = if nomt_core::trie::is_leaf(&node) {
                let leaf = store.load_leaf(node).unwrap(); // TODO: handle error
                TerminalInfo { leaf, depth }
            } else {
                TerminalInfo { leaf: None, depth }
            };
            terminals.lock().insert(path, terminal_info);
        };
        self.warmup_tp.execute(f);
    }
}

#[derive(Default)]
struct WitnessBuilder {
    terminal: Option<(BitVec<u8, Msb0>, TerminalOps)>,
    write_leaves: Vec<LeafData>,
    write_ops: Vec<(KeyPath, Option<ValueHash>)>,
    witness_paths: Vec<(BitVec<u8, Msb0>, Option<LeafData>)>,
    witnessed_ops: WitnessedOperations,
}

impl WitnessBuilder {
    // push keys, terminal info, and read/write info in ascending order.
    fn push(&mut self, key_path: KeyPath, terminal: TerminalInfo, read_write: &KeyReadWrite) {
        let slice = &key_path.view_bits::<Msb0>()[..terminal.depth as usize];
        let cur_terminal = if self
            .terminal
            .as_ref()
            .map_or(true, |(k, _t)| &k[..] != slice)
        {
            // new terminal. end last one
            self.consume_terminal();
            &mut self
                .terminal
                .insert((
                    slice.into(),
                    TerminalOps {
                        leaf: terminal.leaf.clone(),
                        writes: 0,
                    },
                ))
                .1
        } else {
            // unwrap: always exists in this branch
            &mut self.terminal.as_mut().unwrap().1
        };

        if let KeyReadWrite::Read(v) | KeyReadWrite::ReadThenWrite(v, _) = read_write {
            self.witnessed_ops.reads.push(WitnessedRead {
                key: key_path,
                value: v.clone(),
                path_index: self.witness_paths.len(), // index of yet-to-be-consumed terminal
            });
        }

        if let KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) = read_write {
            let value_hash = v.as_ref().map(|v| *blake3::hash(v).as_bytes());

            self.witnessed_ops.writes.push(WitnessedWrite {
                key: key_path,
                value: v.clone(),
                path_index: self.witness_paths.len(), // index of yet-to-be-consumed terminal
            });
            cur_terminal.writes += 1;
            self.write_ops.push((key_path, value_hash));
        }
    }

    fn consume_terminal(&mut self) {
        if let Some((prev_path, t)) = self.terminal.take() {
            self.witness_paths.push((prev_path, t.leaf.clone()));
            if let (true, Some(prev_leaf)) = (t.writes > 0, t.leaf) {
                self.write_leaves.push(prev_leaf);
            }
        }
    }

    // builds the witness, the witnessed operations, and updates the trie against the cursor.
    fn update_and_witness(
        mut self,
        cursor: &mut PageCacheCursor,
    ) -> (Witness, WitnessedOperations) {
        self.consume_terminal();

        let path_proofs = self
            .witness_paths
            .into_iter()
            .map(|(path, l)| {
                let (_, siblings) = nomt_core::proof::record_path(cursor, &path[..]);
                WitnessedPath {
                    path,
                    inner: PathProof {
                        terminal: l,
                        siblings,
                    },
                }
            })
            .collect::<Vec<_>>();

        nomt_core::update::update::<Blake3Hasher>(cursor, &self.write_ops, &self.write_leaves);
        (Witness { path_proofs }, self.witnessed_ops)
    }
}

struct TerminalOps {
    leaf: Option<LeafData>,
    writes: usize,
}
