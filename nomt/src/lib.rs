use bitvec::prelude::*;
use dashmap::DashMap;
use std::{
    collections::BTreeMap,
    mem,
    path::PathBuf,
    rc::Rc,
    sync::{atomic::AtomicUsize, Arc},
};

use cursor::PageCacheCursor;
use nomt_core::{
    proof::PathProof,
    trie::{NodeHasher, TERMINATOR},
    update::VisitedTerminal,
};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::Store;
use threadpool::ThreadPool;

pub use nomt_core::trie::{KeyPath, LeafData, Node};

mod cursor;
mod page_cache;
mod rw_pass_cell;
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
#[derive(Clone)]
struct TerminalInfo {
    /// A leaf, if this is a leaf. `None` indicates this is a terminator node.
    pub leaf: Option<LeafData>,
    /// The depth
    pub depth: u8,
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
    session_cnt: Arc<AtomicUsize>,
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
            session_cnt: Arc::new(AtomicUsize::new(0)),
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
            terminals: Arc::new(DashMap::new()),
            session_cnt: self.session_cnt.clone(),
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
        mut session: Session,
        actuals: Vec<(KeyPath, KeyReadWrite)>,
    ) -> anyhow::Result<(Node, Witness)> {
        // Wait for all warmup tasks to finish. That way, we can be sure that all terminal
        // information is available and that `terminals` would be the only reference.
        self.warmup_tp.join();

        let prev_root = session.prev_root;
        // unwrap: since we waited for all warmup tasks to finish, there should be no other
        // references to `terminals`.
        let terminals = Arc::get_mut(&mut session.terminals).unwrap();

        let mut tx = self.store.new_tx();
        let mut cursor = self.page_cache.new_write_cursor(prev_root);

        let mut ops = vec![];
        let mut witness_builder = WitnessBuilder::default();
        for (path, read_write) in actuals {
            let terminal_info = terminals.remove(&path).expect("terminal info not found").1;
            if let KeyReadWrite::Write(ref value) | KeyReadWrite::ReadThenWrite(_, ref value) =
                read_write
            {
                let value_hash = value.as_ref().map(|v| *blake3::hash(v).as_bytes());
                let prev_value = match terminal_info.leaf.as_ref() {
                    None => None,
                    Some(l) if l.key_path == path => Some(l.value_hash),
                    Some(_) => None,
                };
                ops.push((path, value_hash));
                tx.write_value::<Blake3Hasher>(
                    path,
                    prev_value,
                    value_hash.zip(value.as_ref().map(|v| &v[..])),
                );
            }
            witness_builder.insert(path, terminal_info.clone(), &read_write);
        }
        let (witness, _witnessed, visited_terminals) = witness_builder.build(&mut cursor);
        ops.sort_by(|(a, _), (b, _)| a.cmp(b));

        nomt_core::update::update::<Blake3Hasher>(&mut cursor, &ops, &visited_terminals);
        cursor.rewind();
        let new_root = cursor.node();
        self.shared.lock().root = cursor.node();
        tx.write_root(new_root);

        self.page_cache.commit(cursor, &mut tx);
        self.store.commit(tx)?;
        Ok((new_root, witness))
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
    terminals: Arc<DashMap<KeyPath, TerminalInfo>>,
    session_cnt: Arc<AtomicUsize>,
}

impl Session {
    /// Synchronously read the value stored at the given key.
    ///
    /// Returns `None` if the value is not stored under the given key. Fails only if I/O fails.
    pub fn tentative_read_slot(&mut self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        self.warmup(path);
        let value = self.store.load_value(path)?.map(Rc::new);
        Ok(value)
    }

    /// Signals that the given key is going to be written to.
    pub fn tentative_write_slot(&mut self, path: KeyPath) {
        self.warmup(path);
    }

    fn warmup(&self, path: KeyPath) {
        let page_cache = self.page_cache.clone();
        let store = self.store.clone();
        let terminals = self.terminals.clone();
        let root = self.prev_root;
        let f = move || {
            let mut cur = page_cache.new_read_cursor(root);
            cur.seek(path);
            let (_, depth) = cur.position();
            let node = cur.node();
            let terminal_info = if nomt_core::trie::is_leaf(&node) {
                let leaf = store.load_leaf(node).unwrap(); // TODO: handle error
                TerminalInfo { leaf, depth }
            } else {
                TerminalInfo { leaf: None, depth }
            };
            terminals.insert(path, terminal_info);
        };
        self.warmup_tp.execute(f);
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

#[derive(Default)]
struct WitnessBuilder {
    terminals: BTreeMap<BitVec<u8, Msb0>, TerminalOps>,
}

impl WitnessBuilder {
    fn insert(&mut self, key_path: KeyPath, terminal: TerminalInfo, read_write: &KeyReadWrite) {
        let slice = key_path.view_bits::<Msb0>()[..terminal.depth as usize].into();
        let entry = self.terminals.entry(slice).or_insert_with(|| TerminalOps {
            leaf: terminal.leaf.clone(),
            reads: Vec::new(),
            writes: Vec::new(),
            preserve: true,
        });

        if let KeyReadWrite::Read(v) | KeyReadWrite::ReadThenWrite(v, _) = read_write {
            entry.reads.push((key_path, v.clone()));
        }

        if let KeyReadWrite::Write(v) | KeyReadWrite::ReadThenWrite(_, v) = read_write {
            entry.writes.push((key_path, v.clone()));
            if entry
                .leaf
                .as_ref()
                .map_or(false, |leaf| leaf.key_path == key_path)
            {
                entry.preserve = false;
            }
        }
    }

    // builds the witness, the witnessed operations, and returns additional write operations
    // for leaves which are updated but where the existing value should be preserved.
    fn build(self, cursor: &mut PageCacheCursor) -> (Witness, WitnessedOperations, Vec<VisitedTerminal>) {
        let mut paths = Vec::with_capacity(self.terminals.len());
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut visited_terminals = Vec::new();

        for (i, (path, terminal)) in self.terminals.into_iter().enumerate() {
            let (_, siblings) = nomt_core::proof::record_path(cursor, &path[..]);

            let path_proof = PathProof {
                terminal: terminal.leaf.clone(),
                siblings,
            };
            visited_terminals.push(VisitedTerminal {
                path: {
                    let mut full_path = KeyPath::default();
                    full_path.view_bits_mut::<Msb0>()[..path.len()].copy_from_bitslice(&path);
                    full_path
                },
                depth: path.len() as u8,
                leaf: terminal.leaf,
            });
            paths.push(WitnessedPath {
                inner: path_proof,
                path,
            });
            reads.extend(
                terminal
                    .reads
                    .into_iter()
                    .map(|(key, value)| WitnessedRead {
                        key,
                        value,
                        path_index: i,
                    }),
            );
            writes.extend(
                terminal
                    .writes
                    .iter()
                    .cloned()
                    .map(|(key, value)| WitnessedWrite {
                        key,
                        value,
                        path_index: i,
                    }),
            );
        }

        (
            Witness { path_proofs: paths },
            WitnessedOperations { reads, writes },
            visited_terminals,
        )
    }
}

struct TerminalOps {
    leaf: Option<LeafData>,
    reads: Vec<(KeyPath, Option<Value>)>,
    writes: Vec<(KeyPath, Option<Value>)>,
    preserve: bool,
}
