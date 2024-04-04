use bitvec::prelude::*;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use cursor::PageCacheCursor;
use nomt_core::{
    proof::PathProof,
    trie::{NodeHasher, TERMINATOR},
};
use page_cache::PageCache;
use parking_lot::Mutex;
use store::Store;
use threadpool::ThreadPool;

pub use nomt_core::trie::{KeyPath, LeafData, Node};

mod cursor;
mod page_cache;
mod store;

pub type Value = Vec<u8>;

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
    root: Mutex<Node>,
    /// The handle to the page cache.
    page_cache: PageCache,
    store: Store,
    warmup_tp: ThreadPool,
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

/// Specifies the final set of the keys that are read and written during the session.
///
/// The read set is necessary to create the witness, i.e. the set of all the values read during the
/// session. The write set is necessary to update the key-value store and to provide the witness
/// with the data necessary to update the trie.
///
/// The `LeafData` provided is the preimage of the terminal leaf node encountered when seeking the
/// given key in the  current revision of the trie. This may be `Some` even when the key in
/// question does not have a prior value, but will always be `Some` when the key in question
/// has a prior value.
///
/// For example, when seeking key `01011`, there are three possibilities: finding a terminal,
/// finding a leaf for `01011`, or finding a leaf for another key, such as `01010`. This is
/// what should be provided.
pub struct CommitSpec {
    /// All keys read, written, or both during any non-discarded execution paths.
    /// A single key may appear only once.
    pub updates: Vec<(KeyPath, TerminalInfo, KeyReadWrite)>,
}

/// Information about an encountered terminal during the lookup of some key.
#[derive(Clone)]
pub struct TerminalInfo {
    /// A leaf, if this is a leaf. `None` indicates this is a terminator node.
    pub leaf: Option<LeafData>,
    /// The depth
    pub depth: u8,
}

/// Whether a key was read, written, or both, along with old and new values.
pub enum KeyReadWrite {
    /// The key was read. Contains the read value.
    Read(Option<Value>),
    /// The key was written. Contains the written value.
    Write(Option<Value>),
    /// The key was both read and written. Contains the previous value and the new value.
    Both(Option<Value>, Option<Value>),
}

struct Blake3Hasher;

impl NodeHasher for Blake3Hasher {
    fn hash_node(data: &nomt_core::trie::NodePreimage) -> [u8; 32] {
        blake3::hash(data).into()
    }
}

/// An instance of the Nearly-Optimal Merkle Trie Database.
pub struct Nomt {
    shared: Arc<Shared>,
}

impl Nomt {
    /// Open the database with the given options.
    pub fn open(o: Options) -> anyhow::Result<Self> {
        let store = Store::open(&o)?;
        let page_cache = PageCache::new(store.clone(), &o);
        let root = store.load_root()?;
        Ok(Self {
            shared: Arc::new(Shared {
                root: Mutex::new(root),
                page_cache,
                store,
                warmup_tp: threadpool::Builder::new()
                    .num_threads(o.fetch_concurrency)
                    .thread_name("nomt-warmup".to_string())
                    .build(),
            }),
        })
    }

    /// Returns the current root node of the trie.
    pub fn root(&self) -> Node {
        self.shared.root.lock().clone()
    }

    /// Returns true if the trie has not been modified after the creation.
    pub fn is_empty(&self) -> bool {
        self.root() == TERMINATOR
    }

    /// Synchronously read the value stored at the given key. Fails only if I/O fails.
    pub fn read_slot(&self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        self.warmup(path);
        self.shared.store.load_value(path)
    }

    /// Signals to the backend that the given slot is going to be written to.
    ///
    /// It's not obligatory to call this function, but it is essential to call this function as
    /// early as possible to achieve the best performance.
    pub fn hint_write_slot(&self, path: KeyPath) {
        self.warmup(path);
    }

    fn warmup(&self, path: KeyPath) {
        let page_cache = self.shared.page_cache.clone();
        let root = self.shared.root.lock().clone();
        let f = move || {
            PageCacheCursor::at_root(root, page_cache).seek(path);
        };
        self.shared.warmup_tp.execute(f);
    }

    /// Commit the transaction and create a proof for the given read and write sets.
    pub fn commit_and_prove(&self, proof_spec: CommitSpec) -> anyhow::Result<Witness> {
        let prev_root = self.shared.root.lock().clone();
        let mut tx = self.shared.store.new_tx();
        let mut cursor = PageCacheCursor::at_root(prev_root, self.shared.page_cache.clone());

        let mut ops = vec![];
        let mut witness_builder = WitnessBuilder::default();
        for (path, terminal_info, read_write) in proof_spec.updates {
            witness_builder.insert(path, terminal_info.clone(), &read_write);

            if let KeyReadWrite::Write(value) | KeyReadWrite::Both(_, value) = read_write {
                let value_hash = value.as_ref().map(|v| *blake3::hash(v).as_bytes());
                let prev_value = match terminal_info.leaf.as_ref() {
                    None => None,
                    Some(l) if l.key_path == path => Some(l.value_hash),
                    Some(_) => None,
                };
                ops.push((path, value_hash));
                tx.write_value::<Blake3Hasher>(path, prev_value, value_hash.zip(value));
            }
        }
        let (witness, _witnessed, visited_leaves) = witness_builder.build(&mut cursor);
        ops.sort_by(|(a, _), (b, _)| a.cmp(b));

        nomt_core::update::update::<Blake3Hasher>(&mut cursor, &ops, &visited_leaves);
        cursor.rewind();
        *self.shared.root.lock() = cursor.node();

        self.shared.page_cache.commit(&mut tx);
        self.shared.store.commit(tx)?;
        Ok(witness)
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

        if let KeyReadWrite::Read(v) | KeyReadWrite::Both(v, _) = read_write {
            entry.reads.push((key_path, v.clone()));
        }

        if let KeyReadWrite::Write(v) | KeyReadWrite::Both(_, v) = read_write {
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
    fn build(self, cursor: &mut PageCacheCursor) -> (Witness, WitnessedOperations, Vec<LeafData>) {
        let mut paths = Vec::with_capacity(self.terminals.len());
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut visited_leaves = Vec::new();

        for (i, (path, terminal)) in self.terminals.into_iter().enumerate() {
            let (_, siblings) = nomt_core::proof::record_path(cursor, &path[..]);

            let path_proof = PathProof {
                terminal: terminal.leaf.clone(),
                siblings,
            };
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
            if let Some(leaf) = terminal.leaf {
                visited_leaves.push(leaf)
            }
        }

        (
            Witness { path_proofs: paths },
            WitnessedOperations { reads, writes },
            visited_leaves,
        )
    }
}

struct TerminalOps {
    leaf: Option<LeafData>,
    reads: Vec<(KeyPath, Option<Value>)>,
    writes: Vec<(KeyPath, Option<Value>)>,
    preserve: bool,
}
