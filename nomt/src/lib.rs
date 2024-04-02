use std::{path::PathBuf, sync::Arc};

use cursor::PageCacheCursor;
use nomt_core::{
    proof::PathProof,
    trie::{Node, TERMINATOR},
};
use page_cache::PageCache;
use store::Store;
use threadpool::ThreadPool;

pub use nomt_core::trie::KeyPath;

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
    root: Node,
    /// The handle to the page cache.
    page_cache: PageCache,
    store: Store,
    warmup_tp: ThreadPool,
}

/// A witness that can be used to prove the correctness of state trie retrievals and updates.
///
/// Expected to be serializable.
pub struct Witness {
    _proofs: Vec<(KeyPath, Vec<u8>, PathProof)>,
}

/// Specifies the final set of the keys that are read and written during the session.
///
/// The read set is necessary to create the witness, i.e. the set of all the values read during the
/// session. The write set is necessary to update the key-value store and to provide the witness
/// with the data necessary to update the trie.
pub struct CommitSpec {
    /// All keys read during any non-discarded execution paths, along with their value. A single
    /// key may appear only once.
    pub read_set: Vec<(KeyPath, Option<Value>)>,
    /// All values written during any non-discarded execution paths. A single key may appear
    /// only once - even if a key has intermediate values during execution, only the final change
    /// should be submitted.
    pub write_set: Vec<(KeyPath, Option<Value>)>,
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
                root,
                page_cache,
                store,
                warmup_tp: ThreadPool::new(o.traversal_concurrency),
            }),
        })
    }

    /// Returns the current root node of the trie.
    pub fn root(&self) -> &Node {
        &self.shared.root
    }

    /// Returns true if the trie has not been modified after the creation.
    pub fn is_empty(&self) -> bool {
        self.shared.root == TERMINATOR
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
        let root = self.shared.root;
        let f = move || {
            PageCacheCursor::at_root(root, page_cache).seek(path);
        };
        self.shared.warmup_tp.execute(f);
    }

    /// Commit the transaction and create a proof for the given read and write sets.
    pub fn commit_and_prove(&self, proof_spec: CommitSpec) -> anyhow::Result<Witness> {
        let mut tx = self.shared.store.new_tx();
        for (path, value) in proof_spec.write_set {
            tx.write_value(path, value)
        }
        // TODO: update the root in self.
        self.shared.page_cache.commit(&mut tx);
        self.shared.store.commit(tx)?;
        todo!()
    }
}
