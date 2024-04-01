use std::{path::PathBuf, sync::Arc};

use nomt_core::{
    proof::PathProof,
    trie::{Node, TERMINATOR},
};
use page_cache::PageCache;
use store::Store;
use threadpool::ThreadPool;

pub use nomt_core::trie::KeyPath;

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
    pub read_set: Vec<(KeyPath, Option<Value>)>,
    pub write_set: Vec<(KeyPath, Option<Value>)>,
}

pub struct Nomt {
    shared: Arc<Shared>,
}

impl Nomt {
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

    pub fn read_slot(&self, path: KeyPath) -> anyhow::Result<Option<Value>> {
        self.warmup(path);
        self.shared.store.load_value(path)
    }

    /// Signals to the backend that the given slot is going to be written to.
    ///
    /// It's not obligatory to call this function, but it is essential to do call this function as
    /// early as possible to achieve the best performance.
    pub fn hint_write_slot(&self, path: KeyPath) {
        self.warmup(path);
    }

    fn warmup(&self, path: KeyPath) {
        let page_cache = self.shared.page_cache.clone();
        let root = self.shared.root;
        let f = move || {
            page_cache.create_cursor(root).seek(path);
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
