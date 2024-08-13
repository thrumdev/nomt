//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use crate::{beatree, bitbox, page_cache::PageDiff};
use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use parking_lot::Mutex;
use rocksdb::{self, ColumnFamilyDescriptor, WriteBatch};
use std::sync::Arc;
use parking_lot::Mutex;

static FLAT_KV_CF: &str = "flat_kv";
static METADATA_CF: &str = "metadata";

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    root: Mutex<Node>,
    values: beatree::Tree,
    pages: bitbox::DB,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        let values = beatree::Tree::open(&o.path)?;

        let pages = bitbox::DB::open(o.num_rings, o.path.clone()).unwrap();

        Ok(Self {
            shared: Arc::new(Shared { values, pages, root: TERMINATOR.into() }),
        })
    }

    /// Load the root node from the database. Fails only on I/O.
    /// Returns [`nomt_core::trie::TERMINATOR`] on an empty trie.
    pub fn load_root(&self) -> anyhow::Result<Node> {
        Ok(self.shared.root.lock().clone())
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        Ok(self.shared.values.lookup(key))
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        self.shared.pages.get(&page_id)
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        Transaction {
            shared: self.shared.clone(),
            batch: Vec::new(),
            new_pages: vec![],
            new_root: None,
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, tx: Transaction) -> anyhow::Result<()> {
        self.shared.values.commit(tx.batch);
        self.shared.pages.commit(tx.new_pages)?;
        if let Some(new_root) = tx.new_root {
            *self.shared.root.lock() = new_root;
        }
        self.shared.values.sync();
        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    shared: Arc<Shared>,
    batch: Vec<(KeyPath, Option<Vec<u8>>)>,
    new_pages: Vec<(PageId, Option<(Vec<u8>, PageDiff)>)>,
    new_root: Option<Node>,
}

impl Transaction {
    /// Write a value to flat storage.
    pub fn write_value(&mut self, path: KeyPath, value: Option<&[u8]>) {
        self.batch.push((path, value.map(|v| v.to_vec())))
    }

    /// Write a page to storage in its entirety.
    pub fn write_page(&mut self, page_id: PageId, page: &Vec<u8>, page_diff: PageDiff) {
        self.new_pages
            .push((page_id, Some((page.clone(), page_diff))));
    }

    /// Delete a page from storage.
    pub fn delete_page(&mut self, page_id: PageId) {
        self.new_pages.push((page_id, None));
    }

    /// Write the root to metadata.
    pub fn write_root(&mut self, root: Node) {
        self.new_root = Some(root);
    }
}
