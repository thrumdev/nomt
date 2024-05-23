//! A wrapper around libmdbx (lmbd analog) for avoiding prolifiration DB-specific code.

use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use reth_libmdbx::{Database, Environment, Geometry, PageSize, RW};
use std::sync::Arc;

static FLAT_KV_CF: &str = "flat_kv";
static PAGES_CF: &str = "pages";
static METADATA_CF: &str = "metadata";

const GIGABYTE: usize = 1024 * 1024 * 1024;
const TERABYTE: usize = GIGABYTE * 1024;

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    env: Environment,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        let env = Environment::builder()
            .set_max_dbs(256)
            .set_geometry(Geometry {
                // Maximum database size of 4 terabytes
                size: Some(0..(4 * TERABYTE)),
                // We grow the database in increments of 4 gigabytes
                growth_step: Some(4 * GIGABYTE as isize),
                // The database never shrinks
                shrink_threshold: Some(0),
                page_size: Some(PageSize::Set(8192)),
            })
            .open(&o.path)?;
        {
            let txn = env.begin_rw_txn().unwrap();
            let _ = txn.create_db(Some(FLAT_KV_CF), reth_libmdbx::DatabaseFlags::CREATE)?;
            let _ = txn.create_db(Some(PAGES_CF), reth_libmdbx::DatabaseFlags::CREATE)?;
            let _ = txn.create_db(Some(METADATA_CF), reth_libmdbx::DatabaseFlags::CREATE)?;
            txn.commit()?;
        }
        Ok(Self {
            shared: Arc::new(Shared { env }),
        })
    }

    /// Load the root node from the database. Fails only on I/O.
    /// Returns [`nomt_core::trie::TERMINATOR`] on an empty trie.
    pub fn load_root(&self) -> anyhow::Result<Node> {
        let txn = self.shared.env.begin_ro_txn().unwrap();
        let metadata_table = txn.open_db(Some(METADATA_CF)).unwrap();
        let value: Option<Vec<u8>> = txn.get(metadata_table.dbi(), b"root")?;
        match value {
            None => Ok(TERMINATOR),
            Some(value) => {
                let Ok(node) = value.try_into() else {
                    return Err(anyhow::anyhow!("invalid root hash length"));
                };
                Ok(node)
            }
        }
    }

    /// Loads the flat value stored under the given key.
    pub fn load_value(&self, key: KeyPath) -> anyhow::Result<Option<Vec<u8>>> {
        let txn = self.shared.env.begin_ro_txn().unwrap();
        let flat_kv_table = txn.open_db(Some(FLAT_KV_CF)).unwrap();
        let value: Option<Vec<u8>> = txn.get(flat_kv_table.dbi(), key.as_ref())?;
        Ok(value)
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        let txn = self.shared.env.begin_ro_txn().unwrap();
        let pages_table = txn.open_db(Some(PAGES_CF)).unwrap();
        let value: Option<Vec<u8>> =
            txn.get(pages_table.dbi(), page_id.length_dependent_encoding())?;
        Ok(value)
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        let txn = self.shared.env.begin_rw_txn().unwrap();
        let flat_kv_table = txn.open_db(Some(FLAT_KV_CF)).unwrap();
        let pages_table = txn.open_db(Some(PAGES_CF)).unwrap();
        let metadata_table = txn.open_db(Some(METADATA_CF)).unwrap();
        Transaction {
            txn,
            flat_kv_table,
            pages_table,
            metadata_table,
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, tx: Transaction) -> anyhow::Result<()> {
        tx.txn.commit()?;
        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    txn: reth_libmdbx::Transaction<RW>,
    flat_kv_table: Database,
    pages_table: Database,
    metadata_table: Database,
}

impl Transaction {
    /// Write a value to flat storage.
    pub fn write_value(&mut self, path: KeyPath, value: Option<&[u8]>) {
        match value {
            None => {
                self.txn.del(self.flat_kv_table.dbi(), path, None).unwrap();
            }
            Some(value) => {
                self.txn
                    .put(self.flat_kv_table.dbi(), path, value, Default::default())
                    .unwrap();
            }
        }
    }

    /// Write a page to storage in its entirety.
    pub fn write_page<V: AsRef<[u8]>>(&mut self, page_id: PageId, value: V) {
        self.txn
            .put(
                self.pages_table.dbi(),
                page_id.length_dependent_encoding(),
                value,
                Default::default(),
            )
            .unwrap();
    }

    /// Delete a page from storage.
    pub fn delete_page(&mut self, page_id: PageId) {
        self.txn
            .del(
                self.pages_table.dbi(),
                page_id.length_dependent_encoding(),
                None,
            )
            .unwrap();
    }

    /// Write the root to metadata.
    pub fn write_root(&mut self, root: Node) {
        self.txn
            .put(self.metadata_table.dbi(), b"root", root, Default::default())
            .unwrap()
    }
}
