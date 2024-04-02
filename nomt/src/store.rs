//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use rocksdb::{ColumnFamilyDescriptor, WriteBatch, DB};
use std::sync::Arc;

static FLAT_KV_CF: &str = "flat_kv";
static PAGES_CF: &str = "pages";
static METADATA_CF: &str = "metadata";

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    db: DB,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        let mut open_opts = rocksdb::Options::default();
        open_opts.set_error_if_exists(false);
        open_opts.create_if_missing(true);
        open_opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(FLAT_KV_CF, open_opts.clone()),
            ColumnFamilyDescriptor::new(PAGES_CF, open_opts.clone()),
            ColumnFamilyDescriptor::new(METADATA_CF, open_opts.clone()),
        ];
        let db = DB::open_cf_descriptors(&open_opts, &o.path, cf_descriptors)?;
        Ok(Self {
            shared: Arc::new(Shared { db }),
        })
    }

    /// Load the root node from the database. Fails only on I/O.
    /// Returns [`nomt_core::trie::TERMINATOR`] on an empty trie.
    pub fn load_root(&self) -> anyhow::Result<Node> {
        let cf = self.shared.db.cf_handle(METADATA_CF).unwrap();
        let value = self.shared.db.get_cf(&cf, b"root")?;
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
        let cf = self.shared.db.cf_handle(FLAT_KV_CF).unwrap();
        let value = self.shared.db.get_cf(&cf, key.as_ref())?;
        Ok(value)
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Vec<u8>> {
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        let value = self.shared.db.get_cf(&cf, page_id.as_ref())?;
        Ok(value.unwrap())
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        Transaction {
            shared: self.shared.clone(),
            batch: WriteBatch::default(),
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, tx: Transaction) -> anyhow::Result<()> {
        self.shared.db.write(tx.batch)?;
        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    shared: Arc<Shared>,
    batch: WriteBatch,
}

impl Transaction {
    /// Write a value to flat storage as part of the atomic batch.
    /// Any previous value is overwritten. `None` means delete.
    pub fn write_value(&mut self, path: KeyPath, value: Option<Vec<u8>>) {
        let cf = self.shared.db.cf_handle(FLAT_KV_CF).unwrap();
        match value {
            None => self.batch.delete_cf(&cf, path.as_ref()),
            Some(value) => self.batch.put_cf(&cf, path.as_ref(), value),
        }
    }

    /// Write a page to flat storage as part of the atomic batch.
    /// Any previous page is overwritten. This does not sanity check page-length.
    pub fn write_page<V: AsRef<[u8]>>(&mut self, page_id: PageId, value: Option<V>) {
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        match value {
            None => self.batch.delete_cf(&cf, page_id.as_ref()),
            Some(value) => self.batch.put_cf(&cf, page_id.as_ref(), value),
        }
    }
}