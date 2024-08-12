//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use crate::{bitbox, page_cache::PageDiff};
use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use rocksdb::{self, ColumnFamilyDescriptor, WriteBatch};
use std::sync::Arc;

static FLAT_KV_CF: &str = "flat_kv";
static METADATA_CF: &str = "metadata";

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    // TODO: change values name, it stores also the root
    values: rocksdb::DB,
    pages: bitbox::DB,
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
            ColumnFamilyDescriptor::new(METADATA_CF, open_opts.clone()),
        ];

        let values = rocksdb::DB::open_cf_descriptors(&open_opts, &o.path, cf_descriptors)?;

        // TODO: add option to specify number of io_uring instances
        let pages = bitbox::DB::open(3, o.path.clone())?;

        Ok(Self {
            shared: Arc::new(Shared { values, pages }),
        })
    }

    /// Load the root node from the database. Fails only on I/O.
    /// Returns [`nomt_core::trie::TERMINATOR`] on an empty trie.
    pub fn load_root(&self) -> anyhow::Result<Node> {
        let cf = self.shared.values.cf_handle(METADATA_CF).unwrap();
        let value = self.shared.values.get_cf(&cf, b"root")?;
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
        let cf = self.shared.values.cf_handle(FLAT_KV_CF).unwrap();
        let value = self.shared.values.get_cf(&cf, key.as_ref())?;
        Ok(value)
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        self.shared.pages.get(&page_id)
    }

    /// Create a new transaction to be applied against this database.
    pub fn new_tx(&self) -> Transaction {
        Transaction {
            shared: self.shared.clone(),
            batch: WriteBatch::default(),
            new_pages: vec![],
        }
    }

    /// Atomically apply the given transaction.
    ///
    /// After this function returns, accessor methods such as [`Self::load_page`] will return the
    /// updated values.
    pub fn commit(&self, tx: Transaction) -> anyhow::Result<()> {
        self.shared.values.write(tx.batch)?;
        self.shared.pages.commit(tx.new_pages)?;
        Ok(())
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    shared: Arc<Shared>,
    batch: WriteBatch,
    new_pages: Vec<(PageId, Option<(Vec<u8>, PageDiff)>)>,
}

impl Transaction {
    /// Write a value to flat storage.
    pub fn write_value(&mut self, path: KeyPath, value: Option<&[u8]>) {
        let flat_cf = self.shared.values.cf_handle(FLAT_KV_CF).unwrap();

        match value {
            None => self.batch.delete_cf(&flat_cf, path.as_ref()),
            Some(value) => {
                self.batch.put_cf(&flat_cf, path.as_ref(), value);
            }
        }
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
        let cf = self.shared.values.cf_handle(METADATA_CF).unwrap();
        self.batch.put_cf(&cf, b"root", &root[..]);
    }
}
