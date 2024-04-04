//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, LeafData, Node, NodeHasher, NodeHasherExt, ValueHash, TERMINATOR},
};
use rocksdb::{ColumnFamilyDescriptor, WriteBatch, DB};
use std::sync::Arc;

static FLAT_KV_CF: &str = "flat_kv";
static LEAF_CF: &str = "leaves";
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
            ColumnFamilyDescriptor::new(LEAF_CF, open_opts.clone()),
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

    /// Load the preimage of a leaf, given its hash.
    pub fn load_leaf(&self, hash: Node) -> anyhow::Result<Option<LeafData>> {
        let cf = self.shared.db.cf_handle(LEAF_CF).unwrap();
        let value = self.shared.db.get_cf(&cf, hash.as_ref())?;
        match value.map(|b| LeafData::decode(&b[..])) {
            None => Ok(None),
            Some(None) => Err(anyhow::anyhow!("invalid leaf preimage length")),
            Some(Some(value)) => Ok(Some(value)),
        }
    }

    /// Loads the given page.
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        let value = self.shared.db.get_cf(&cf, page_id.to_bytes().as_ref())?;
        Ok(value)
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
    /// Write a value to flat storage as well as its current leaf node representation
    /// as part of the atomic batch. Provide the previous value's hash to clean up any previous leaf
    /// node.
    pub fn write_value<H: NodeHasher>(
        &mut self,
        path: KeyPath,
        prev_value: Option<ValueHash>,
        value: Option<(ValueHash, Vec<u8>)>,
    ) {
        if value.as_ref().map(|(v, _)| v) == prev_value.as_ref() {
            return;
        }

        let flat_cf = self.shared.db.cf_handle(FLAT_KV_CF).unwrap();
        let leaf_cf = self.shared.db.cf_handle(LEAF_CF).unwrap();

        // Clears the previous leaf.
        if let Some(prev_value) = prev_value {
            let prev_hash = H::hash_leaf(&LeafData {
                key_path: path,
                value_hash: prev_value,
            });
            self.batch.delete_cf(&leaf_cf, prev_hash.as_ref());
        }

        match value {
            None => self.batch.delete_cf(&flat_cf, path.as_ref()),
            Some((value_hash, value)) => {
                self.batch.put_cf(&flat_cf, path.as_ref(), value);
                let leaf_data = LeafData {
                    key_path: path,
                    value_hash,
                };
                let new_hash = H::hash_leaf(&leaf_data);
                self.batch.put_cf(&leaf_cf, new_hash, leaf_data.encode());
            }
        }
    }

    /// Write a page to flat storage as part of the atomic batch.
    /// Any previous page is overwritten. This does not sanity check page-length.
    pub fn write_page<V: AsRef<[u8]>>(&mut self, page_id: PageId, value: Option<V>) {
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        match value {
            None => self.batch.delete_cf(&cf, page_id.to_bytes().as_ref()),
            Some(value) => self.batch.put_cf(&cf, page_id.to_bytes().as_ref(), value),
        }
    }
}
