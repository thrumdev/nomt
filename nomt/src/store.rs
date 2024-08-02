//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use rocksdb::{ColumnFamilyDescriptor, MergeOperands, WriteBatch, DB};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{bitbox, page_cache::PageDiff};

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
    bitbox: bitbox::DB,
}

impl Store {
    /// Open the store with the provided `Options`.
    pub fn open(o: &crate::Options) -> anyhow::Result<Self> {
        let mut open_opts = rocksdb::Options::default();
        open_opts.set_error_if_exists(false);
        open_opts.create_if_missing(true);
        open_opts.create_missing_column_families(true);

        let pages_cf_opts = {
            let mut opts = open_opts.clone();
            opts.set_merge_operator("page update operator", merge_page, partial_merge_page_ops);

            opts
        };

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(FLAT_KV_CF, open_opts.clone()),
            ColumnFamilyDescriptor::new(PAGES_CF, pages_cf_opts),
            ColumnFamilyDescriptor::new(METADATA_CF, open_opts.clone()),
        ];
        let db = DB::open_cf_descriptors(&open_opts, &o.path, cf_descriptors)?;

        // TODO: add option to specify number of io_uring instances
        let bitbox = bitbox::DB::open(3, o.path.clone()).unwrap();

        Ok(Self {
            shared: Arc::new(Shared { db, bitbox }),
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
    pub fn load_page(&self, page_id: PageId) -> anyhow::Result<Option<Vec<u8>>> {
        self.shared.bitbox.get(&page_id)
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
        self.shared.db.write(tx.batch)?;
        self.shared.bitbox.commit(tx.new_pages)?;
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
        let flat_cf = self.shared.db.cf_handle(FLAT_KV_CF).unwrap();

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
        let cf = self.shared.db.cf_handle(METADATA_CF).unwrap();
        self.batch.put_cf(&cf, b"root", &root[..]);
    }
}

fn merge_page(_key: &[u8], prev_value: Option<&[u8]>, operands: &MergeOperands) -> Option<Vec<u8>> {
    let mut val = prev_value.map(|x| x.to_vec());
    for op in operands.iter() {
        for op in op.chunks_exact(33) {
            let val = val.get_or_insert_with(|| vec![0u8; 4096]);
            let slot_index = op[0] as usize;

            let slot_start = slot_index * 32;
            let slot_end = slot_start + 32;
            val[slot_start..slot_end].copy_from_slice(&op[1..]);
        }
    }

    // merge operators are not supposed to fail unless there is corruption.
    val
}

fn partial_merge_page_ops(
    _key: &[u8],
    prev_value: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut slots = HashMap::new();

    for prev_op in prev_value.into_iter().flat_map(|x| x.chunks_exact(33)) {
        slots.insert(prev_op[0], prev_op);
    }

    for op in operands.iter() {
        for op in op.chunks_exact(33) {
            slots.insert(op[0], op);
        }
    }

    if slots.is_empty() {
        None
    } else {
        let mut v = Vec::with_capacity(33 * slots.len());
        for (_, op) in slots.into_iter() {
            v.extend(op);
        }
        Some(v)
    }
}
