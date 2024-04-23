//! A wrapper around RocksDB for avoiding prolifiration of RocksDB-specific code.

use nomt_core::{
    page_id::PageId,
    trie::{KeyPath, Node, TERMINATOR},
};
use rocksdb::{ColumnFamilyDescriptor, WriteBatch, WriteOptions, DB};
use std::{sync::Arc, thread::JoinHandle};

static FLAT_KV_CF: &str = "flat_kv";
static PAGES_CF: &str = "pages";
static METADATA_CF: &str = "metadata";

/// This is a lightweight handle and can be cloned cheaply.
#[derive(Clone)]
pub struct Store {
    shared: Arc<Shared>,
}

struct Shared {
    db: Arc<DB>,
    bg_flusher: Option<JoinHandle<()>>,
    /// Channel used to communicate that a flush is requested.
    tx_flush: crossbeam_channel::Sender<()>,
    /// Channel used to communicate that the background flusher should stop.
    tx_stop: crossbeam_channel::Sender<()>,
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
        let db = Arc::new(db);

        let (bg_flusher, tx_flush, tx_stop) = spawn_bg_flusher(db.clone());
        Ok(Self {
            shared: Arc::new(Shared {
                db,
                bg_flusher: Some(bg_flusher),
                tx_flush,
                tx_stop,
            }),
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
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        let value = self
            .shared
            .db
            .get_cf(&cf, page_id.length_dependent_encoding())?;
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
        let mut writeopts = WriteOptions::new();
        writeopts.disable_wal(true);
        self.shared.db.write_opt(tx.batch, &writeopts)?;
        // NOTE on the ?. the other side can hung up only after the destructor of Shared is called,
        // and that cannot happen evidently because we are holding a reference to it.
        self.shared.tx_flush.send(())?;
        Ok(())
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let _ = self.tx_stop.send(());
        // unwrap: the only place we take from `bg_flusher` Option is here, in this destructor.
        // Destructor is called only once, so this is unwrap-safe.
        let _ = self.bg_flusher.take().unwrap().join();
    }
}

fn spawn_bg_flusher(
    db: Arc<DB>,
) -> (
    JoinHandle<()>,
    crossbeam_channel::Sender<()>,
    crossbeam_channel::Sender<()>,
) {
    let (tx_flush, rx_flush) = crossbeam_channel::bounded(1);
    let (tx_stop, rx_stop) = crossbeam_channel::bounded(1);
    let jh = std::thread::Builder::new()
        .name("nomt-bg-flusher".to_string())
        .spawn(move || {
            if let Err(e) = std::panic::catch_unwind(|| bg_flusher(db, rx_flush, rx_stop)) {
                // Best effort.
                eprintln!("background flusher panicked: {:?}", e);
            }
        })
        .unwrap();
    (jh, tx_flush, tx_stop)
}

fn bg_flusher(
    db: Arc<DB>,
    flush: crossbeam_channel::Receiver<()>,
    stop: crossbeam_channel::Receiver<()>,
) {
    loop {
        crossbeam_channel::select! {
            recv(flush) -> _ => {
                let _ = db.flush_wal(true); // TODO: handle error
            },
            recv(stop) -> _ => {
                let _ = db.flush_wal(true);
                return;
            }
        }
    }
}

/// An atomic transaction to be applied against th estore with [`Store::commit`].
pub struct Transaction {
    shared: Arc<Shared>,
    batch: WriteBatch,
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

    /// Write a page to flat storage as part of the atomic batch.
    /// Any previous page is overwritten. This does not sanity check page-length.
    pub fn write_page<V: AsRef<[u8]>>(&mut self, page_id: PageId, value: Option<V>) {
        let cf = self.shared.db.cf_handle(PAGES_CF).unwrap();
        match value {
            None => self
                .batch
                .delete_cf(&cf, page_id.length_dependent_encoding()),
            Some(value) => self
                .batch
                .put_cf(&cf, page_id.length_dependent_encoding(), value),
        }
    }

    /// Write the root to metadata.
    pub fn write_root(&mut self, root: Node) {
        let cf = self.shared.db.cf_handle(METADATA_CF).unwrap();
        self.batch.put_cf(&cf, b"root", &root[..]);
    }
}
