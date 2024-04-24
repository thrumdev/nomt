use crate::{
    backend::{Action, Db},
    timer::Timer,
};
use hash_db::{AsHashDB, HashDB, Hasher as _, Prefix};
use kvdb::KeyValueDB;
use kvdb_rocksdb::{Database, DatabaseConfig};
use sha2::Digest;
use sp_trie::trie_types::TrieDBMutBuilderV1;
use sp_trie::DBValue;
use std::collections::HashMap;
use std::sync::Arc;
use trie_db::TrieMut;

type Hasher = sp_core::Blake2Hasher;
type Hash = sp_core::H256;

const SP_TRIE_DB_FOLDER: &str = "sp_trie_db";
const SP_TRIE_DB_FOLDER_COPY: &str = "sp_trie_db_copy";

pub struct SpTrieDB {
    pub kvdb: Arc<dyn KeyValueDB>,
    pub root: Hash,
}

pub struct Trie<'a> {
    pub db: Arc<dyn KeyValueDB>,
    pub overlay: &'a mut HashMap<Vec<u8>, Option<Vec<u8>>>,
}

impl SpTrieDB {
    pub fn open(reset: bool) -> Self {
        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(SP_TRIE_DB_FOLDER);
        }

        let db_cfg = DatabaseConfig::with_columns(1);
        let kvdb =
            Arc::new(Database::open(&db_cfg, SP_TRIE_DB_FOLDER).expect("Database backend error"));

        let mut root = Hash::default();
        let mut overlay = HashMap::new();
        overlay.insert(
            array_bytes::hex2bytes(
                "03170a2e7597b7b7e3d84c05391d139a62b157e78786d8c082f29dcf4c111314",
            )
            .expect("null key is valid"),
            Some(vec![0]),
        );

        {
            let mut trie = Trie {
                db: kvdb.clone(),
                overlay: &mut overlay,
            };
            TrieDBMutBuilderV1::<Hasher>::new(&mut trie, &mut root)
                .build()
                .commit();
        }

        let mut transaction = kvdb.transaction();
        for (key, value) in overlay.into_iter() {
            match value {
                Some(value) => transaction.put(0, &key[..], &value[..]),
                None => transaction.delete(0, &key[..]),
            }
        }
        kvdb.write(transaction)
            .expect("Failed to write transaction");

        Self { kvdb, root }
    }
}

impl Db for SpTrieDB {
    fn open_copy(&self) -> Box<dyn Db> {
        // Delete any previously existing copy of the db
        let _ = std::fs::remove_dir_all(SP_TRIE_DB_FOLDER_COPY);

        std::process::Command::new("cp")
            .args(["-r", SP_TRIE_DB_FOLDER, SP_TRIE_DB_FOLDER_COPY])
            .output()
            .expect("Impossible make a copy of the nomt db");

        let db_cfg = DatabaseConfig::with_columns(1);
        let kvdb = Arc::new(
            Database::open(&db_cfg, SP_TRIE_DB_FOLDER_COPY).expect("Database backend error"),
        );

        Box::new(Self {
            kvdb,
            root: self.root,
        })
    }

    fn apply_actions(&mut self, actions: Vec<Action>, mut timer: Option<&mut Timer>) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        let mut new_root = self.root;
        let mut overlay: HashMap<Vec<u8>, Option<Vec<u8>>> = HashMap::new();

        let mut trie = Trie {
            db: self.kvdb.clone(),
            overlay: &mut overlay,
        };

        let recorder: sp_trie::recorder::Recorder<Hasher> = Default::default();
        let mut trie_recorder = recorder.as_trie_recorder(new_root);
        let mut trie_db_mut = TrieDBMutBuilderV1::from_existing(&mut trie, &mut new_root)
            .with_recorder(&mut trie_recorder)
            .build();

        for action in actions.into_iter() {
            match action {
                Action::Write { key, value } => {
                    // sp_trie does not require hashed keys,
                    // but if keys are not hashed, the comparison does not seem to be efficient.
                    // Not applying hashing to keys would significantly speed up sp_trie.
                    let key_path = sha2::Sha256::digest(key);

                    trie_db_mut
                        .insert(&key_path, &value.unwrap_or(vec![]))
                        .expect("Impossible writing into sp-trie db");
                }
                Action::Read { key } => {
                    let key_path = sha2::Sha256::digest(key);

                    let _timer_guard_read = timer.as_mut().map(|t| t.record_span("read"));
                    trie_db_mut
                        .get(&key_path)
                        .expect("Impossible fetching from sp-trie db");
                }
            }
        }

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));

        trie_db_mut.commit();
        drop(trie_db_mut);
        drop(trie_recorder);

        let _proof = recorder.drain_storage_proof().is_empty();

        let mut transaction = self.kvdb.transaction();
        for (key, value) in overlay.into_iter() {
            match value {
                Some(value) => transaction.put(0, &key[..], &value[..]),
                None => transaction.delete(0, &key[..]),
            }
        }
        self.kvdb
            .write(transaction)
            .expect("Failed to write transaction");

        self.root = new_root;
    }
}

impl<'a> AsHashDB<Hasher, DBValue> for Trie<'a> {
    fn as_hash_db(&self) -> &dyn hash_db::HashDB<Hasher, DBValue> {
        self
    }

    fn as_hash_db_mut<'b>(&'b mut self) -> &'b mut (dyn HashDB<Hasher, DBValue> + 'b) {
        &mut *self
    }
}

impl<'a> HashDB<Hasher, DBValue> for Trie<'a> {
    fn get(&self, key: &Hash, prefix: Prefix) -> Option<DBValue> {
        let key = sp_trie::prefixed_key::<Hasher>(key, prefix);
        if let Some(value) = self.overlay.get(&key) {
            return value.clone();
        }
        self.db.get(0, &key).expect("Database backend error")
    }

    fn contains(&self, hash: &Hash, prefix: Prefix) -> bool {
        self.get(hash, prefix).is_some()
    }

    fn insert(&mut self, prefix: Prefix, value: &[u8]) -> Hash {
        let key = Hasher::hash(value);
        self.emplace(key, prefix, value.to_vec());
        key
    }

    fn emplace(&mut self, key: Hash, prefix: Prefix, value: DBValue) {
        let key = sp_trie::prefixed_key::<Hasher>(&key, prefix);
        self.overlay.insert(key, Some(value));
    }

    fn remove(&mut self, key: &Hash, prefix: Prefix) {
        let key = sp_trie::prefixed_key::<Hasher>(key, prefix);
        self.overlay.insert(key, None);
    }
}
