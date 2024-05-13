use crate::backend::Transaction;
use crate::timer::Timer;
use crate::workload::Workload;
use fxhash::FxHashMap;
use jmt::KeyHash;
use jmt::{storage::TreeWriter, JellyfishMerkleTree, Version};
use sov_db::state_db::StateDB;
use sov_prover_storage_manager::SnapshotManager;
use sov_schema_db::snapshot::DbSnapshot;
use std::{
    collections::hash_map::Entry,
    sync::{Arc, RwLock},
};

const SOV_DB_FOLDER: &str = "sov_db";

pub struct SovDB {
    state_db: StateDB<SnapshotManager>,
    version: Version,
}

impl SovDB {
    pub fn open(reset: bool) -> Self {
        if reset {
            // Delete previously existing db
            let _ = std::fs::remove_dir_all(SOV_DB_FOLDER);
        }

        // Create the underlying rocks db database
        let state_db_raw = StateDB::<SnapshotManager>::setup_schema_db(SOV_DB_FOLDER).unwrap();
        // Create a 'dummy' SnapshotManager who just reads from the provided database
        let state_db_sm = Arc::new(RwLock::new(SnapshotManager::orphan(state_db_raw)));
        // Create a snapshot db and then the state db over RocksDB
        let state_db_snapshot = DbSnapshot::<SnapshotManager>::new(0, state_db_sm.into());
        let state_db = StateDB::with_db_snapshot(state_db_snapshot).unwrap();

        Self {
            state_db,
            version: 1,
        }
    }

    pub fn execute(&mut self, mut timer: Option<&mut Timer>, workload: &mut dyn Workload) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        self.state_db.inc_next_version();

        // Actions are applied to jmt and then applied to the backend
        let jmt = JellyfishMerkleTree::<_, sha2::Sha256>::new(&self.state_db);

        // Sov-db uses the struct `ProverStorage` to handle the modification of the trie.
        // https://github.com/Sovereign-Labs/sovereign-sdk/blob/2cc0656df3f12fca2026c20554b5f78ccb210b89/module-system/sov-state/src/prover_storage.rs#L75
        // What it does is:
        // + get
        //      read value from db (apparently not from the trie but from the other DBs),
        //      Add an hint into the witness, the hint is just the serialization of the value.
        // + compute_state_update
        //      accepts `OrderedReadsAndWrites` and `Witness`,
        //      For each value that's been read from the tree,
        //      read it from the JMT and populate witness with proof
        //      (proofs are sequentil, one for each read)
        //      Create the key_preimages vector (key_hash to key) and
        //      the batch of writes.
        //      Change version of the JMT and put the batch with the proof
        //      that will be added to the witness
        // + commit
        //      insert key_preimages with `put_preimage` and write the
        //      node batch created by the `compute_state_update` method
        //
        // Reads are executed sequentially, reading actions, while writes
        // are collected and applied at the end before committing everything
        // to the database

        let mut transaction = Tx {
            timer,
            access: FxHashMap::default(),
            jmt,
            version: self.version,
        };
        workload.run(&mut transaction);
        let Tx {
            mut timer,
            mut access,
            jmt,
            ..
        } = transaction;
        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        // apply all writes
        // We are not interested in storing the witness, but we want to measure
        // the time required to create the proof
        let tree_update = {
            let value_set = access
                .iter_mut()
                .map(|(k, v)| (k.clone(), v.as_mut().map(|v| std::mem::take(&mut v.value))));
            let (_new_root, _proof, tree_update) = jmt
                .put_value_set_with_proof(value_set, self.version)
                .expect("JMT update must succeed");

            tree_update
        };

        let preimages = access
            .iter()
            .filter_map(|(k, v)| v.as_ref().map(move |v| (*k, &v.preimage)));

        self.state_db.put_preimages(preimages).unwrap();
        self.state_db
            .write_node_batch(&tree_update.node_batch)
            .unwrap();

        self.version += 1;
    }
}

struct ValueWithPreimage {
    value: Vec<u8>,
    preimage: Vec<u8>,
}

struct Tx<'a> {
    timer: Option<&'a mut Timer>,
    access: FxHashMap<KeyHash, Option<ValueWithPreimage>>,
    jmt: JellyfishMerkleTree<'a, StateDB<SnapshotManager>, sha2::Sha256>,
    version: Version,
}

impl<'a> Transaction for Tx<'a> {
    fn read(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        let key_hash = KeyHash::with::<sha2::Sha256>(&key);
        let _timer_guard_read = self.timer.as_mut().map(|t| t.record_span("read"));

        match self.access.entry(key_hash) {
            Entry::Occupied(o) => o.get().as_ref().map(|v| v.value.clone()),
            Entry::Vacant(_) => self.jmt.get_with_proof(key_hash, self.version).unwrap().0,
        }
    }
    fn write(&mut self, key: &[u8], value: Option<&[u8]>) {
        let key_hash = KeyHash::with::<sha2::Sha256>(&key);

        let value = value.map(|v| ValueWithPreimage {
            value: v.to_vec(),
            preimage: key.to_vec(),
        });

        self.access.insert(key_hash, value);
    }
}
