use crate::backend::{Action, Db};
use crate::timer::Timer;
use jmt::KeyHash;
use jmt::{storage::TreeWriter, JellyfishMerkleTree};
use sov_db::state_db::StateDB;
use sov_prover_storage_manager::SnapshotManager;
use sov_schema_db::snapshot::DbSnapshot;
use std::sync::{Arc, RwLock};

const SOV_DB_FOLDER: &str = "sov_db";
const SOV_DB_FOLDER_COPY: &str = "sov_db_copy";

pub struct SovDB {
    state_db: StateDB<SnapshotManager>,
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

        Self { state_db }
    }
}

impl Db for SovDB {
    fn open_copy(&self) -> Box<dyn Db> {
        // Delete any previously existing copy of the db
        let _ = std::fs::remove_dir_all(SOV_DB_FOLDER_COPY);

        std::process::Command::new("cp")
            .args(["-r", SOV_DB_FOLDER, SOV_DB_FOLDER_COPY])
            .output()
            .expect("Impossible make a copy of the nomt db");

        // Create the underlying rocks db database
        let state_db_raw = StateDB::<SnapshotManager>::setup_schema_db(SOV_DB_FOLDER_COPY).unwrap();
        // Create a 'dummy' SnapshotManager who just reads from the provided database
        let state_db_sm = Arc::new(RwLock::new(SnapshotManager::orphan(state_db_raw)));
        // Create a snapshot db and then the state db over RocksDB
        let state_db_snapshot = DbSnapshot::<SnapshotManager>::new(0, state_db_sm.into());
        let state_db = StateDB::with_db_snapshot(state_db_snapshot).unwrap();

        Box::new(Self { state_db })
    }
    fn apply_actions(&mut self, actions: Vec<Action>, mut timer: Option<&mut Timer>) {
        let _timer_guard_total = timer.as_mut().map(|t| t.record_span("workload"));

        self.state_db.inc_next_version();

        // Actions are applied to jmt and then applied to the backend
        let jmt = JellyfishMerkleTree::<_, sha2::Sha256>::new(&self.state_db);

        let mut preimages = vec![];
        let mut value_set = vec![];

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
        //
        // TODO: Collect writes is not technically correct, in a real scenario,
        // there could be reads that refer to previous writes,
        // but for now let's just collect them
        for action in actions.into_iter() {
            match action {
                Action::Write { key, value } => {
                    let key_hash = KeyHash::with::<sha2::Sha256>(&key);

                    value_set.push((key_hash, value));
                    preimages.push((key_hash, key.clone()));
                }
                Action::Read { key } => {
                    let key_hash = KeyHash::with::<sha2::Sha256>(&key);

                    let _timer_guard_read = timer.as_mut().map(|t| t.record_span("read"));
                    let _result = jmt.get_with_proof(key_hash, 1);
                }
            }
        }

        let _timer_guard_commit = timer.as_mut().map(|t| t.record_span("commit_and_prove"));
        // apply all writes
        // We are not interested in storing the witness, but we want to measure
        // the time required to create the proof
        let (_new_root, _proof, tree_update) = jmt
            .put_value_set_with_proof(value_set, 1)
            .expect("JMT update must succeed");

        self.state_db
            .put_preimages(preimages.iter().map(|(k, v)| (*k, v)))
            .unwrap();
        self.state_db
            .write_node_batch(&tree_update.node_batch)
            .unwrap();
    }
}
