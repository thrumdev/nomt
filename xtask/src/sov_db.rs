use crate::{Action, DB};
use jmt::KeyHash;
use jmt::{storage::TreeWriter, JellyfishMerkleTree};
use sov_db::state_db::StateDB;
use sov_prover_storage_manager::SnapshotManager;
use sov_schema_db::snapshot::DbSnapshot;
use std::sync::{Arc, RwLock};

// https://github.com/Sovereign-Labs/sovereign-sdk/blob/2cc0656df3f12fca2026c20554b5f78ccb210b89/full-node/db/sov-db/benches/state_db_bench.rs#L70

pub struct SovDB {
    state_db: StateDB<SnapshotManager>,
}

impl SovDB {
    pub fn new() -> Self {
        // Delete previously existing db
        let _ = std::fs::remove_dir_all("tmp_sov_db");

        // Create the underlying rocks db database
        let state_db_raw = StateDB::<SnapshotManager>::setup_schema_db("tmp_sov_db").unwrap();

        // Create a 'dummy' SnapshotManager who just reads from the provided database
        let state_db_sm = Arc::new(RwLock::new(SnapshotManager::orphan(state_db_raw)));

        // Create a snapshot db and then the state db over RocksDB
        let state_db_snapshot = DbSnapshot::<SnapshotManager>::new(0, state_db_sm.into());
        let state_db = StateDB::with_db_snapshot(state_db_snapshot).unwrap();

        Self { state_db }
    }
}

impl DB for SovDB {
    fn apply_actions(&mut self, actions: Vec<Action>) {
        self.state_db.inc_next_version();

        // Actions are applied to jmt and then applied to the backend if committed
        let jmt = JellyfishMerkleTree::<_, sha2::Sha256>::new(&self.state_db);

        let mut preimages = vec![];
        let mut tree_update = None;

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
        for action in actions.into_iter() {
            match action {
                Action::Writes(writes) => {
                    let mut value_set = vec![];
                    for (key, val) in writes.into_iter() {
                        let key_hash = KeyHash::with::<sha2::Sha256>(&key);
                        value_set.push((key_hash, val));
                        preimages.push((key_hash, key.clone()));
                    }

                    // We are not interested in storing the witness, but we want to measure
                    // the time required to create the proof
                    let (_new_root, _proof, jmt_tree_update) = jmt
                        .put_value_set_with_proof(value_set, 1)
                        .expect("JMT update must succeed");
                    tree_update = Some(jmt_tree_update);
                }
                _ => todo!(),
            }
        }

        // commit and prove
        let Some(ref update) = tree_update else {
            panic!("Commit without any write")
        };
        self.state_db
            .put_preimages(preimages.iter().map(|(k, v)| (*k, v)))
            .unwrap();
        self.state_db.write_node_batch(&update.node_batch).unwrap();
    }
}
