use crate::{
    backend::Transaction,
    workload::{Init, Workload},
};
use rand::{Rng, SeedableRng as _};
use std::time::{SystemTime, UNIX_EPOCH};

/// Initialize a database with the given amount of key-value pairs.
pub fn init(db_size: u64) -> Init {
    Init {
        keys: (0..db_size).map(|id| encode_id(id).to_vec()).collect(),
        value: vec![64u8; 32],
    }
}

fn encode_id(id: u64) -> [u8; 8] {
    id.to_be_bytes()
}

/// Build a RwWorkload.
pub fn build(reads: u8, writes: u8, workload_size: u64, db_size: u64) -> RwWorkload {
    RwWorkload {
        reads,
        writes,
        workload_size,
        db_size,
    }
}

// The read-write workload will follow these rules:
// 1. Reads and writes are randomly and uniformly distributed across the key space.
// 2. The DB size indicates the number of entries in the database.
// 3. The workload size represents the total number of operations, where reads and writes
//     are numbers that need to sum to 100 and represent a percentage of the total size.
pub struct RwWorkload {
    pub reads: u8,
    pub writes: u8,
    pub workload_size: u64,
    pub db_size: u64,
}

impl Workload for RwWorkload {
    fn run(&mut self, transaction: &mut dyn Transaction) {
        let from_percentage = |p: u8| (self.workload_size as f64 * p as f64 / 100.0) as u64;
        let n_reads = from_percentage(self.reads);
        let n_writes = from_percentage(self.writes);

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("no time?")
            .as_nanos()
            .to_le_bytes()[0..16]
            .try_into()
            .unwrap();

        let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);

        for _ in 0..n_reads {
            let key = rng.gen_range(0..self.db_size);
            let _ = transaction.read(&encode_id(key));
        }

        for _ in 0..n_writes {
            let key = rng.gen_range(0..self.db_size);
            let value = rand_key(&mut rng);

            transaction.write(&encode_id(key), Some(&value));
        }
    }

    fn size(&self) -> usize {
        self.workload_size as usize
    }
}

fn rand_key(rng: &mut impl Rng) -> [u8; 32] {
    // keys must be uniformly distributed
    let mut key = [0; 32];
    key[0..4].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[4..8].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[8..12].copy_from_slice(&rng.next_u32().to_le_bytes());
    key[12..16].copy_from_slice(&rng.next_u32().to_le_bytes());
    key
}
