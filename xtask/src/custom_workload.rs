use crate::{
    backend::{Action, Db},
    timer::Timer,
    workload::Workload,
};
use anyhow::Result;

#[derive(Debug, Clone)]
#[allow(unused)]
pub struct CustomWorkload {
    init_actions: Vec<Action>,
    run_actions: Vec<Action>,
}

impl CustomWorkload {
    // The custom workload will follow these rules:
    // 1. Reads and writes are randomly and uniformly distributed over the key space.
    // 2. Deletes and updates will be performed on already present keys.
    // 3. additional_initial_capacity represents the amount of items already present
    //     in the DB in addition to the ones required to perform deletes and updates.
    // 4. Size represents the total number of operations, where reads, writes, etc
    //     are numbers that need to sum to 100 and represent a percentage of the total size.
    pub fn new(
        reads: u8,
        writes: u8,
        deletes: u8,
        updates: u8,
        size: u64,
        additional_initial_capacity: u64,
    ) -> Result<Self> {
        if reads + writes + deletes + updates != 100 {
            anyhow::bail!("Operations (reads, writes, deletes, updates) must sum to 100");
        }

        let from_percentage = |p: u8| (size as f64 * p as f64 / 100.0) as u64;
        let n_reads = from_percentage(reads);
        let n_writes = from_percentage(writes);
        let n_deletes = from_percentage(deletes);
        let n_updates = from_percentage(updates);

        let n_required_key = n_deletes + n_updates;
        let initial_writes = (0..n_required_key + additional_initial_capacity)
            .map(|i| Action::Write {
                key: i.to_be_bytes().to_vec(),
                value: Some(vec![64u8; 32]),
            })
            .collect();

        let read_actions = (0..n_reads).map(|i| Action::Read { key: rand_key(i) });
        let write_actions = (n_reads..n_reads + n_writes).map(|i| Action::Write {
            key: rand_key(i),
            value: Some(vec![8; 16]),
        });

        let delete_actions = (0..n_deletes).map(|i| Action::Write {
            key: i.to_be_bytes().to_vec(),
            value: None,
        });
        let update_actions = (n_deletes..n_deletes + n_updates).map(|i| Action::Write {
            key: i.to_be_bytes().to_vec(),
            value: Some(vec![2; 16]),
        });

        let custom_actions = read_actions
            .chain(write_actions)
            .chain(delete_actions)
            .chain(update_actions)
            .collect();

        Ok(Self {
            init_actions: initial_writes,
            run_actions: custom_actions,
        })
    }
}

impl Workload for CustomWorkload {
    fn init(&self, backend: &mut Box<dyn Db>) {
        backend.apply_actions(self.init_actions.clone(), None);
    }
    fn run(&self, backend: &mut Box<dyn Db>, timer: &mut Timer) {
        backend.apply_actions(self.run_actions.clone(), Some(timer));
    }
}

fn rand_key(id: u64) -> Vec<u8> {
    // keys must be uniformly distributed
    use rand::{RngCore as _, SeedableRng as _};
    let mut seed = [0; 16];
    seed[0..8].copy_from_slice(&id.to_le_bytes());
    let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);
    rng.next_u32().to_le_bytes().to_vec()
}
