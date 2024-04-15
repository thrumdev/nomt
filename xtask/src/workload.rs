use crate::{
    backend::{Action, Db},
    custom_workload::new_custom_workload,
    timer::Timer,
    transfer_workload::new_transfer_workload,
};
use anyhow::Result;

// Workload abstracts the type of work the DB will have to deal with.
//
// Generally, the operations on a DB could be:
// + read
// + write
// + delete
// + update
//
// Each workload will set up the DB differently and reads and writes arbitrarily,
// whether the key is not present or already present.
pub struct Workload {
    pub init_actions: Vec<Action>,
    pub run_actions: Vec<Action>,
}

impl Workload {
    pub fn init(&self, backend: &mut Box<dyn Db>) {
        backend.apply_actions(self.init_actions.clone(), None);
    }

    pub fn run(&self, backend: &mut Box<dyn Db>, timer: Option<&mut Timer>) {
        backend.apply_actions(self.run_actions.clone(), timer);
    }
}

pub fn parse(
    name: &str,
    size: u64,
    additional_initial_capacity: u64,
    percentage_cold_transfer: Option<u8>,
) -> Result<Workload> {
    Ok(match name {
        "transfer" => new_transfer_workload(
            size,
            percentage_cold_transfer.unwrap_or(0),
            additional_initial_capacity,
        ),
        "randw" => new_custom_workload(
            0,     // reads
            100,   // writes
            0,     // deletes
            0,     // updates
            false, // seq
            size,
            additional_initial_capacity,
        )?,
        "randr" => new_custom_workload(
            100,   // reads
            0,     // writes
            0,     // deletes
            0,     // updates
            false, // seq
            size,
            additional_initial_capacity,
        )?,
        "seqw" => new_custom_workload(
            0,    // reads
            100,  // writes
            0,    // deletes
            0,    // updates
            true, // seq
            size,
            additional_initial_capacity,
        )?,
        "seqr" => new_custom_workload(
            100,  // reads
            0,    // writes
            0,    // deletes
            0,    // updates
            true, // seq
            size,
            additional_initial_capacity,
        )?,
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
