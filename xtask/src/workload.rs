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
        "set_balance" | "heavy_write" => new_custom_workload(
            0,   // reads
            100, // writes
            0,   // deletes
            0,   // updates
            size,
            additional_initial_capacity,
        )?,
        "heavy_read" => new_custom_workload(
            90, // reads
            10, // writes
            0,  // deletes
            0,  // updates
            size,
            additional_initial_capacity,
        )?,
        "heavy_update" => new_custom_workload(
            5,  // reads
            5,  // writes
            0,  // deletes
            90, // updates
            size,
            additional_initial_capacity,
        )?,
        "heavy_delete" => new_custom_workload(
            5,  // reads
            5,  // writes
            90, // deletes
            0,  // updates
            size,
            additional_initial_capacity,
        )?,
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
