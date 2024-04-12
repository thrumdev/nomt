use crate::{
    backend::Db, custom_workload::CustomWorkload, timer::Timer, transfer_workload::TransferWorkload,
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
pub trait Workload {
    fn run(&self, backend: Box<dyn Db>, timer: &mut Timer);
}

pub fn parse(
    name: &str,
    size: u64,
    additional_initial_capacity: u64,
    percentage_cold_transfer: Option<u8>,
) -> Result<Box<dyn Workload>> {
    Ok(match name {
        "transfer" => Box::new(TransferWorkload {
            size,
            percentage_cold_transfer: percentage_cold_transfer.unwrap_or(0),
            additional_initial_capacity,
        }),
        "set_balance" | "heavy_write" => Box::new(CustomWorkload::new(
            0,   // reads
            100, // writes
            0,   // deletes
            0,   // updates
            size,
            additional_initial_capacity,
        )?),
        "heavy_read" => Box::new(CustomWorkload::new(
            90, // reads
            10, // writes
            0,  // deletes
            0,  // updates
            size,
            additional_initial_capacity,
        )?),
        "heavy_update" => Box::new(CustomWorkload::new(
            5,  // reads
            5,  // writes
            0,  // deletes
            90, // updates
            size,
            additional_initial_capacity,
        )?),
        "heavy_delete" => Box::new(CustomWorkload::new(
            5,  // reads
            5,  // writes
            90, // deletes
            0,  // updates
            size,
            additional_initial_capacity,
        )?),
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
