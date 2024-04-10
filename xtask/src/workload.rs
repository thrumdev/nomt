use crate::{backend::Db, timer::Timer, transfer_workload::TransferWorkload};
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

// The custom workload will follow these rules:
// 1. Reads and writes are randomly and uniformly distributed over the key space.
// 2. Deletes and updates will be performed on already present keys.
// 3. additional_initial_capacity represents the amount of items already present
//     in the DB in addition to the ones required to perform deletes and updates.
// 4. Size represents the total number of operations, where reads, writes, etc
//     are numbers that need to sum to 100 and represent a percentage of the total size.
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct CustomWorkload {
    reads: u8,
    writes: u8,
    delete: u8,
    update: u8,
    size: u64,
    additional_initial_capacity: u64,
}

impl Workload for CustomWorkload {
    fn run(&self, _backend: Box<dyn Db>, _timer: &mut Timer) {
        todo!()
    }
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
        "set_balance" => Box::new(CustomWorkload {
            reads: 0,
            writes: 100,
            delete: 0,
            update: 0,
            size,
            additional_initial_capacity,
        }),
        "heavy_read" => Box::new(CustomWorkload {
            reads: 10,
            writes: 90,
            delete: 0,
            update: 0,
            size,
            additional_initial_capacity,
        }),
        "heavy_update" => Box::new(CustomWorkload {
            reads: 5,
            writes: 5,
            delete: 0,
            update: 90,
            size,
            additional_initial_capacity,
        }),
        "heavy_delete" => Box::new(CustomWorkload {
            reads: 5,
            writes: 5,
            delete: 90,
            update: 0,
            size,
            additional_initial_capacity,
        }),
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
