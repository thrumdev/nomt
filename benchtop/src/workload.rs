/// Workload abstracts the type of work the DB will have to deal with.
///
/// Generally, the operations on a DB could be:
/// + read
/// + write
/// + delete
/// + update
///
/// Each workload will set up the DB differently and reads and writes arbitrarily,
/// whether the key is not present or already present.
use crate::{backend::Transaction, custom_workload, transfer_workload};
use anyhow::Result;

/// An interface for generating new sets of actions.
pub trait Workload {
    /// Run the workload against the given database transaction.
    ///
    /// Workloads may be run repeatedly and should vary from run to run.
    fn run(&mut self, transaction: &mut dyn Transaction);

    /// Get the size of the workload.
    fn size(&self) -> usize;
}

/// A database initialization workload where a number of keys have the same initial value.
pub struct Init {
    pub keys: Vec<Vec<u8>>,
    pub value: Vec<u8>,
}

impl Workload for Init {
    fn run(&mut self, transaction: &mut dyn Transaction) {
        for key in &self.keys {
            transaction.write(key, Some(&self.value));
        }
    }

    fn size(&self) -> usize {
        self.keys.len()
    }
}

pub fn parse(
    name: &str,
    workload_size: u64,
    db_size: u64,
    percentage_cold_transfer: Option<u8>,
) -> Result<(Init, Box<dyn Workload>)> {
    Ok(match name {
        "transfer" => (
            transfer_workload::init(db_size),
            Box::new(transfer_workload::build(
                db_size,
                workload_size,
                percentage_cold_transfer.unwrap_or(0),
            )),
        ),
        "randw" => (
            custom_workload::init(db_size),
            Box::new(custom_workload::build(0, 100, workload_size, db_size)),
        ),
        "randr" => (
            custom_workload::init(db_size),
            Box::new(custom_workload::build(100, 0, workload_size, db_size)),
        ),
        "randrw" => (
            custom_workload::init(db_size),
            Box::new(custom_workload::build(50, 50, workload_size, db_size)),
        ),
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
