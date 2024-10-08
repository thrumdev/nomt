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
pub trait Workload: Send {
    /// Run a step of the workload against the given database transaction.
    ///
    /// Workloads may be run repeatedly and should vary from run to run.
    fn run_step(&mut self, transaction: &mut dyn Transaction);

    /// Whether the workload is done.
    fn is_done(&self) -> bool;
}

pub fn parse(
    name: &str,
    workload_size: u64,
    db_size: u64,
    fresh: Option<u8>,
    op_limit: u64,
    threads: usize,
) -> Result<(Box<dyn Workload>, Vec<Box<dyn Workload>>)> {
    fn dyn_vec(v: Vec<impl Workload + 'static>) -> Vec<Box<dyn Workload>> {
        v.into_iter()
            .map(|w| Box::new(w) as Box<dyn Workload>)
            .collect()
    }

    Ok(match name {
        "transfer" => (
            Box::new(transfer_workload::init(db_size)),
            dyn_vec(transfer_workload::build(
                db_size,
                workload_size,
                fresh.unwrap_or(0),
                op_limit,
                threads,
            )),
        ),
        "randw" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(custom_workload::build(
                0,
                100,
                workload_size,
                fresh.unwrap_or(0),
                db_size,
                op_limit,
                threads,
            )),
        ),
        "randr" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(custom_workload::build(
                100,
                0,
                workload_size,
                fresh.unwrap_or(0),
                db_size,
                op_limit,
                threads,
            )),
        ),
        "randrw" => (
            Box::new(custom_workload::init(db_size)),
            dyn_vec(custom_workload::build(
                50,
                50,
                workload_size,
                fresh.unwrap_or(0),
                db_size,
                op_limit,
                threads,
            )),
        ),
        name => anyhow::bail!("invalid workload name: {}", name),
    })
}
