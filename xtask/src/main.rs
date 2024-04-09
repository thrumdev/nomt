mod bench;
mod cli;
mod nomt;
mod sov_db;
mod timer;

use anyhow::Result;
use bench::bench;
use clap::Parser;
use cli::{Backend, Cli, Commands, Workload};
use nomt::NomtDB;
use sov_db::SovDB;
use std::rc::Rc;

pub trait DB {
    /// Apply the given actions to the storage, committing them
    /// to the database at the end.
    fn apply_actions(&mut self, actions: Vec<Action>);
}

type Value = Vec<u8>;
type Key = Vec<u8>;

#[derive(Clone)]
pub enum Action {
    /// Batch of writes into the storage,
    /// A value of None means delete that key
    Writes(Vec<(Key, Option<Value>)>),
    /// Batch of reads
    Reads(Vec<Key>),
}

impl Workload {
    // Given the workload and its size, translate them into a
    // vector of raw actions that need to be performed on the database
    pub fn get_actions(&self, size: u64) -> Vec<Action> {
        match self {
            Workload::SetBalance => {
                let mut accont_id: u64 = 0;
                let writes = std::iter::from_fn(move || {
                    accont_id += 1;

                    if accont_id < size {
                        Some((
                            accont_id.to_be_bytes().to_vec(),
                            // TODO: should be better to randomize the value?
                            Some(1000u128.to_be_bytes().to_vec()),
                        ))
                    } else {
                        None
                    }
                })
                .collect();
                vec![Action::Writes(writes)]
            }
            // The problem with transfer is that it needs a crafted
            // storage to work with, so accounts should already have
            // balance in the storage and the transfer should be made between two
            // already existing accounts.
            // TODO: move the bench logic into the workloads
            Workload::Transfer => todo!(),
        }
    }
}

impl Backend {
    fn all_backends() -> Vec<Self> {
        vec![Backend::SovDB, Backend::Nomt]
    }

    pub fn new(&self) -> Box<dyn DB> {
        match self {
            Backend::SovDB => Box::new(SovDB::new()),
            Backend::Nomt => Box::new(NomtDB::new()),
        }
    }
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench(params) => bench::bench(params),
    }
}
