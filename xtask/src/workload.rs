use crate::{backend::Db, timer::Timer};
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
// The custom workload will follow these rules:
// 1. Reads and writes are randomly and uniformly distributed over the key space.
// 2. Deletes and updates will be performed on already present keys.
// 3. additional_initial_capacity represents the amount of items already present
//     in the DB in addition to the ones required to perform deletes and updates.
// 4. Size represents the total number of operations, where reads, writes, etc
//     are numbers that need to sum to 100 and represent a percentage of the total size.
#[derive(Debug, Clone)]
pub enum Workload {
    // All transfers happen between different accounts to ensure a fair comparison.
    // In this way, the possible caching mechanism over writes would not benefit either backend.
    //
    // additional_initial_capacity is the amount of items in the storage in addition
    // to all accounts used to execute the transfer
    Transfer {
        size: u64,
        additional_initial_capacity: u64,
    },
    Custom {
        reads: u8,
        writes: u8,
        delete: u8,
        update: u8,
        size: u64,
        additional_initial_capacity: u64,
    },
}

impl Workload {
    // Create a workload over some provided type of workload
    pub fn parse(name: &str, size: u64, additional_initial_capacity: u64) -> Result<Self> {
        Ok(match name {
            "transfer" => Self::Transfer {
                size,
                additional_initial_capacity,
            },
            "set_balance" => Self::Custom {
                reads: 0,
                writes: 100,
                delete: 0,
                update: 0,
                size,
                additional_initial_capacity,
            },
            "heavy_read" => Self::Custom {
                reads: 10,
                writes: 90,
                delete: 0,
                update: 0,
                size,
                additional_initial_capacity,
            },
            "heavy_update" => Self::Custom {
                reads: 5,
                writes: 5,
                delete: 0,
                update: 90,
                size,
                additional_initial_capacity,
            },
            "heavy_delete" => Self::Custom {
                reads: 5,
                writes: 5,
                delete: 90,
                update: 0,
                size,
                additional_initial_capacity,
            },
            name => anyhow::bail!("invalid workload name: {}", name),
        })
    }
}

impl Workload {
    pub fn run(&self, _backend: Box<dyn Db>, _timer: &mut Timer) {
        match self {
            Workload::Custom { .. } => todo!(),
            Workload::Transfer { .. } => todo!(),
        }
    }
}
