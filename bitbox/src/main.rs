use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use wal::ConsistencyError;

mod beatree;
mod io;
mod meta_map;
mod sim;
mod store;
mod wal;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Init {
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
        #[arg(short, long)]
        num_pages: usize,
    },
    Run {
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
        #[arg(short, long, value_name = "WAL_FILE")]
        wal_file: PathBuf,
        #[arg(long, default_value_t = 3)]
        num_readers: usize,
        #[arg(long)]
        num_rings: usize,
        #[arg(long)]
        pages_to_use: usize,
        #[arg(long)]
        workload_size: usize,
        #[arg(long, default_value_t = 0.0)]
        cold_rate: f32,
        #[arg(long, default_value_t = 3)]
        preload_count: usize,
        #[arg(long, default_value_t = 0.05)]
        load_extra_rate: f32,
        #[arg(long, default_value_t = 0.08)]
        update_rate: f32,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Init { file, num_pages } => {
            if let Err(e) = store::create(file.clone(), num_pages) {
                println!("encountered error in creation {e:?}");
                let _ = std::fs::remove_file(file);
            }
        }
        Command::Run {
            file,
            wal_file,
            num_readers,
            num_rings,
            pages_to_use,
            workload_size,
            cold_rate,
            preload_count,
            load_extra_rate,
            update_rate,
        } => {
            let (store, meta_page, meta_map) = match store::Store::open(file) {
                Ok(x) => x,
                Err(e) => {
                    println!("encountered error in opening store: {e:?}");
                    return;
                }
            };

            // Open the WAL, check its integrity and make sure the store is consistent with it
            let wal = wal::WalChecker::open_and_recover(wal_file.clone());
            let pending_batch = match wal.check_consistency(meta_page.sequence_number()) {
                Ok(()) => {
                    println!(
                        "Wal and Store are consistent, last sequence number: {}",
                        meta_page.sequence_number()
                    );
                    None
                }
                Err(ConsistencyError::LastBatchCrashed(crashed_batch)) => {
                    println!(
                        "Wal and Store are not consistent, pending sequence number: {}",
                        meta_page.sequence_number() + 1
                    );
                    Some(crashed_batch)
                }
                Err(ConsistencyError::NotConsistent(wal_seqn)) => {
                    // This is useful for testing. If the WAL sequence number is zero, it means the WAL is empty.
                    // For example, it could have been deleted, and it's okay to continue working on the store
                    // by appending new batches to the new WAL
                    if wal_seqn == 0 {
                        None
                    } else {
                        panic!(
                            "Store and Wal have two inconsistent serial numbers. wal: {}, store: {}",
                            wal_seqn,
                            meta_page.sequence_number()
                        );
                    }
                }
            };

            // Create a WalWriter, able to append new batch and prune older ones
            let wal = match wal::WalWriter::open(wal_file) {
                Ok(x) => x,
                Err(e) => {
                    println!("encountered error in opening wal: {e:?}");
                    return;
                }
            };

            // run simulation
            let sim_params = sim::Params {
                num_workers: num_readers,
                num_pages: pages_to_use,
                num_rings,
                workload_size,
                cold_rate,
                preload_count,
                load_extra_rate,
                page_item_update_rate: update_rate,
                pending_batch,
            };

            sim::run_simulation(Arc::new(store), meta_page, wal, sim_params, meta_map);
        }
    }
}
