use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

mod meta_map;
mod sim;
mod store;

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
            num_readers,
            num_rings,
            pages_to_use,
            workload_size,
            cold_rate,
            preload_count,
            load_extra_rate,
            update_rate,
        } => {
            let (store, meta_map) = match store::Store::open(file) {
                Ok(x) => x,
                Err(e) => {
                    println!("encountered error in opening store: {e:?}");
                    return;
                }
            };

            let sim_params = sim::Params {
                num_workers: num_readers,
                num_pages: pages_to_use,
                num_rings,
                workload_size,
                cold_rate,
                preload_count,
                load_extra_rate,
                page_item_update_rate: update_rate,
            };

            sim::run_simulation(Arc::new(store), sim_params, meta_map);
        }
    }
}
