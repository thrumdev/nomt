mod backend;
mod bench;
mod cli;
mod nomt;
mod sov_db;
mod timer;
mod workload;

use clap::Parser;
use cli::{Cli, Commands};

pub fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench(params) => bench::bench(params),
    }
}
