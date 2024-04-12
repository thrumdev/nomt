mod backend;
mod bench;
mod cli;
mod custom_workload;
mod nomt;
mod profile;
mod sov_db;
mod timer;
mod transfer_workload;
mod workload;

use clap::Parser;
use cli::{Cli, Commands};

pub fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench(params) => bench::bench(params),
        Commands::Profile(params) => profile::profile(params),
        Commands::Exec(params) => profile::exec(params),
    }
}
