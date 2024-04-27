mod backend;
mod bench;
mod cli;
mod custom_workload;
mod nomt;
mod regression;
mod sov_db;
mod sp_trie;
mod timer;
mod transfer_workload;
mod workload;

use anyhow::Result;
use backend::Backend;
use clap::Parser;
use cli::{Cli, Commands, WorkloadParams};

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench(params) => bench::bench(params),
        Commands::Init(params) => init(params),
        Commands::Run(params) => run(params),
        Commands::Regression(params) => regression::regression(params),
    }
}

pub fn init(params: WorkloadParams) -> Result<()> {
    let workload = workload::parse(
        params.name.as_str(),
        params.size,
        params.initial_capacity.map(|s| 1u64 << s).unwrap_or(0),
        params.percentage_cold,
    )?;

    workload.init(&mut Backend::Nomt.instantiate(true));

    Ok(())
}

pub fn run(params: WorkloadParams) -> Result<()> {
    let workload = workload::parse(
        params.name.as_str(),
        params.size,
        params.initial_capacity.map(|s| 1u64 << s).unwrap_or(0),
        params.percentage_cold,
    )?;

    workload.run(&mut Backend::Nomt.instantiate(false), None, true);

    Ok(())
}
