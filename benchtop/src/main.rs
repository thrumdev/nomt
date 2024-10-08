mod backend;
mod cli;
mod custom_workload;
mod nomt;
mod sov_db;
mod sp_trie;
mod timer;
mod transfer_workload;
mod workload;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, InitParams, RunParams};
use timer::Timer;

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init(params) => init(params),
        Commands::Run(params) => run(params),
    }
}

pub fn init(params: InitParams) -> Result<()> {
    let workload_params = params.workload;
    let (mut init, _) = workload::parse(
        workload_params.name.as_str(),
        workload_params.size,
        workload_params
            .initial_capacity
            .map(|s| 1u64 << s)
            .unwrap_or(0),
        workload_params.fresh,
        u64::max_value(),
    )?;

    let mut db = params.backend.instantiate(
        true,
        workload_params.commit_concurrency,
        workload_params.io_workers,
        workload_params.hashtable_buckets,
    );
    db.execute(None, &mut *init, None);

    Ok(())
}

pub fn run(params: RunParams) -> Result<()> {
    let workload_params = params.workload;
    let (mut init, mut workload) = workload::parse(
        workload_params.name.as_str(),
        workload_params.size,
        workload_params
            .initial_capacity
            .map(|s| 1u64 << s)
            .unwrap_or(0),
        workload_params.fresh,
        params.limits.ops.unwrap_or(u64::max_value()),
    )?;

    let mut db = params.backend.instantiate(
        params.reset,
        workload_params.commit_concurrency,
        workload_params.io_workers,
        workload_params.hashtable_buckets,
    );

    if params.reset {
        db.execute(None, &mut *init, None);
    }

    let mut timer = Timer::new(format!("{}", params.backend));
    let timeout = params
        .limits
        .time
        .map(|time_limit| std::time::Instant::now() + time_limit.into());
    db.execute(Some(&mut timer), &mut *workload, timeout);

    db.print_metrics();
    timer.print();

    Ok(())
}
