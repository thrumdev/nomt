use crate::{cli::profile::Params, workload};
use anyhow::Result;
use std::collections::VecDeque;
use std::process::Command;

pub fn profile(params: Params) -> Result<()> {
    // check for samply to be installed
    Command::new("sh")
        .args(["-c", "command", "-v", "samply"])
        .output()
        .map_err(|_| {
            anyhow::anyhow!("Install sampy (`cargo install samply`) to execute profiling")
        })?;

    let workload = workload::parse(
        params.workload.name.as_str(),
        params.workload.size,
        params
            .workload
            .initial_capacity
            .map(|s| 1u64 << s)
            .unwrap_or(0),
        params.workload.percentage_cold,
    )?;

    workload.init(&mut params.backend.instantiate(true));

    // TODO: make cwd independent
    let mut samply_args: VecDeque<String> = std::env::args().collect();
    samply_args[1] = "exec".to_string();
    samply_args.push_front("record".to_string());

    let mut profiler_command = Command::new("samply").args(samply_args).spawn()?;
    profiler_command.wait()?;

    Ok(())
}

pub fn exec(params: Params) -> Result<()> {
    // TODO: it would be super to avoid the construction
    // of the workload in the profiler
    // TODO: at least avoid constructing the init_actions that will not be used
    let workload = workload::parse(
        params.workload.name.as_str(),
        params.workload.size,
        params
            .workload
            .initial_capacity
            .map(|s| 1u64 << s)
            .unwrap_or(0),
        params.workload.percentage_cold,
    )?;

    let mut backend_instance = params.backend.instantiate(false);

    workload.run(&mut backend_instance, None);
    Ok(())
}
