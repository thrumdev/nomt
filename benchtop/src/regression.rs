use crate::{bench, cli::regression::Params, timer::pretty_display_ns, workload, Backend};
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

#[derive(serde::Deserialize, Debug)]
struct RegressionInputs {
    workloads: BTreeMap<String, WorkloadInfo>,
}

#[derive(serde::Deserialize, Debug)]
struct WorkloadInfo {
    name: String,
    size: u64,
    initial_capacity: u64,
    fetch_concurrency: u64,
    isolate: Option<Isolate>,
    sequential: Option<Sequential>,
}

#[derive(serde::Deserialize, Debug)]
pub struct Isolate {
    iterations: u64,
    mean: u64,
}

#[derive(serde::Deserialize, Debug)]
pub struct Sequential {
    time_limit: Option<u64>,
    op_limit: Option<u64>,
    mean: u64,
}

pub fn regression(params: Params) -> Result<()> {
    // load toml file
    let input = std::fs::read_to_string(params.input_file)
        .map_err(|_| anyhow!("regression input file does not exists"))?;

    // parse toml file
    let regr_inputs: RegressionInputs =
        toml::from_str(&input).map_err(|e| anyhow!("regression input file wrong format: {}", e))?;

    for (workload_name, workload_info) in regr_inputs.workloads {
        println!("\nExecuting Workload: {}\n", workload_name);

        if let Some(Isolate {
            iterations,
            mean: prev_mean,
        }) = workload_info.isolate
        {
            let (init, workload) = workload::parse(
                workload_info.name.as_str(),
                workload_info.size,
                1u64 << workload_info.initial_capacity,
                None, // TODO: support cold percentage
            )?;

            print!("Isolate: -");
            let bench_results = bench::bench_isolate(
                init,
                workload,
                vec![Backend::Nomt],
                iterations,
                false,
                workload_info.fetch_concurrency as usize,
            )?;
            let mean = *bench_results.first().expect("There must be nomt results");
            print_results(prev_mean, mean);
        };

        if let Some(Sequential {
            op_limit,
            time_limit,
            mean: prev_mean,
        }) = workload_info.sequential
        {
            let (init, workload) = workload::parse(
                workload_info.name.as_str(),
                workload_info.size,
                1u64 << workload_info.initial_capacity,
                None, // TODO: support cold percentage
            )?;

            print!("Sequential - ");
            let bench_results = bench::bench_sequential(
                init,
                workload,
                vec![Backend::Nomt],
                op_limit,
                time_limit,
                false,
                workload_info.fetch_concurrency as usize,
            )?;
            let mean = *bench_results.first().expect("There must be nomt results");
            print_results(prev_mean, mean);
        };
    }

    Ok(())
}

fn print_results(prev_mean: u64, mean: u64) {
    let results = format!(
        "\n  Previous mean:  {}\n  Current mean:  {}",
        pretty_display_ns(prev_mean),
        pretty_display_ns(mean)
    );

    if prev_mean < mean {
        println!("Regression {}", results);
    } else if prev_mean == mean {
        println!("Nothing changed");
    } else {
        println!("Improvement {}", results);
        println!("  Current mean nanoseconds: {}", mean);
    }
}
