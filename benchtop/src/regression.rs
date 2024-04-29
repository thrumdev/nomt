mod default_input;

use crate::{bench, cli::regression::Params, timer::pretty_display_ns, workload, Backend};
use anyhow::{anyhow, Result};
use std::collections::BTreeMap;

#[derive(serde::Deserialize, serde::Serialize, Debug)]
struct RegressionInputs {
    workloads: BTreeMap<String, WorkloadInfo>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
struct WorkloadInfo {
    name: String,
    size: u64,
    initial_capacity: u64,
    isolate: Option<Isolate>,
    sequential: Option<Sequential>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Isolate {
    iterations: u64,
    mean: Option<u64>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Sequential {
    time_limit: Option<u64>,
    op_limit: Option<u64>,
    mean: Option<u64>,
}

pub fn regression(params: Params) -> Result<()> {
    let mut regr_inputs: RegressionInputs = match &params.input_file {
        Some(input_path) => {
            // load toml file
            let input = std::fs::read_to_string(input_path)
                .map_err(|_| anyhow!("regression input file does not exists"))?;

            // parse toml file
            toml::from_str(&input)
                .map_err(|e| anyhow!("regression input file wrong format: {}", e))?
        }
        None => default_input::default_regression_input(),
    };

    for (workload_name, workload_info) in &mut regr_inputs.workloads {
        println!("\nExecuting Workload: {}\n", workload_name);

        if let Some(Isolate {
            iterations,
            mean: ref mut prev_mean,
        }) = workload_info.isolate
        {
            let (init, workload) = workload::parse(
                workload_info.name.as_str(),
                workload_info.size,
                1u64 << workload_info.initial_capacity,
                None, // TODO: support cold percentage
            )?;

            print!("Isolate: ");
            let bench_results =
                bench::bench_isolate(init, workload, vec![Backend::Nomt], iterations, false)?;
            let mean = *bench_results.first().expect("There must be nomt results");
            print_results(prev_mean, mean);
            *prev_mean = Some(mean);
        };

        if let Some(Sequential {
            op_limit,
            time_limit,
            mean: ref mut prev_mean,
        }) = workload_info.sequential
        {
            let (init, workload) = workload::parse(
                workload_info.name.as_str(),
                workload_info.size,
                1u64 << workload_info.initial_capacity,
                None, // TODO: support cold percentage
            )?;

            print!("Sequential: ");
            let bench_results = bench::bench_sequential(
                init,
                workload,
                vec![Backend::Nomt],
                op_limit,
                time_limit,
                false,
            )?;
            let mean = *bench_results.first().expect("There must be nomt results");
            print_results(prev_mean, mean);
            *prev_mean = Some(mean);
        };
    }

    if let Some(output_path) = params.output_file {
        std::fs::write(output_path, toml::to_string(&regr_inputs)?)?;
    }

    Ok(())
}

fn print_results(maybe_prev_mean: &Option<u64>, curr_mean: u64) {
    let format_curr_mean = format!(
        "  Current mean:  {} ( {} [ns] )",
        pretty_display_ns(curr_mean),
        curr_mean
    );

    let Some(prev_mean) = *maybe_prev_mean else {
        println!("No previous mean specified");
        println!("{}", format_curr_mean);
        return;
    };

    let format_prev_mean = format!("  Previous mean:  {}", pretty_display_ns(prev_mean));

    if prev_mean < curr_mean {
        println!("Regression");
    } else if prev_mean == curr_mean {
        println!("Nothing changed");
    } else {
        println!("Improvement");
    }

    println!("{}", format_prev_mean);
    println!("{}", format_curr_mean);
}
