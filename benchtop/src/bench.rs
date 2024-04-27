use crate::{backend::Backend, cli::bench::BenchType, timer::Timer, workload, workload::Workload};
use anyhow::Result;

pub fn bench(bench_type: BenchType) -> Result<()> {
    let common_params = match bench_type {
        BenchType::Isolate(ref params) => &params.common_params,
        BenchType::Sequential(ref params) => &params.common_params,
    };

    let workload = workload::parse(
        common_params.workload.name.as_str(),
        common_params.workload.size,
        common_params
            .workload
            .initial_capacity
            .map(|s| 1u64 << s)
            .unwrap_or(0),
        common_params.workload.percentage_cold,
    )?;

    let backends = if common_params.backends.is_empty() {
        Backend::all_backends()
    } else {
        common_params.backends.clone()
    };

    match bench_type {
        BenchType::Isolate(params) => {
            bench_isolate(workload, backends, params.iterations, true).map(|_| ())
        }
        BenchType::Sequential(params) => {
            bench_sequential(workload, backends, params.op_limit, params.time_limit, true)
                .map(|_| ())
        }
    }
}

// Benchmark the workload across multiple backends multiple times.
// Each iteration will be executed on a freshly initialized database.
//
// Return the mean execution time of the workloads for each backends
// in the order the backends are provided
pub fn bench_isolate(
    workload: Workload,
    backends: Vec<Backend>,
    iterations: u64,
    print: bool,
) -> Result<Vec<u64>> {
    let mut mean_results = vec![];
    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));
        let mut backend_instance = backend.instantiate(true);

        workload.init(&mut backend_instance);

        for _ in 0..iterations {
            workload.run(&mut backend_instance, Some(&mut timer), true);
        }

        if print {
            timer.print();
        }
        mean_results.push(timer.get_mean_workload_duration()?);
    }

    Ok(mean_results)
}

// Benchmark the workload across multiple backends multiple times.
// Each iteration will be executed on the same db repeatedly
// without clearing it until a time or operation count limit is reaced.
//
// Return the mean execution time of the workloads for each backends
// in the order the backends are provided
pub fn bench_sequential(
    workload: Workload,
    backends: Vec<Backend>,
    op_limit: Option<u64>,
    time_limit: Option<u64>,
    print: bool,
) -> Result<Vec<u64>> {
    if let (None, None) = (op_limit, time_limit) {
        anyhow::bail!("You need to specify at least one limiter between operations and time")
    }

    let mut mean_results = vec![];

    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));
        let mut backend_instance = backend.instantiate(true);

        let mut elapsed_time = 0;
        let mut op_count = 0;

        workload.init(&mut backend_instance);

        loop {
            workload.run(&mut backend_instance, Some(&mut timer), false);

            // check if time limit exceeded
            elapsed_time += timer.get_last_workload_duration()?;
            match time_limit {
                Some(limit) if elapsed_time >= (limit * 1000000) => break,
                _ => (),
            };

            // check if op limit exceeded
            op_count += workload.run_actions.len() as u64;
            match op_limit {
                Some(limit) if op_count >= limit => break,
                _ => (),
            };
        }

        if print {
            timer.print();
        }
        mean_results.push(timer.get_mean_workload_duration()?);
    }
    Ok(mean_results)
}
