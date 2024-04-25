use crate::{
    backend::Backend,
    cli::bench::{BenchType, CommonParams, IsolateParams, SequentialParams},
    timer::Timer,
    workload,
    workload::Workload,
};
use anyhow::Result;

pub fn bench(bench_type: BenchType) -> Result<()> {
    match bench_type {
        BenchType::Isolate(params) => bench_isolate(params),
        BenchType::Sequential(params) => bench_sequential(params),
    }
}

fn get_workload_and_backends(params: CommonParams) -> Result<(Workload, Vec<Backend>)> {
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

    let backends = if params.backends.is_empty() {
        Backend::all_backends()
    } else {
        params.backends
    };

    Ok((workload, backends))
}

pub fn bench_isolate(params: IsolateParams) -> Result<()> {
    let (workload, backends) = get_workload_and_backends(params.common_params)?;

    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));
        let mut backend_instance = backend.instantiate(true);

        workload.init(&mut backend_instance);

        for _ in 0..params.iterations {
            workload.run(&mut backend_instance, Some(&mut timer), true);
        }

        timer.print();
    }

    Ok(())
}

pub fn bench_sequential(params: SequentialParams) -> Result<()> {
    let (workload, backends) = get_workload_and_backends(params.common_params)?;

    if let (None, None) = (params.op_limit, params.time_limit) {
        anyhow::bail!("You need to specify at least one limiter between operations and time")
    }

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
            match params.time_limit {
                Some(limit) if elapsed_time >= (limit * 1000000) => break,
                _ => (),
            };

            // check if op limit exceeded
            op_count += workload.run_actions.len() as u64;
            match params.op_limit {
                Some(limit) if op_count >= limit => break,
                _ => (),
            };
        }

        timer.print();
    }
    Ok(())
}
