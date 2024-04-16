use crate::{backend::Backend, cli::bench::Params, timer::Timer, workload};
use anyhow::Result;

pub fn bench(params: Params) -> Result<()> {
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

    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));

        for _ in 0..params.iteration {
            let mut backend_instance = backend.instantiate(true);

            // TODO: if the initial capacity is large, this repetition could become time-consuming.
            // It would be better to initialize the database once,
            // copy it to a location, and then only run the workload for each iteration
            workload.init(&mut backend_instance);
            // it's up to the workload implementation to measure the relevant parts
            workload.run(&mut backend_instance, Some(&mut timer));
        }

        timer.print();
    }
    Ok(())
}
