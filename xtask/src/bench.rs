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
            let mut backend_instance = backend.instantiate();

            workload.init(&mut backend_instance);
            // it's up to the workload implementation to measure the relevant parts
            workload.run(&mut backend_instance, &mut timer);
        }

        timer.print();
    }
    Ok(())
}
