use crate::{backend::Backend, cli::bench::Params, timer::Timer, workload::Workload};
use anyhow::Result;

pub fn bench(params: Params) -> Result<()> {
    let workload = Workload::parse(
        params.workload.name.as_str(),
        params.workload.size,
        params
            .workload
            .initial_capacity
            .and_then(|s| Some(1u64 << s))
            .unwrap_or(0),
    )?;

    let backends = if params.backends.is_empty() {
        Backend::all_backends()
    } else {
        params.backends
    };

    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));

        for _ in 0..params.iteration {
            let backend_instance = backend.instantiate();

            // it's up to the workload implementation to measure the relevant parts
            workload.run(backend_instance, &mut timer);
        }

        timer.print();
    }
    Ok(())
}
