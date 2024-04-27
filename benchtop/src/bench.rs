use crate::{backend::Backend, cli::bench::Params, timer::Timer, workload};
use anyhow::Result;

pub fn bench(params: Params) -> Result<()> {
    let workloads: Vec<_> = (0..params.iteration)
        .map(|_| {
            workload::parse(
                params.workload.name.as_str(),
                params.workload.size,
                params
                    .workload
                    .initial_capacity
                    .map(|s| 1u64 << s)
                    .unwrap_or(0),
                params.workload.percentage_cold,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let backends = if params.backends.is_empty() {
        Backend::all_backends()
    } else {
        params.backends
    };

    for backend in backends {
        let mut timer = Timer::new(format!("{}", backend));

        let mut backend_instance = backend.instantiate(true);

        if let Some(workload) = workloads.first() {
            workload.init(&mut backend_instance);
        }

        for workload in &workloads {
            workload.run(&mut backend_instance, Some(&mut timer));
        }

        timer.print();
    }
    Ok(())
}
