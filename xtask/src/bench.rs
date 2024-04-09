use crate::{
    cli::{bench::Params, Backend},
    timer::Timer,
    Action,
};
use anyhow::Result;

fn rand_keys(len: usize) -> Vec<[u8; 32]> {
    // keys must be uniformly distributed, but we don't want to spend time on a good hash. So
    // the next best option is to use a seeded PRNG.
    use rand::{RngCore as _, SeedableRng as _};
    let mut keys: Vec<[u8; 32]> = vec![];
    for i in 0..len {
        let mut seed = [0; 16];
        seed[0..8].copy_from_slice(&i.to_le_bytes());
        let mut rng = rand_pcg::Lcg64Xsh32::from_seed(seed);
        let mut key = [0u8; 32];
        for i in 0..8 {
            let rand = rng.next_u32().to_be_bytes();
            key[i] = rand[0];
            key[i + 1] = rand[1];
            key[i + 2] = rand[2];
            key[i + 3] = rand[3];
        }
        keys.push(key);
    }
    keys
}

pub fn bench(mut params: Params) -> Result<()> {
    let actions = params.workload.get_actions(params.workload_size);

    if params.backend.is_empty() {
        params.backend = Backend::all_backends();
    }

    for backend in params.backend {
        let mut timer = Timer::new(format!("{}", backend));
        for _ in 0..params.iteration {
            let mut backend_instance = backend.new();

            if let Some(initial_size) = params.initial_size {
                // TODO: Make sure the inserted value does not matter here and
                // it is ok to insert the key as a value.
                let writes = rand_keys((initial_size as usize) << 2)
                    .into_iter()
                    .map(|k| (k.to_vec(), Some(k.to_vec())))
                    .collect();
                backend_instance.apply_actions(vec![Action::Writes(writes)]);
            }

            let bench_actions = actions.clone();
            {
                let _guard = timer.record();
                // TODO: Instead of measuring the entire function,
                // it is probably better to provide a timer to each
                // backend_instance so that each module
                // will measure each relevant part.
                // Or each function will return something like `TimerOutcome`
                // where all the more granular information will be stored
                backend_instance.apply_actions(bench_actions);
            }
        }
        timer.print();
    }
    Ok(())
}
