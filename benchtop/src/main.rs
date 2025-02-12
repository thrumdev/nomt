mod backend;
mod cli;
mod custom_workload;
mod nomt;
mod sov_db;
mod sp_trie;
mod timer;
mod transfer_workload;
mod workload;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, InitParams, RunParams};
use timer::Timer;

pub fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init(params) => init(params),
        Commands::Run(params) => run(params),
    }
}

pub fn init(params: InitParams) -> Result<()> {
    let workload_params = params.workload;
    let (mut init, _) = workload::parse(&workload_params, u64::max_value())?;

    let mut db = params.backend.instantiate(
        true,
        workload_params.commit_concurrency,
        workload_params.io_workers,
        workload_params.hashtable_buckets,
        workload_params.page_cache_size,
        workload_params.leaf_cache_size,
        workload_params.page_cache_upper_levels,
        workload_params.prepopulate_page_cache,
        0,
    );
    db.execute(None, &mut *init, None);

    Ok(())
}

pub fn run(params: RunParams) -> Result<()> {
    let workload_params = params.workload;
    let (mut init, mut workloads) = workload::parse(
        &workload_params,
        params.limits.ops.unwrap_or(u64::max_value()),
    )?;

    if params.reset {
        let mut db = params.backend.instantiate(
            params.reset,
            workload_params.commit_concurrency,
            workload_params.io_workers,
            workload_params.hashtable_buckets,
            workload_params.page_cache_size,
            workload_params.leaf_cache_size,
            workload_params.page_cache_upper_levels,
            workload_params.prepopulate_page_cache,
            workload_params.overlay_window_length,
        );
        db.execute(None, &mut *init, None);
    }

    let mut exec = move || -> Result<()> {
        let mut db = params.backend.instantiate(
            params.reset,
            workload_params.commit_concurrency,
            workload_params.io_workers,
            workload_params.hashtable_buckets,
            workload_params.page_cache_size,
            workload_params.leaf_cache_size,
            workload_params.page_cache_upper_levels,
            workload_params.prepopulate_page_cache,
            workload_params.overlay_window_length,
        );

        let mut timer = Timer::new(format!("{}", params.backend));
        let warmup_timeout = params
            .warm_up
            .map(|time_limit| std::time::Instant::now() + time_limit.into());

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|_| "benchtop-workload".into())
            .num_threads(workload_params.workload_concurrency as usize)
            .build()?;

        if let Some(t) = warmup_timeout {
            if workload_params.workload_concurrency == 1 {
                db.execute(Some(&mut timer), &mut *workloads[0], Some(t));
            } else {
                db.parallel_execute(Some(&mut timer), &thread_pool, &mut workloads, Some(t))?;
            };

            timer = Timer::new(format!("{}", params.backend));
        }

        let timeout = params
            .limits
            .time
            .map(|time_limit| std::time::Instant::now() + time_limit.into());

        if workload_params.workload_concurrency == 1 {
            db.execute(Some(&mut timer), &mut *workloads[0], timeout);
        } else {
            db.parallel_execute(Some(&mut timer), &thread_pool, &mut workloads, timeout)?;
        };

        db.print_metrics();
        timer.print(workload_params.size);
        print_max_rss();
        Ok(())
    };

    let n_iterations = 1;
    let mut exec_n_times = || -> Result<()> {
        for _ in 0..n_iterations {
            exec()?;
        }
        Ok(())
    };

    println!("\n\n --- NO FEATURES");
    exec_n_times()?;

    println!("\n\n --- IOPOLL");
    std::env::set_var("IOPOLL", "true");
    exec_n_times()?;
    std::env::remove_var("IOPOLL");

    println!("\n\n --- SQPOLL - 5ms idle");
    std::env::set_var("SQPOLL", "true");
    std::env::set_var("SQPOLL_IDLE", "5");
    exec_n_times()?;
    std::env::remove_var("SQPOLL");
    std::env::remove_var("SQPOLL_IDLE");

    println!("\n\n --- SQPOLL - 10ms idle");
    std::env::set_var("SQPOLL", "true");
    std::env::set_var("SQPOLL_IDLE", "10");
    exec_n_times()?;
    std::env::remove_var("SQPOLL");
    std::env::remove_var("SQPOLL_IDLE");

    println!("\n\n --- SQPOLL - 50ms idle");
    std::env::set_var("SQPOLL", "true");
    std::env::set_var("SQPOLL_IDLE", "50");
    exec_n_times()?;
    std::env::remove_var("SQPOLL");
    std::env::remove_var("SQPOLL_IDLE");

    println!("\n\n --- SINGLE_ISSUER");
    std::env::set_var("SINGLE_ISSUER", "true");
    exec_n_times()?;
    std::env::remove_var("SINGLE_ISSUER");

    println!("\n\n --- SINGLE_ISSUER + COOP_TASKRUN");
    std::env::set_var("SINGLE_ISSUER", "true");
    std::env::set_var("COOP_TASKRUN", "true");
    exec_n_times()?;
    std::env::remove_var("SINGLE_ISSUER");
    std::env::remove_var("COOP_TASKRUN");

    println!("\n\n --- SINGLE_ISSUER + DEFER_TASKRUN");
    std::env::set_var("SINGLE_ISSUER", "true");
    std::env::set_var("DEFER_TASKRUN", "true");
    exec_n_times()?;
    std::env::remove_var("SINGLE_ISSUER");
    std::env::remove_var("DEFER_TASKRUN");

    Ok(())
}

fn print_max_rss() {
    let max_rss = get_max_rss().unwrap_or(0);
    println!("max rss: {} MiB", max_rss / 1024);
    fn get_max_rss() -> Option<usize> {
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if ret == 0 {
            Some(usage.ru_maxrss as usize)
        } else {
            None
        }
    }
}
