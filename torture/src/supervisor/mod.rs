//! The supervisor part. Spawns and manages agents. Assigns work to agents.

use std::{path::PathBuf, process::exit};

use anyhow::Result;
use clap::Parser;
use cli::{Cli, WorkloadParams};
use tokio::{
    signal::unix::{signal, SignalKind},
    task::{self, JoinHandle},
};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, trace_span, warn, Instrument};

use workload::Workload;

mod cli;
mod comms;
mod controller;
mod workload;

/// The entrypoint for the supervisor part of the program.
///
/// This is not expected to return explicitly unless there was an error.
pub async fn run() -> Result<()> {
    crate::logging::init_supervisor();

    let cli = Cli::parse();
    let seed = cli.seed.unwrap_or_else(rand::random);

    // Create a cancellation token and spawn the control loop task passing the token to it and
    // wait until the control loop task finishes or the user interrupts it.
    let ct = CancellationToken::new();
    let control_loop_jh = task::spawn(control_loop(
        ct.clone(),
        seed,
        cli.workload_params,
        cli.flag_limit,
    ));
    match join_interruptable(control_loop_jh, ct).await {
        ExitReason::Finished => {
            exit(0);
        }
        ExitReason::Error(error) => {
            error!("Error occured: {:?}", error);
            exit(1);
        }
        ExitReason::Panic(panic_box) => {
            // The `Any` payload from a panic
            let panic_string = if let Some(s) = panic_box.downcast_ref::<&str>() {
                Some(s.to_string())
            } else if let Some(s) = panic_box.downcast_ref::<String>() {
                Some(s.clone())
            } else {
                None
            };
            if let Some(s) = panic_string {
                error!("Panic occured: {}", s);
            } else {
                error!("Panic occured (no message)");
            }
            exit(2);
        }
        ExitReason::Interrupted => {
            // The exit code is 128 + 2 because in UNIX, signals are mapped to exit codes by adding
            // 128 to the signal number. SIGINT is signal number 2, so the exit code is:
            //
            //     128 + 2 = 130
            //
            // This is what `cargo test` does.
            exit(130);
        }
    }
}

enum ExitReason {
    /// The `control_loop` task finished successfully.
    Finished,
    /// The `control_loop` task returned an error.
    Error(anyhow::Error),
    /// Similiar to `Error` but the error is a panic.
    Panic(Box<dyn std::any::Any + Send + 'static>),
    /// The `control_loop` task was interrupted by the user.
    Interrupted,
}

/// Wait for the control loop to finish or for the user to interrupt it. Returns the reason for the
/// exit.
///
/// The handling the ^C signal is done as follows:
///
/// 1. First ^C emits the `CancellationToken::cancel` call.
/// 2. Second ^C aborst the control loop task.
async fn join_interruptable(
    mut control_loop_jh: JoinHandle<Result<()>>,
    ct: CancellationToken,
) -> ExitReason {
    // UNWRAP: we don't expect this to fail.
    //         If that happens then something is terribly wrong anyway.
    let mut sigint = signal(SignalKind::interrupt()).unwrap();

    // Either catch the first ^C signal or wait until the succesful completion of the control loop.
    tokio::select! {
        _ = sigint.recv() => {
            info!("Received ^C...");
            ct.cancel();
        }
        res = &mut control_loop_jh => {
            // Control loop ended before receiving ^C.
            match res {
                Ok(Ok(())) => return ExitReason::Finished,
                Ok(Err(e)) => return ExitReason::Error(e),
                Err(join_err) => {
                    // UNWRAP: the task cannot be cancelled because the only place it can get
                    // cancelled is here and we don't cancel it. So it must be a panic.
                    let panic_box = join_err.try_into_panic().unwrap();
                    return ExitReason::Panic(panic_box);
                },
            }
        }
    }

    // At this point we received the first ^C signal and the control loop is still running.
    tokio::select! {
        _ = sigint.recv() => {
            control_loop_jh.abort();
            return ExitReason::Interrupted;
        }
        res = &mut control_loop_jh => {
            // Control loop ended before receiving the second ^C.
            match res {
                Ok(Ok(())) => {
                    // We received the first interrupt meaning that the control loop was cancelled.
                    return ExitReason::Interrupted
                },
                Ok(Err(e)) => return ExitReason::Error(e),
                Err(join_err) => {
                    // UNWRAP: the task cannot be cancelled because the only place it can get
                    // cancelled is here and we don't cancel it. So it must be a panic.
                    let panic_box = join_err.try_into_panic().unwrap();
                    return ExitReason::Panic(panic_box);
                },
            }
        }
    }
}

#[derive(Debug)]
pub struct InvestigationFlag {
    seed: u64,
    workload_id: u64,
    /// The directory the agent was working in.
    workdir: PathBuf,
    /// The reason for flagging.
    reason: anyhow::Error,
}

/// Run the workload until either it either finishes, errors or gets cancelled.
///
/// Returns `None` if the investigation is not required (i.e. cancelled or succeeded), otherwise,
/// returns the investigation report.
async fn run_workload(
    cancel_token: CancellationToken,
    seed: u64,
    workload_params: &WorkloadParams,
    workload_id: u64,
) -> Result<Option<InvestigationFlag>> {
    // This creates a temp dir for the working dir of the workload.
    let workdir = tempfile::Builder::new()
        .prefix("torture-")
        .suffix(format!("-workload-{}", workload_id).as_str())
        .tempdir()
        .expect("Failed to create a temp dir");
    let mut workload = Workload::new(seed, workdir, workload_params, workload_id);
    let result = workload.run(cancel_token).await;
    match result {
        Ok(()) => Ok(None),
        Err(err) => Ok(Some(InvestigationFlag {
            seed,
            workload_id,
            workdir: workload.into_workdir().into_path(),
            reason: err,
        })),
    }
}

fn print_flag(flag: &InvestigationFlag) {
    warn!(
        "Flagged for investigation:\n seed={seed}\n workload_id={workload_id}\n  workdir={workdir}\n  reason={reason}",
        seed = flag.seed,
        workload_id = flag.workload_id,
        workdir = flag.workdir.display(),
        reason = flag.reason,
    );
}

const NON_DETERMINISM_DISCLAIMER: &str = "torture is a non-deterministic fuzzer.\nThe workload \
    generated by this tool is not guaranteed to be reproducible, but specifying the seed may \
    improve the chances of reproducing the issue.";

/// Run the control loop creating and tearing down the agents.
///
/// `cancel_token` is used to gracefully shutdown the supervisor.
async fn control_loop(
    cancel_token: CancellationToken,
    seed: u64,
    workload_params: WorkloadParams,
    flag_num_limit: usize,
) -> Result<()> {
    let mut flags = Vec::new();
    let mut workload_cnt = 0;
    // TODO: Run workloads in parallel. Make the concurrency factor configurable.
    info!("Starting control loop, seed={seed}.\n{NON_DETERMINISM_DISCLAIMER}");
    loop {
        let workload_id = workload_cnt;
        let workload_seed = seed + workload_cnt;
        workload_cnt += 1;
        let maybe_flag = run_workload(
            cancel_token.clone(),
            workload_seed,
            &workload_params,
            workload_id,
        )
        .instrument(trace_span!("workload", workload_id))
        .await?;
        if let Some(flag) = maybe_flag {
            print_flag(&flag);
            flags.push(flag);
        }
        if cancel_token.is_cancelled() {
            break;
        }
        if flags.len() >= flag_num_limit {
            info!("Flag limit reached. Exiting.");
            break;
        }
    }
    for flag in flags {
        print_flag(&flag);
    }
    Ok(())
}
