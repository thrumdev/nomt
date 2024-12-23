//! The supervisor part. Spawns and manages agents. Assigns work to agents.

use std::{path::PathBuf, process::exit};

use crate::spawn::{self, Child};
use anyhow::{bail, Result};
use tempfile::TempDir;
use tokio::{
    signal::unix::{signal, SignalKind},
    task::{self, JoinHandle},
};
use tokio_util::sync::CancellationToken;

mod comms;
mod controller;

/// The entrypoint for the supervisor part of the program.
///
/// This is not expected to return explicitly unless there was an error.
pub async fn run() -> Result<()> {
    // TODO: this is the main entrypoint of the program and as such should handle a few things:
    //
    // 1. Parse command line arguments.
    // 2. Logging.

    // Create a cancellation token and spawn the control loop task passing the token to it and
    // wait until the control loop task finishes or the user interrupts it.
    let ct = CancellationToken::new();
    let control_loop_jh = task::spawn(control_loop(ct.clone()));
    match join_interruptable(control_loop_jh, ct).await {
        ExitReason::Finished => {
            exit(0);
        }
        ExitReason::Error(error) => {
            eprintln!("Error occured: {:?}", error);
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
                eprintln!("Panic occured: {}", s);
            } else {
                eprintln!("Panic occured (no message)");
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
pub enum FlagReason {
    Timeout,
}

pub struct InvestigationFlag {
    /// The directory the agent was working in.
    workdir: PathBuf,
    /// The reason for flagging.
    reason: FlagReason,
}

/// This is a struct that controls workload execution.
///
/// A workload is a set of tasks that the agents should perform. We say agents, plural, because
/// the same workload can be executed by multiple agents. Howeever, it's always sequential. This
/// arises from the fact that as part of the workload we need to crash the agent to check how
/// it behaves.
struct Workload {
    // TODO: state of the workload.
    //
    // The stuff that helps to generate the workload. For example, the keys that were inserted.
    /// Working directory for this particular workload.
    workdir: TempDir,
}

impl Workload {
    fn new(workdir: TempDir) -> Self {
        Self { workdir }
    }

    /// Run the workload.
    async fn run(&mut self, cancel_token: CancellationToken) -> Result<()> {
        // TODO: we will need to spawn an agent here for workload execution.

        loop {
            // TODO: do one iteration of the work. Perhaps respawning the agent in a controlled
            // crash.
            if true {
                break;
            }
        }
        Ok(())
    }
}

/// Run the workload until either it either finishes, errors or gets cancelled.
///
/// Returns `None` if the investigation is not required (i.e. cancelled or succeeded), otherwise,
/// returns the investigation report.
async fn run_workload(cancel_token: CancellationToken) -> Result<Option<InvestigationFlag>> {
    // This creates a temp dir for the working dir of the workload.
    let workdir = TempDir::new()?;
    let mut workload = Workload::new(workdir);
    let result = workload.run(cancel_token).await;

    // match result {
    //     None => {
    //         // The task was cancelled. Send SIGKILL to the process.
    //         child.send_sigkill();
    //         drop(controller.workdir);
    //         return Ok(None);
    //     }
    //     Some(result) => {
    //         // Either there was an error or the task finished successfully.
    //         // If there was an error, we should flag for investigation.
    //     }
    // }
    Ok(None)
}

/// Run the control loop creating and tearing down the agents.
///
/// `cancel_token` is used to gracefully shutdown the supervisor.
async fn control_loop(cancel_token: CancellationToken) -> Result<()> {
    const FLAG_NUMBER_LIMIT: usize = 10;
    let mut flags = Vec::new();
    loop {
        // TODO:
        let maybe_flag = run_workload(cancel_token.clone()).await?;
        if let Some(flag) = maybe_flag {
            println!(
                "Flagged for investigation\nreason={reason:?}\nworkdir={workdir}",
                reason = flag.reason,
                workdir = flag.workdir.display()
            );
            flags.push(flag);
        }
        if cancel_token.is_cancelled() {
            break;
        }
        if flags.len() >= FLAG_NUMBER_LIMIT {
            break;
        }
    }
    Ok(())
}
