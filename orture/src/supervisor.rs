//! The supervisor part. Spawns and manages agents. Assigns work to agents.

use std::process::exit;

use anyhow::{bail, Result};
use tempfile::TempDir;
use tokio::{
    net::UnixStream,
    signal::unix::{signal, SignalKind},
    task::{self, JoinHandle},
};
use tokio_util::sync::CancellationToken;

use crate::spawn::{self, Child};

mod comms;

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

/// A controller is responsible for overseeing a single agent process and handle its lifecycle.
struct Controller {
    /// Working directory for the agent.
    workdir: TempDir,
}

impl Controller {
    async fn run(&self, stream: UnixStream) -> Result<()> {
        let rr = comms::start_comms(stream).await;
        let mut workload = Workload { rr };
        workload.run().await?;
        // TODO: watch notification stream.
        Ok(())
    }
}

/// Wait for the child process to exit.
///
/// Reaps the child process and returns its exit status. Under the hood this uses `waitpid(2)`.
///
/// # Cancel safety
///
/// This is cancel safe.
///
/// Dropping the future returned by this function will not cause `waitpid` to
/// be interrupted. Instead, it will be allowed to finish. We cannot cancel `waitpid` because it
/// blocks the thread neither we want to do that because we want to reap the child process. In case
/// we fail to do so, the child process will become a zombie and too many zombies will exhaust the
/// system resources.
async fn wait_for_exit(child: Child) -> Result<i32> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    task::spawn_blocking(move || {
        let result = child.wait();
        let _ = tx.send(result);
    });
    // UNWRAP: the channel should never panic.
    let Ok(exit_status) = rx.await else {
        bail!("unexpected hungup of waitpid");
    };
    Ok(exit_status)
}

/// `cancel_token` is used to gracefully shutdown the supervisor.
async fn control_loop(cancel_token: CancellationToken) -> Result<()> {
    if true {
        Err::<(), anyhow::Error>(anyhow::anyhow!("this is a test error")).unwrap();
    }

    let (child, sock) = spawn::spawn_child()?;
    let stream = UnixStream::from_std(sock)?;
    let workdir = TempDir::new()?;
    let controller = Controller { workdir };
    let ctrl_c = tokio::signal::ctrl_c();

    // cancel_token
    //     .run_until_cancelled(controller.run())
    //     .await
    //     .unwrap();
    //
    // TODO: initiate graceful shutdown.
    //
    // Graceful shutdown should involve:
    //
    // 1. waiting until the agent was killed.
    // 2. it's working directory was removed.
    //
    // In the future, we should be ready to handle multiple ^C signals. The second one
    // means that the user is really impatient and wants to exit immediately and we
    // should comply.
    //
    // Well, actually, I think the first signal should be handled via the cancellation token and
    // the second ctrl-c should be handled by the higher level logic.

    // loop {
    //     tokio::select! {
    //         ctrl_c = ctrl_c => {
    //             let () = ctrl_c?;
    //             break;
    //         }
    //         exit_status = wait_for_exit(child) => {
    //             // TODO: this is not cancel safe.
    //             println!("Child exited with status: {}", exit_status.unwrap());
    //             break;
    //         }
    //         // TODO: assign workload.
    //     }
    // }
    Ok(())
}

struct SpawnedAgentController {}

/// This is a struct that controls the workload execution for a particular agent.
struct Workload {
    // TODO: state of the workload.
    //
    // The stuff that helps to generate the workload. For example, the keys that were inserted.
    rr: comms::RequestResponse,
}

impl Workload {
    /// Run the workload.
    async fn run(&mut self) -> Result<()> {
        // TODO: send the init message.
        // self.rr.send_request(ToAgent::Init()).await?;
        loop {
            // TODO: do one iteration of the work.
            if true {
                break;
            }
        }
        Ok(())
    }
}
