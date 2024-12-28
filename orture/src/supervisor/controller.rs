use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use super::comms;
use crate::{
    spawn::{self, Child},
    util::alarm::Alarm,
};
use anyhow::{bail, Result};
use tokio::{net::UnixStream, sync::oneshot, task};
use tracing::trace;

/// A controller is responsible for overseeing a single agent process and handle its lifecycle.
pub struct SpawnedAgentController {
    child: Child,
    rr: comms::RequestResponse,
    unhealthy: Arc<Alarm>,
    comms_ah: task::AbortHandle,
    waitpid_ah: task::AbortHandle,
    torn_down: AtomicBool,
}

// This is a safe-guard to ensure that the controller is torn down properly.
impl Drop for SpawnedAgentController {
    fn drop(&mut self) {
        if !self.torn_down.load(Ordering::Relaxed) {
            panic!("controller was not torn down properly");
        }
    }
}

impl SpawnedAgentController {
    /// Kills the process, shuts down the comms, and cleans up the resources.
    pub fn teardown(self) {
        self.torn_down.store(true, Ordering::Relaxed);
        self.child.send_sigkill();
        self.comms_ah.abort();
        self.waitpid_ah.abort();
    }

    /// Reads and returns the current virtual memory size of the agent process.
    pub async fn current_vm_rss(&self) -> Result<usize> {
        let path = format!("/proc/{}/status", self.child.pid);
        let status = tokio::fs::read(path).await?;
        drop(status);
        todo!()
    }

    pub fn rr(&self) -> &comms::RequestResponse {
        &self.rr
    }

    /// Resolves when the agent died, the stream is closed, or otherwise the agent is unhealthy.
    pub async fn resolve_when_unhealthy(&self) {
        self.unhealthy.triggered().await;
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
    let (tx, rx) = oneshot::channel();
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

/// Spawns an agent process creating a controller.
///
/// The controller is placed in the `place` argument. `place` must be `None` when calling this
/// function.
pub async fn spawn_agent_into(
    place: &mut Option<SpawnedAgentController>,
    workdir: String,
    workload_id: u64,
) -> Result<()> {
    assert!(place.is_none(), "the controller must be empty");

    let (child, sock) = spawn::spawn_child()?;
    trace!("spawned agent, pid={}", child.pid);

    let stream = UnixStream::from_std(sock)?;
    let unhealthy = Arc::new(Alarm::new());

    let (rr, task) = comms::run(stream);
    let comms_ah = tokio::spawn({
        // Spawn a task that drives the comms logic.
        //
        // This triggers the unhealthy alarm when finishes (irregardless whether Ok or Err).
        let unhealthy = unhealthy.clone();
        async move {
            let result = task.await;
            trace!("comms finished: {:?}", result);
            unhealthy.trigger();
        }
    })
    .abort_handle();

    let waitpid_ah = tokio::spawn({
        // Spawn a task that monitors the exit code of the process.
        let unhealthy = unhealthy.clone();
        let child = child.clone();
        async move {
            let pid = child.pid;
            let e = wait_for_exit(child).await;
            trace!(pid, "child exit code: {:?}", e);
            unhealthy.trigger();
        }
    })
    .abort_handle();

    // Assign a unique ID to the agent.
    static AGENT_COUNT: AtomicUsize = AtomicUsize::new(0);
    let agent_number = AGENT_COUNT.fetch_add(1, Ordering::Relaxed);
    let id = format!("agent-{}-{}", workload_id, agent_number);

    // TODO: a proper init.
    //
    // id, bitbox_seed, etc.
    rr.send_request(crate::message::ToAgent::Init(crate::message::InitPayload {
        id,
        workdir,
        bitbox_seed: [0; 16],
    }))
    .await?;

    *place = Some(SpawnedAgentController {
        child,
        rr,
        unhealthy,
        comms_ah,
        waitpid_ah,
        torn_down: AtomicBool::new(false),
    });
    Ok(())
}
