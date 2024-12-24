use super::comms;
use crate::spawn::{self, Child};
use anyhow::{bail, Result};
use tokio::{net::UnixStream, sync::oneshot, task};

/// A controller is responsible for overseeing a single agent process and handle its lifecycle.
pub struct SpawnedAgentController {
    child: Child,
    rr: comms::RequestResponse,
}

impl SpawnedAgentController {
    pub fn send_sigkill(&self) {
        self.child.send_sigkill();
    }

    /// Reads and returns the current virtual memory size of the agent process.
    pub async fn current_vm_rss(&self) -> Result<usize> {
        let path = format!("/proc/{}/status", self.child.pid);
        let status = tokio::fs::read(path).await?;
        todo!()
    }

    pub fn rr(&self) -> &comms::RequestResponse {
        &self.rr
    }

    /// Resolves when the agent died, the stream is closed, or otherwise the agent is unhealthy.
    pub async fn resolve_when_unhealthy(&self) {
        todo!()
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

/// Spawns an agent process and returns a controller for it.
pub async fn spawn_agent(workdir: String) -> Result<SpawnedAgentController> {
    let (child, sock) = spawn::spawn_child()?;
    let stream = UnixStream::from_std(sock)?;

    let (rr, task) = comms::run(stream);
    // TODO: decide how to drive the `task`.

    rr.send_request(crate::message::ToAgent::Init(crate::message::InitPayload {
        id: "1".to_string(),
        workdir,
        bitbox_seed: [0; 16],
    }))
    .await?;

    Ok(SpawnedAgentController { child, rr })
}
