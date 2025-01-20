use super::comms;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::{net::UnixStream, process::Child};

/// A controller is responsible for overseeing a single agent process and handle its lifecycle.
pub struct SpawnedAgentController {
    child: Child,
    rr: comms::RequestResponse,
    torn_down: AtomicBool,
}

// This is a safe-guard to ensure that the [`SpawnedAgentController::teardown`] is called
// properly.
impl Drop for SpawnedAgentController {
    fn drop(&mut self) {
        if self.torn_down.load(Ordering::Relaxed) {
            // The controller was torn down properly, disarm.
            return;
        }
        if std::thread::panicking() {
            // The controller was not torn down properly, but we are panicking.
            eprintln!("controller was not torn down properly");
            return;
        }
        panic!("controller was not torn down properly");
    }
}

impl SpawnedAgentController {
    pub async fn init(
        &mut self,
        workdir: String,
        workload_id: u64,
        bitbox_seed: [u8; 16],
        rollback: bool,
    ) -> Result<()> {
        // Assign a unique ID to the agent.
        static AGENT_COUNT: AtomicUsize = AtomicUsize::new(0);
        let agent_number = AGENT_COUNT.fetch_add(1, Ordering::Relaxed);
        let id = format!("agent-{}-{}", workload_id, agent_number);
        self.rr
            .send_request(crate::message::ToAgent::Init(crate::message::InitPayload {
                id,
                workdir,
                bitbox_seed,
                rollback,
            }))
            .await?;
        Ok(())
    }

    /// Kills the process, shuts down the comms, and cleans up the resources.
    ///
    /// This returns only when the process is dead and the resources are cleaned up.
    ///
    /// The controller must be torn down manually. Dropping the controller is disallowed. This is
    /// done to control precisely when the agent process is killed.
    pub async fn teardown(mut self) {
        self.torn_down.store(true, Ordering::Relaxed);
        let _ = self.child.kill().await;
    }

    /// Resolves when the agent process exits.
    pub async fn died(&mut self) {
        let _ = self.child.wait().await;
    }

    pub fn rr(&self) -> &comms::RequestResponse {
        &self.rr
    }
}

/// Spawns an agent process creating a controller.
///
/// The controller is placed in the `place` argument. `place` must be `None` when calling this
/// function.
pub async fn spawn_agent_into(place: &mut Option<SpawnedAgentController>) -> Result<()> {
    assert!(place.is_none(), "the controller must be empty");

    let (child, sock) = crate::spawn::spawn_child()?;

    let stream = UnixStream::from_std(sock)?;

    let (rr, task) = comms::run(stream);
    let _ = tokio::spawn(task);

    *place = Some(SpawnedAgentController {
        child,
        rr,
        torn_down: AtomicBool::new(false),
    });
    Ok(())
}
