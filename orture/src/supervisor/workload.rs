use std::{future::Future, time::Duration};

use anyhow::{bail, Result};
use imbl::OrdMap;
use rand::prelude::*;
use tempfile::TempDir;
use tokio::time::{error::Elapsed, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{trace, trace_span, Instrument as _};

use crate::message::{self, KeyValueChange};

use super::{
    comms,
    controller::{self, SpawnedAgentController},
};

const SEQNO_KEY: [u8; 32] = [0xAA; 32];

#[derive(Clone)]
struct Biases {
    delete: f64,
    overflow: f64,
}

impl Biases {
    fn new() -> Self {
        Self {
            delete: 0.1,
            overflow: 0.1,
        }
    }
}

/// Represents a snapshot of the state of the database.
#[derive(Clone)]
struct Snapshot {
    seqno: u64,
    state: OrdMap<[u8; 32], Option<Vec<u8>>>,
}

impl Snapshot {
    fn empty() -> Self {
        Self {
            seqno: 0,
            state: OrdMap::new(),
        }
    }
}

struct WorkloadState {
    rng: rand_pcg::Pcg64,
    biases: Biases,
    /// The values that were committed.
    committed: Snapshot,
}

impl WorkloadState {
    fn new(seed: u64) -> Self {
        Self {
            rng: rand_pcg::Pcg64::seed_from_u64(seed),
            biases: Biases::new(),
            committed: Snapshot::empty(),
        }
    }

    fn gen_commit(&mut self) -> (Snapshot, Vec<KeyValueChange>) {
        let new_seqno = self.committed.seqno + 1;
        let size = self.rng.gen_range(0..10000);
        let mut changes = Vec::with_capacity(size);
        for _ in 0..size {
            let key = self.gen_key();
            let kvc = if self.rng.gen_bool(self.biases.delete) {
                KeyValueChange::Delete(key)
            } else {
                let value = self.rng.gen::<[u8; 32]>().to_vec();
                KeyValueChange::Insert(key, value.clone())
            };
            changes.push(kvc);
        }
        // TODO: Consider other ways of detecting whether the commit was applied or not.
        changes.push(KeyValueChange::Insert(
            SEQNO_KEY,
            new_seqno.to_le_bytes().to_vec(),
        ));
        let mut snapshot = self.committed.clone();
        snapshot.seqno = new_seqno;
        for change in &changes {
            match change {
                KeyValueChange::Insert(key, value) => {
                    snapshot.state.insert(*key, Some(value.clone()));
                }
                KeyValueChange::Delete(key) => {
                    snapshot.state.remove(key);
                }
            }
        }
        (snapshot, changes)
    }

    fn gen_key(&mut self) -> [u8; 32] {
        // TODO: sophisticated key generation.
        //
        // - Return a key that was already generated before.
        // - Pick a key that was already generated before, but generate a key that shares some bits.
        let mut key = [0; 32];
        self.rng.fill_bytes(&mut key);
        key
    }

    fn gen_value(&mut self) -> Vec<u8> {
        // TODO: sophisticated value generation.
        //
        // - Different power of two sizes.
        // - Change it to be a non-even.
        let len = if self.rng.gen_bool(self.biases.overflow) {
            32 * 1024
        } else {
            32
        };
        let mut value = vec![0; len];
        self.rng.fill_bytes(&mut value);
        value
    }

    fn commit(&mut self, snapshot: Snapshot) {
        self.committed = snapshot;
    }

    fn snapshot(&self) -> &Snapshot {
        &self.committed
    }
}

/// This is a struct that controls workload execution.
///
/// A workload is a set of tasks that the agents should perform. We say agents, plural, because
/// the same workload can be executed by multiple agents. Howeever, it's always sequential. This
/// arises from the fact that as part of the workload we need to crash the agent to check how
/// it behaves.
pub struct Workload {
    /// Working directory for this particular workload.
    workdir: TempDir,
    /// The currently spawned agent.
    ///
    /// Initially `None`.
    agent: Option<SpawnedAgentController>,
    /// How many iterations the workload should perform.
    workload_size: usize,
    /// The current state of the workload.
    state: WorkloadState,
    /// The identifier of the workload. Useful for debugging.
    workload_id: u64,
}

impl Workload {
    pub fn new(seed: u64, workdir: TempDir, workload_id: u64) -> Self {
        // TODO: Make the workload size configurable and more sophisticated.
        //
        // Right now the workload size is a synonym for the number of iterations. We probably
        // want to make it more about the number of keys inserted. Potentially, we might consider
        // testing the "edging" scenario where the supervisor tries to stay below the maximum
        // number of items in the database occasionally going over the limit expecting proper
        // handling of the situation.
        let workload_size = 50;
        Self {
            workdir,
            agent: None,
            workload_size,
            state: WorkloadState::new(seed),
            workload_id,
        }
    }

    /// Run the workload.
    ///
    /// Pass the cancellation token to the workload. The workload will run until the token is
    /// cancelled or the workload finishes.
    pub async fn run(&mut self, cancel_token: CancellationToken) -> Result<()> {
        match cancel_token.run_until_cancelled(self.run_inner()).await {
            Some(r) => return r,
            None => {
                // Cancelled. Send SIGKILL to the agent.
                if let Some(agent) = self.agent.take() {
                    agent.teardown();
                }
            }
        }
        Ok(())
    }

    async fn run_inner(&mut self) -> Result<()> {
        controller::spawn_agent_into(
            &mut self.agent,
            self.workdir.path().display().to_string(),
            self.workload_id,
        )
        .await?;
        for iterno in 0..self.workload_size {
            self.run_iteration()
                .instrument(trace_span!("iteration", iterno))
                .await?;
        }
        Ok(())
    }

    async fn run_iteration(&mut self) -> Result<()> {
        let agent = self.agent.as_ref().unwrap();
        let rr = agent.rr().clone();
        // TODO: make the choice of the exercise more sophisticated.
        //
        // - commits should be much more frequent.
        // - crashes should be less frequent.
        // - rollbacks should be less frequent.
        let exercise_ix = self.state.rng.gen_range(0..2);
        let s = &mut self.state;
        trace!("run_iteration: choice {}", exercise_ix);
        match exercise_ix {
            0 => {
                assert_healthy_agent(agent, exercise_commit(&rr, s)).await?;
            }
            1 => {
                let workdir = self.workdir.path().display().to_string();
                exercise_commit_crashing(&rr, s, &mut self.agent, workdir, self.workload_id)
                    .await?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Release potentially held resources.
    pub fn teardown(&mut self) {
        if let Some(agent) = self.agent.take() {
            agent.teardown();
        }
    }

    /// Return the working directory.
    pub fn into_workdir(self) -> TempDir {
        self.workdir
    }
}

async fn assert_healthy_agent<F>(agent: &SpawnedAgentController, future: F) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<()>>,
{
    let future = future.instrument(trace_span!("assert_healthy_agent"));
    tokio::select! {
        result = future => {
            result
        },
        _ = agent.resolve_when_unhealthy() => {
            panic!("agent is unhealthy");
        }
    }
}

// TODO: Exercise should sample the values from the state.

/// Commit a changeset.
async fn exercise_commit(rr: &comms::RequestResponse, s: &mut WorkloadState) -> anyhow::Result<()> {
    let (snapshot, changeset) = s.gen_commit();
    rr.send_request(crate::message::ToAgent::Commit(
        crate::message::CommitPayload {
            changeset,
            should_crash: false,
        },
    ))
    .await?;
    s.commit(snapshot);
    Ok(())
}

/// Commit a changeset and induce a crash.
///
/// `workload_agent` is the agent that should be crashed. It is assumed that the agent is healthy.
/// After the crash, the agent is respawned and saved into the `workload_agent`.
async fn exercise_commit_crashing(
    rr: &comms::RequestResponse,
    s: &mut WorkloadState,
    workload_agent: &mut Option<SpawnedAgentController>,
    workdir: String,
    workload_id: u64,
) -> anyhow::Result<()> {
    // Generate a changeset and the associated snapshot. Ask the agent to commit the changeset.
    //
    // The agent should crash before or after the commit.
    let (snapshot, changeset) = s.gen_commit();
    rr.send_request(crate::message::ToAgent::Commit(
        crate::message::CommitPayload {
            changeset,
            should_crash: true,
        },
    ))
    .await?;

    // Wait until the agent dies. We give it a grace period of 5 seconds.
    //
    // Note that we don't "take" the agent from the `workload_agent` place. This is because there is
    // always a looming possibility of SIGINT arriving.
    const TOLERANCE: Duration = Duration::from_secs(5);
    let agent = workload_agent.as_ref().unwrap();
    let agent_died_or_timeout = timeout(TOLERANCE, agent.resolve_when_unhealthy()).await;
    workload_agent.take().unwrap().teardown();
    if let Err(Elapsed { .. }) = agent_died_or_timeout {
        // TODO: flag for investigation.
        return Err(anyhow::anyhow!("agent did not die"));
    }

    // Spawns a new agent and checks whether the commit was applied to the database and if so
    // we commit the snapshot to the state.
    controller::spawn_agent_into(workload_agent, workdir, workload_id).await?;
    let rr = workload_agent.as_ref().unwrap().rr();
    let seqno = request_seqno(rr).await?;
    trace!("commit. seqno ours: {}, theirs: {}", snapshot.seqno, seqno);
    if seqno == snapshot.seqno {
        s.commit(snapshot);
    }
    Ok(())
}

/// Request the current sequence number from the agent.
async fn request_seqno(rr: &comms::RequestResponse) -> anyhow::Result<u64> {
    trace!("sending seqno query");
    let value = match rr.send_request_query(SEQNO_KEY).await? {
        Some(vec) => vec,
        None => bail!("seqno key not found"),
    };
    let seqno = u64::from_le_bytes(value.try_into().unwrap());
    Ok(seqno)
}
