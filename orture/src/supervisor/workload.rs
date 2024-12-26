use std::{future::Future, time::Duration};

use anyhow::Result;
use imbl::OrdMap;
use rand::prelude::*;
use tempfile::TempDir;
use tokio::time::{error::Elapsed, timeout};
use tokio_util::sync::CancellationToken;

use crate::message::KeyValueChange;

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
}

impl Workload {
    pub fn new(workdir: TempDir) -> Self {
        Self {
            workdir,
            agent: None,
            workload_size: 1000,
            state: WorkloadState::new(0xFAD0),
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
                    agent.send_sigkill();
                }
            }
        }
        Ok(())
    }

    async fn run_inner(&mut self) -> Result<()> {
        self.agent = Some(
            controller::spawn_agent({
                let this = &self;
                this.workdir.path().display().to_string()
            })
            .await?,
        );
        for _ in 0..self.workload_size {
            self.run_iteration().await?;
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
        match exercise_ix {
            0 => {
                assert_healthy_agent(agent, exercise_commit(&rr, s)).await?;
            }
            1 => {
                let workdir = self.workdir.path().display().to_string();
                exercise_crashing_commit(&rr, s, &mut self.agent, workdir).await?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    pub fn into_workdir(self) -> TempDir {
        self.workdir
    }
}

async fn assert_healthy_agent<F>(agent: &SpawnedAgentController, future: F) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<()>>,
{
    tokio::select! {
        result = future => result,
        _ = agent.resolve_when_unhealthy() => {
            panic!("agent is unhealthy");
        }
    }
}

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

async fn exercise_crashing_commit(
    rr: &comms::RequestResponse,
    s: &mut WorkloadState,
    workload_agent: &mut Option<SpawnedAgentController>,
    workdir: String,
) -> anyhow::Result<()> {
    let (snapshot, changeset) = s.gen_commit();
    rr.send_request(crate::message::ToAgent::Commit(
        crate::message::CommitPayload {
            changeset,
            should_crash: true,
        },
    ))
    .await?;
    // Wait until the agent dies, kill it to make sure.
    const TOLERANCE: Duration = Duration::from_secs(5);
    let agent = workload_agent.take().unwrap();
    match timeout(TOLERANCE, agent.resolve_when_unhealthy()).await {
        Ok(()) => (),
        Err(Elapsed { .. }) => {
            agent.send_sigkill();
            // TODO: flag for investigation.
            return Err(anyhow::anyhow!("agent did not die"));
        }
    }
    agent.send_sigkill();
    drop(agent);
    let agent = controller::spawn_agent(workdir).await?;
    *workload_agent = Some(agent);

    let seqno = match rr
        .send_request(crate::message::ToAgent::Query(SEQNO_KEY))
        .await?
    {
        crate::message::ToSupervisor::QueryResponse(vec) => {
            // TODO: clean this up.
            let mut buf = [0; 8];
            buf.copy_from_slice(&vec.unwrap());
            u64::from_le_bytes(buf)
        }
        _ => unreachable!(),
    };
    if seqno == snapshot.seqno {
        s.commit(snapshot);
    }
    Ok(())
}
