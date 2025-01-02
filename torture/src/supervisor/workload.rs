use std::time::Duration;

use anyhow::Result;
use imbl::OrdMap;
use rand::prelude::*;
use tempfile::TempDir;
use tokio::time::{error::Elapsed, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{info, trace, trace_span, Instrument as _};

use crate::message::KeyValueChange;

use super::{
    comms,
    controller::{self, SpawnedAgentController},
};

/// Biases is configuration for the workload generator. Its values are used to bias the probability
/// of certain events happening.
#[derive(Clone)]
struct Biases {
    /// The probability of a delete operation as opposed to an insert operation.
    delete: f64,
    /// When generating a value, the probability of generating a value that will spill into the
    /// overflow pages.
    overflow: f64,
    /// When generating a key, whether it should be one that was appeared somewhere or a brand new
    /// key.
    new_key: f64,
}

impl Biases {
    fn new() -> Self {
        Self {
            delete: 0.1,
            overflow: 0.1,
            new_key: 0.5,
        }
    }
}

/// Represents a snapshot of the state of the database.
#[derive(Clone)]
struct Snapshot {
    sync_seqn: u32,
    state: OrdMap<[u8; 32], Option<Vec<u8>>>,
}

impl Snapshot {
    fn empty() -> Self {
        Self {
            sync_seqn: 0,
            state: OrdMap::new(),
        }
    }
}

struct WorkloadState {
    rng: rand_pcg::Pcg64,
    biases: Biases,
    /// The values that were committed.
    committed: Snapshot,
    /// All keys that were generated during this run.
    ///
    /// It's extremely unlikely to find any duplicates in there. Each key is very likely to store
    /// a value.
    mentions: Vec<[u8; 32]>,
}

impl WorkloadState {
    fn new(seed: u64) -> Self {
        Self {
            rng: rand_pcg::Pcg64::seed_from_u64(seed),
            biases: Biases::new(),
            committed: Snapshot::empty(),
            mentions: Vec::with_capacity(32 * 1024),
        }
    }

    fn gen_bitbox_seed(&mut self) -> [u8; 16] {
        let mut bitbox_seed = [0u8; 16];
        self.rng.fill_bytes(&mut bitbox_seed);
        bitbox_seed
    }

    fn gen_commit(&mut self) -> (Snapshot, Vec<KeyValueChange>) {
        let new_sync_seqn = self.committed.sync_seqn + 1;
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
        changes.sort_by(|a, b| a.key().cmp(b.key()));
        changes.dedup_by(|a, b| a.key() == b.key());
        let mut snapshot = self.committed.clone();
        snapshot.sync_seqn = new_sync_seqn;
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
        // - Pick a key that was already generated before, but generate a key that shares some bits.
        if self.mentions.is_empty() || self.rng.gen_bool(self.biases.new_key) {
            let mut key = [0; 32];
            self.rng.fill_bytes(&mut key);
            self.mentions.push(key.clone());
            key
        } else {
            let ix = self.rng.gen_range(0..self.mentions.len());
            self.mentions[ix].clone()
        }
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
    /// The seed for bitbox generated for this workload.
    bitbox_seed: [u8; 16],
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
        let mut state = WorkloadState::new(seed);
        let bitbox_seed = state.gen_bitbox_seed();
        Self {
            workdir,
            agent: None,
            workload_size,
            state,
            workload_id,
            bitbox_seed,
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
        self.spawn_new_agent().await?;
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
        trace!("run_iteration: choice {}", exercise_ix);
        match exercise_ix {
            0 => {
                // TODO: assert that it doesn't crash.
                self.exercise_commit(&rr).await?;
            }
            1 => {
                self.exercise_commit_crashing(&rr).await?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Commit a changeset.
    async fn exercise_commit(&mut self, rr: &comms::RequestResponse) -> anyhow::Result<()> {
        let (snapshot, changeset) = self.state.gen_commit();
        rr.send_request(crate::message::ToAgent::Commit(
            crate::message::CommitPayload {
                changeset,
                should_crash: false,
            },
        ))
        .await?;
        self.state.commit(snapshot);
        // TODO: Exercise should sample the values from the state.
        Ok(())
    }

    /// Commit a changeset and induce a crash.
    async fn exercise_commit_crashing(
        &mut self,
        rr: &comms::RequestResponse,
    ) -> anyhow::Result<()> {
        // Generate a changeset and the associated snapshot. Ask the agent to commit the changeset.
        //
        // The agent should crash before or after the commit.
        let (snapshot, changeset) = self.state.gen_commit();
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
        let agent = self.agent.as_ref().unwrap();
        let agent_died_or_timeout = timeout(TOLERANCE, agent.resolve_when_unhealthy()).await;
        self.agent.take().unwrap().teardown();
        if let Err(Elapsed { .. }) = agent_died_or_timeout {
            // TODO: flag for investigation.
            return Err(anyhow::anyhow!("agent did not die"));
        }

        // Spawns a new agent and checks whether the commit was applied to the database and if so
        // we commit the snapshot to the state.
        self.spawn_new_agent().await?;
        let rr = self.agent.as_ref().unwrap().rr();
        let seqno = rr.send_query_sync_seqn().await?;
        if seqno == snapshot.sync_seqn {
            self.state.commit(snapshot);
        } else {
            info!(
                "commit. seqno ours: {}, theirs: {}",
                snapshot.sync_seqn, seqno
            );
        }

        // TODO: Exercise should sample the values from the state.

        Ok(())
    }

    async fn spawn_new_agent(&mut self) -> anyhow::Result<()> {
        assert!(self.agent.is_none());
        controller::spawn_agent_into(
            &mut self.agent,
            self.workdir.path().display().to_string(),
            self.workload_id,
            self.bitbox_seed,
        )
        .await?;
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
