use anyhow::Result;
use imbl::OrdMap;
use rand::prelude::*;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::{error::Elapsed, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{info, trace, trace_span, Instrument as _};

use crate::{
    message::{KeyValueChange, ToSupervisor},
    supervisor::{
        cli::WorkloadParams,
        comms,
        controller::{self, SpawnedAgentController},
    },
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
    /// When exercising a new commit, the probability of causing it to crash.
    crash: f64,
}

impl Biases {
    fn new(delete: u8, overflow: u8, new_key: u8, crash: u8) -> Self {
        Self {
            delete: (delete as f64) / 100.0,
            overflow: (overflow as f64) / 100.0,
            new_key: (new_key as f64) / 100.0,
            crash: (crash as f64) / 100.0,
        }
    }
}

/// Represents a snapshot of the state of the database.
#[derive(Clone)]
struct Snapshot {
    sync_seqn: u32,
    /// The state of the database at this snapshot.
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
    /// The number of items that each commit will contain in its changeset.
    size: usize,
    /// If true, the size of each commit will be within 0 and self.size,
    /// otherwise it will always be workload-size.
    random_size: bool,
    /// The values that were committed.
    committed: Snapshot,
}

impl WorkloadState {
    fn new(seed: u64, biases: Biases, size: usize, random_size: bool) -> Self {
        Self {
            rng: rand_pcg::Pcg64::seed_from_u64(seed),
            biases,
            size,
            random_size,
            committed: Snapshot::empty(),
        }
    }

    fn gen_bitbox_seed(&mut self) -> [u8; 16] {
        let mut bitbox_seed = [0u8; 16];
        self.rng.fill_bytes(&mut bitbox_seed);
        bitbox_seed
    }

    fn gen_commit(&mut self) -> (Snapshot, Vec<KeyValueChange>) {
        let mut snapshot = self.committed.clone();
        snapshot.sync_seqn += 1;

        let num_changes = if self.random_size {
            self.rng.gen_range(0..self.size)
        } else {
            self.size
        };
        let mut changes = Vec::with_capacity(num_changes);

        // Commiting requires using only the unique keys. To ensure that we deduplicate the keys
        // using a hash set.
        let mut used_keys = std::collections::HashSet::with_capacity(num_changes);
        loop {
            let change = self.gen_key_value_change();
            if used_keys.contains(change.key()) {
                continue;
            }

            snapshot.state.insert(*change.key(), change.value());
            used_keys.insert(*change.key());
            changes.push(change);

            if used_keys.len() >= num_changes {
                break;
            }
        }

        changes.sort_by(|a, b| a.key().cmp(&b.key()));

        (snapshot, changes)
    }

    /// Returns a KeyValueChange with a new key, a deleted or a modified one.
    fn gen_key_value_change(&mut self) -> KeyValueChange {
        // TODO: sophisticated key generation.
        //
        // - Pick a key that was already generated before, but generate a key that shares some bits.
        let mut key = [0; 32];
        if !self.committed.state.is_empty() && self.rng.gen_bool(self.biases.delete) {
            loop {
                self.rng.fill_bytes(&mut key);
                if let Some((next_key, Some(_))) = self.committed.state.get_next(&key) {
                    return KeyValueChange::Delete(*next_key);
                }
            }
        }

        if self.committed.state.is_empty() || self.rng.gen_bool(self.biases.new_key) {
            loop {
                self.rng.fill_bytes(&mut key);
                if !self.committed.state.contains_key(&key) {
                    return KeyValueChange::Insert(key, self.gen_value());
                }
            }
        }

        loop {
            self.rng.fill_bytes(&mut key);
            if let Some((next_key, _)) = self.committed.state.get_next(&key) {
                return KeyValueChange::Insert(*next_key, self.gen_value());
            }
        }
    }

    fn gen_value(&mut self) -> Vec<u8> {
        // TODO: sophisticated value generation.
        //
        // - Different power of two sizes.
        // - Change it to be a non-even.
        let len = if self.rng.gen_bool(self.biases.overflow) {
            // TODO: handle multiple overflow page scenario.
            // Currently, using a value that is too large causes the channel to hang
            // when the agent is responding with the queried value.
            // However MAX_LEAF_VALUE_SIZE is 1332 thus 2000 is enough
            // to test simple overflow values where only one additional page is used.
            2000
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
    iterations: usize,
    /// The current state of the workload.
    state: WorkloadState,
    /// The identifier of the workload. Useful for debugging.
    workload_id: u64,
    /// The seed for bitbox generated for this workload.
    bitbox_seed: [u8; 16],
    /// Data collected to evaluate the average commit time [ns].
    tot_commit_time: u64,
    n_successfull_commit: u64,
}

impl Workload {
    pub fn new(
        seed: u64,
        workdir: TempDir,
        workload_params: &WorkloadParams,
        workload_id: u64,
    ) -> Self {
        // TODO: Make the workload size configurable and more sophisticated.
        //
        // Right now the workload size is a synonym for the number of iterations. We probably
        // want to make it more about the number of keys inserted. Potentially, we might consider
        // testing the "edging" scenario where the supervisor tries to stay below the maximum
        // number of items in the database occasionally going over the limit expecting proper
        // handling of the situation.
        let workload_biases = Biases::new(
            workload_params.delete,
            workload_params.overflow,
            workload_params.new_key,
            workload_params.crash,
        );
        let mut state = WorkloadState::new(
            seed,
            workload_biases,
            workload_params.size,
            workload_params.random_size,
        );
        let bitbox_seed = state.gen_bitbox_seed();
        Self {
            workdir,
            agent: None,
            iterations: workload_params.iterations,
            state,
            workload_id,
            bitbox_seed,
            tot_commit_time: 0,
            n_successfull_commit: 0,
        }
    }

    /// Run the workload.
    ///
    /// Pass the cancellation token to the workload. The workload will run until the token is
    /// cancelled or the workload finishes.
    pub async fn run(&mut self, cancel_token: CancellationToken) -> Result<()> {
        let result = match cancel_token.run_until_cancelled(self.run_inner()).await {
            Some(r) => r,
            None => Ok(()),
        };
        // Irregardless of the result or if the workload was cancelled, we need to release the
        // resources.
        self.teardown().await;
        result
    }

    async fn run_inner(&mut self) -> Result<()> {
        self.spawn_new_agent().await?;
        for iterno in 0..self.iterations {
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
        let exercise_crash = self.state.rng.gen_bool(self.state.biases.crash);
        if exercise_crash {
            trace!("run_iteration, should crash");
            self.exercise_commit_crashing(&rr).await?;
        } else {
            trace!("run_iteration");
            self.exercise_commit(&rr).await?;
        }
        Ok(())
    }

    /// Commit a changeset.
    async fn exercise_commit(&mut self, rr: &comms::RequestResponse) -> anyhow::Result<()> {
        let (snapshot, changeset) = self.state.gen_commit();
        let ToSupervisor::CommitSuccessful(elapsed) = rr
            .send_request(crate::message::ToAgent::Commit(
                crate::message::CommitPayload {
                    changeset: changeset.clone(),
                    should_crash: None,
                },
            ))
            .await?
        else {
            return Err(anyhow::anyhow!("Commit did not execute successfully"));
        };

        self.n_successfull_commit += 1;
        self.tot_commit_time += elapsed;

        // Sample the agent to make sure the changeset was correctly applied.
        let agent_sync_seqn = rr.send_query_sync_seqn().await?;
        if snapshot.sync_seqn != agent_sync_seqn {
            return Err(anyhow::anyhow!("Unexpected sync_seqn after commit"));
        }
        self.ensure_changeset_applied(rr, &changeset).await?;

        self.state.commit(snapshot);

        Ok(())
    }

    async fn ensure_changeset_applied(
        &self,
        rr: &comms::RequestResponse,
        changeset: &Vec<KeyValueChange>,
    ) -> anyhow::Result<()> {
        for change in changeset {
            match change {
                KeyValueChange::Insert(key, value)
                    if rr.send_request_query(*key).await?.as_ref() != Some(&value) =>
                {
                    return Err(anyhow::anyhow!("Inserted item not present after commit"));
                }
                KeyValueChange::Delete(key) if rr.send_request_query(*key).await?.is_some() => {
                    return Err(anyhow::anyhow!("Deleted item still present after commit"));
                }
                _ => (),
            }
        }
        Ok(())
    }

    /// Commit a changeset and induce a crash.
    async fn exercise_commit_crashing(
        &mut self,
        rr: &comms::RequestResponse,
    ) -> anyhow::Result<()> {
        // Generate a changeset and the associated snapshot. Ask the agent to commit the changeset.
        let (snapshot, changeset) = self.state.gen_commit();

        // The agent should crash after `crash_time`ns.
        // If no data avaible crash after 300ms.
        let mut crash_time = self
            .tot_commit_time
            .checked_div(self.n_successfull_commit)
            .unwrap_or(Duration::from_millis(300).as_nanos() as u64);
        // Crash a little bit earlier than the average commit time to increase the
        // possibilities of crashing during sync.
        crash_time = (crash_time as f64 * 0.98) as u64;

        rr.send_request(crate::message::ToAgent::Commit(
            crate::message::CommitPayload {
                changeset: changeset.clone(),
                should_crash: Some(crash_time),
            },
        ))
        .await?;

        // Wait until the agent dies. We give it a grace period of 5 seconds.
        //
        // Note that we don't "take" the agent from the `workload_agent` place. This is because there is
        // always a looming possibility of SIGINT arriving.
        const TOLERANCE: Duration = Duration::from_secs(5);
        let agent = self.agent.as_mut().unwrap();
        let agent_died_or_timeout = timeout(TOLERANCE, agent.died()).await;
        self.agent.take().unwrap().teardown().await;
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
            self.ensure_changeset_applied(rr, &changeset).await?;
            self.state.commit(snapshot);
        } else {
            info!(
                "commit. seqno ours: {}, theirs: {}",
                snapshot.sync_seqn, seqno
            );
            self.ensure_changeset_reverted(rr, &changeset).await?;
        }
        Ok(())
    }

    async fn ensure_changeset_reverted(
        &self,
        rr: &comms::RequestResponse,
        changeset: &Vec<KeyValueChange>,
    ) -> anyhow::Result<()> {
        // Given a reverted changeset, we need to ensure that each modified/deleted key
        // is equal to its previous state and that each new key is not available.
        for change in changeset {
            match change {
                KeyValueChange::Insert(key, _value) => {
                    // The current value must be equal to the previous one.
                    let current_value = rr.send_request_query(*key).await?;
                    match self.state.committed.state.get(key) {
                        None | Some(None) if current_value.is_some() => {
                            return Err(anyhow::anyhow!("New inserted item should not be present"));
                        }
                        Some(prev_value) if current_value != *prev_value => {
                            return Err(anyhow::anyhow!(
                                "Modified item should be reverted to previous state"
                            ));
                        }
                        _ => (),
                    }
                }
                KeyValueChange::Delete(key) => {
                    // UNWRAP: Non existing keys are not deleted.
                    let prev_value = self.state.committed.state.get(key).unwrap();
                    assert!(prev_value.is_some());
                    if rr.send_request_query(*key).await?.as_ref() != prev_value.as_ref() {
                        return Err(anyhow::anyhow!(
                            "Deleted item should be reverted to previous state"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    async fn ensure_snapshot_validity(&self, rr: &comms::RequestResponse) -> anyhow::Result<()> {
        for (i, (key, expected_value)) in (self.state.committed.state.iter()).enumerate() {
            let value = rr.send_request_query(*key).await?;
            if &value != expected_value {
                return Err(anyhow::anyhow!(
                    "Wrong {}ith key in snapshot,\n key: {:?},\n expected value: {:?},\n found value: {:?}",
                    i,
                    hex::encode(key),
                    expected_value.as_ref().map(hex::encode),
                    value.as_ref().map(hex::encode),
                ));
            }
        }
        Ok(())
    }

    async fn spawn_new_agent(&mut self) -> anyhow::Result<()> {
        assert!(self.agent.is_none());
        controller::spawn_agent_into(&mut self.agent).await?;
        let workdir = self.workdir.path().display().to_string();
        self.agent
            .as_mut()
            .unwrap()
            .init(workdir, self.workload_id, self.bitbox_seed)
            .await?;

        self.ensure_snapshot_validity(self.agent.as_ref().unwrap().rr())
            .await?;

        Ok(())
    }

    /// Release potentially held resources.
    async fn teardown(&mut self) {
        if let Some(agent) = self.agent.take() {
            agent.teardown().await;
        }
    }

    /// Return the working directory.
    pub fn into_workdir(self) -> TempDir {
        self.workdir
    }
}
