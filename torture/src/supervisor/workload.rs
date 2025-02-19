use anyhow::Result;
use imbl::OrdMap;
use rand::{distributions::WeightedIndex, prelude::*};
use std::time::Duration;
use tokio::time::{error::Elapsed, timeout};
use tokio_util::sync::CancellationToken;
use tracing::{info, trace, trace_span, Instrument as _};

use crate::{
    message::{InitOutcome, KeyValueChange, OpenOutcome, ToSupervisor},
    supervisor::{
        cli::WorkloadParams,
        comms,
        controller::{self, SpawnedAgentController},
        dump_changest, WorkloadDir,
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
    /// Distribution used when generating a new key to decide how many bytes needs to be shared
    /// with an already existing key.
    new_key_distribution: WeightedIndex<usize>,
    /// When executing a workload iteration ,this is the probability of executing a rollback.
    rollback: f64,
    /// When executing a commit this is the probability of causing it to crash.
    commit_crash: f64,
    /// When executing a rollback this is the probability of causing it to crash.
    rollback_crash: f64,
    /// The probability of turning on the `ENOSPC` error.
    enospc_on: f64,
    /// The probability of turning off the `ENOSPC` error.
    enospc_off: f64,
}

impl Biases {
    fn new(
        delete: u8,
        overflow: u8,
        new_key: u8,
        rollback: u8,
        commit_crash: u8,
        rollback_crash: u8,
    ) -> Self {
        // When generating a new key to be inserted in the database,
        // this distribution will generate the key.
        // There is a 25% chance that the key is completely random,
        // half of the 25% chance that the first byte will be shared with an existing key,
        // one third of the 25% chance that two bytes will be shared with an existing key,
        // and so on.
        //
        // There are:
        // + 25% probability of having a key with 0 shared bytes.
        // + 48% probability of having a key with 1 to 9 shared bytes.
        // + 27% probability of having a key with more than 10 shared bytes.
        //
        // UNWRAP: provided iterator is not empty, no item is lower than zero
        // and the total sum is greater than one.
        let new_key_distribution = WeightedIndex::new((1usize..33).map(|x| (32 * 32) / x)).unwrap();

        Self {
            delete: (delete as f64) / 100.0,
            overflow: (overflow as f64) / 100.0,
            new_key: (new_key as f64) / 100.0,
            rollback: (rollback as f64) / 100.0,
            commit_crash: (commit_crash as f64) / 100.0,
            rollback_crash: (rollback_crash as f64) / 100.0,
            new_key_distribution,
            enospc_on: 0.1,
            enospc_off: 0.2,
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
    /// All committed key values.
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
        let mut key = [0; 32];
        // Generate a Delete KeyValueChange
        if !self.committed.state.is_empty() && self.rng.gen_bool(self.biases.delete) {
            loop {
                self.rng.fill_bytes(&mut key);
                if let Some((next_key, Some(_))) = self.committed.state.get_next(&key) {
                    return KeyValueChange::Delete(*next_key);
                }
            }
        }

        // Generate a new key KeyValueChange
        if self.committed.state.is_empty() {
            self.rng.fill_bytes(&mut key);
            return KeyValueChange::Insert(key, self.gen_value());
        }

        if self.rng.gen_bool(self.biases.new_key) {
            loop {
                self.rng.fill_bytes(&mut key);

                let Some(next_key) = self.committed.state.get_next(&key).map(|(k, _)| *k) else {
                    continue;
                };

                let common_bytes =
                    self.rng.sample(self.biases.new_key_distribution.clone()) as usize;
                key[..common_bytes].copy_from_slice(&next_key[..common_bytes]);

                if !self.committed.state.contains_key(&key) {
                    return KeyValueChange::Insert(key, self.gen_value());
                }
            }
        }

        // Generate an update KeyValueChange
        loop {
            self.rng.fill_bytes(&mut key);
            if let Some((next_key, _)) = self.committed.state.get_next(&key) {
                return KeyValueChange::Insert(*next_key, self.gen_value());
            }
        }
    }

    fn gen_value(&mut self) -> Vec<u8> {
        // MAX_LEAF_VALUE_SIZE is 1332,
        // thus every value size bigger than this will create an overflow value.
        let len = if self.rng.gen_bool(self.biases.overflow) {
            self.rng.gen_range(1333..32 * 1024)
        } else {
            self.rng.gen_range(1..1333)
        };
        let mut value = vec![0; len];
        self.rng.fill_bytes(&mut value);
        value
    }

    fn rollback(&mut self, snapshot: Snapshot) {
        // The application of a rollback counts as increased sync_seq.
        self.committed.sync_seqn += 1;
        self.committed.state = snapshot.state;
    }

    fn commit(&mut self, snapshot: Snapshot) {
        self.committed = snapshot;
    }
}

/// This is a struct that controls workload execution.
///
/// A workload is a set of tasks that the agents should perform. We say agents, plural, because
/// the same workload can be executed by multiple agents. However, it's always sequential. This
/// arises from the fact that as part of the workload we need to crash the agent to check how
/// it behaves.
pub struct Workload {
    /// Working directory for this particular workload.
    workload_dir: WorkloadDir,
    /// The handle to the trickfs FUSE FS.
    ///
    /// `Some` until the workload is torn down.
    trick_handle: Option<trickfs::TrickHandle>,
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
    /// Data collected to evaluate the average commit time.
    tot_commit_time: Duration,
    n_successfull_commit: u64,
    /// Whether to ensure the correct application of the changeset after every commit.
    ensure_changeset: bool,
    /// Whether to ensure the correctness of the entire state after every crash or rollback.
    ensure_snapshot: bool,
    /// Whether the trickfs is currently configured to return `ENOSPC` errors for every write.
    enabled_enospc: bool,
    /// Whether to randomly sample the state after every crash or rollback.
    sample_snapshot: bool,
    /// The max number of commits involved in a rollback.
    max_rollback_commits: u32,
    /// If `Some` there is rollback waiting to be applied,
    /// possibly alongside the delay after which the rollback process should panic.
    scheduled_rollback: Option<(ScheduledRollback, Option<Duration>)>,
}

/// Contains the information required to apply a rollback.
struct ScheduledRollback {
    /// The sync_seqn at which the rollback will be applied.
    sync_seqn: u32,
    /// The number of commits the rollback will revert.
    n_commits: usize,
    /// The state at which the state is expected to be found after the rollback is applied.
    snapshot: Snapshot,
}

impl Workload {
    pub fn new(
        seed: u64,
        workload_dir: WorkloadDir,
        workload_params: &WorkloadParams,
        workload_id: u64,
    ) -> anyhow::Result<Self> {
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
            workload_params.rollback,
            workload_params.commit_crash,
            workload_params.rollback_crash,
        );
        let mut state = WorkloadState::new(
            seed,
            workload_biases,
            workload_params.size,
            workload_params.random_size,
        );
        let bitbox_seed = state.gen_bitbox_seed();

        #[cfg(target_os = "linux")]
        let trick_handle = workload_params
            .trickfs
            .then(|| trickfs::spawn_trick(&workload_dir.path()))
            .transpose()?;

        #[cfg(not(target_os = "linux"))]
        let trick_handle = None;

        Ok(Self {
            workload_dir,
            trick_handle,
            agent: None,
            iterations: workload_params.iterations,
            state,
            workload_id,
            bitbox_seed,
            tot_commit_time: Duration::ZERO,
            n_successfull_commit: 0,
            ensure_changeset: workload_params.ensure_changeset,
            ensure_snapshot: workload_params.ensure_snapshot,
            sample_snapshot: workload_params.sample_snapshot,
            max_rollback_commits: workload_params.max_rollback_commits,
            scheduled_rollback: None,
            enabled_enospc: false,
        })
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
        trace!("run_iteration");

        if self.scheduled_rollback.as_ref().map_or(false, |(r, _)| {
            r.sync_seqn == self.state.committed.sync_seqn
        }) {
            // UNWRAP: scheduled_rollback has just be checked to be `Some`
            let (scheduled_rollback, should_crash) = self.scheduled_rollback.take().unwrap();
            self.exercise_rollback(&rr, scheduled_rollback, should_crash)
                .await?;
            return Ok(());
        }

        if self.trick_handle.is_some() {
            if self.enabled_enospc {
                let should_turn_off = self.state.rng.gen_bool(self.state.biases.enospc_off);
                if should_turn_off {
                    info!("unsetting ENOSPC");
                    self.enabled_enospc = false;
                    self.trick_handle
                        .as_ref()
                        .unwrap()
                        .set_trigger_enospc(false);
                }
            } else {
                let should_turn_on = self.state.rng.gen_bool(self.state.biases.enospc_on);
                if should_turn_on {
                    info!("setting ENOSPC");
                    self.enabled_enospc = true;
                    self.trick_handle.as_ref().unwrap().set_trigger_enospc(true);
                }
            }
        }

        // Do not schedule new rollbacks if they are already scheduled.
        let is_rollback_scheduled = self.scheduled_rollback.is_some();
        if !is_rollback_scheduled && self.state.rng.gen_bool(self.state.biases.rollback) {
            let should_crash = self.state.rng.gen_bool(self.state.biases.rollback_crash);
            self.schedule_rollback(should_crash).await?
        }

        let should_crash = self.state.rng.gen_bool(self.state.biases.commit_crash);
        self.exercise_commit(&rr, should_crash).await?;

        Ok(())
    }

    /// Commit a changeset.
    async fn exercise_commit(
        &mut self,
        rr: &comms::RequestResponse,
        should_crash: bool,
    ) -> anyhow::Result<()> {
        let should_crash = if should_crash {
            trace!("exercising commit crash");
            Some(self.get_crash_delay())
        } else {
            trace!("exercising commit");
            None
        };

        // Generate a changeset and the associated snapshot
        let (snapshot, changeset) = self.state.gen_commit();
        let commit_response = rr
            .send_request(crate::message::ToAgent::Commit(
                crate::message::CommitPayload {
                    changeset: changeset.clone(),
                    should_crash,
                },
            ))
            .await?;

        let is_applied = if should_crash.is_some() {
            let ToSupervisor::Ack = commit_response else {
                return Err(anyhow::anyhow!("Commit crash did not execute successfully"));
            };

            self.wait_for_crash().await?;

            // During a commit crash, every type of error could happen.
            // However the agent will be respawned, so it will just
            // make sure the changeset was correctly applied or reverted.
            self.spawn_new_agent().await?;
            let agent = self.agent.as_ref().unwrap();
            let rr = agent.rr().clone();

            // Sample the agent to make sure the changeset was correctly applied or reverted.
            let agent_sync_seqn = rr.send_query_sync_seqn().await?;
            if snapshot.sync_seqn == agent_sync_seqn {
                true
            } else if self.state.committed.sync_seqn == agent_sync_seqn {
                false
            } else {
                return Err(anyhow::anyhow!("Unexpected sync_seqn after commit crash",));
            }
        } else {
            let ToSupervisor::CommitResponse { elapsed, outcome } = commit_response else {
                dump_changest(&changeset, self.workload_dir.path())?;
                return Err(anyhow::anyhow!("Commit did not execute successfully"));
            };

            // Keep track of ENOSPC because the flag could be erased during the agent's respawn
            let was_enospc_enabled = self.enabled_enospc;
            self.ensure_outcome_validity(rr, &outcome).await?;

            if matches!(outcome, crate::message::Outcome::Success) {
                self.n_successfull_commit += 1;
                self.tot_commit_time += elapsed;
            }

            // Sample the agent to make sure the changeset was correctly applied or reverted.
            let agent_sync_seqn = rr.send_query_sync_seqn().await?;
            if was_enospc_enabled {
                false
            } else if snapshot.sync_seqn == agent_sync_seqn {
                true
            } else {
                dump_changest(&changeset, self.workload_dir.path())?;
                return Err(anyhow::anyhow!("Unexpected sync_seqn after commit"));
            }
        };

        if is_applied {
            self.ensure_changeset_applied(&rr, &changeset).await?;
            self.state.commit(snapshot);
        } else {
            self.ensure_changeset_reverted(&rr, &changeset).await?;
        }

        Ok(())
    }

    fn get_crash_delay(&self) -> Duration {
        // The agent should crash after `crash_delay`ns.
        // If no data avaible crash after 300ms.
        let mut crash_delay_millis = self
            .tot_commit_time
            .as_millis()
            .checked_div(self.n_successfull_commit as u128)
            .unwrap_or(300) as u64;
        // Crash a little bit earlier than the average commit time to increase the
        // possibilities of crashing during sync.
        crash_delay_millis = (crash_delay_millis as f64 * 0.98) as u64;
        Duration::from_millis(crash_delay_millis)
    }

    async fn ensure_outcome_validity(
        &mut self,
        rr: &comms::RequestResponse,
        outcome: &crate::message::Outcome,
    ) -> Result<()> {
        match outcome {
            crate::message::Outcome::Success => {
                // The operation was successful.
                if self.enabled_enospc {
                    return Err(anyhow::anyhow!("Operation should have failed with ENOSPC"));
                }
            }
            crate::message::Outcome::StorageFull => {
                if !self.enabled_enospc {
                    return Err(anyhow::anyhow!("Operation should have succeeded"));
                }

                // At this point, we expect the agent will have its NOMT instance poisoned.
                //
                // But we still should be able to make the sync_seqn and the kv queries.
                let agent_sync_seqn = rr.send_query_sync_seqn().await?;
                if self.state.committed.sync_seqn != agent_sync_seqn {
                    return Err(anyhow::anyhow!(
                        "Unexpected sync_seqn after failed operation with ENOSPC"
                    ));
                }

                // Now we instruct the agent to re-open NOMT and check if the sync_seqn is still the same.
                // We will reuse the same process.
                self.ensure_agent_open_db().await?;

                // Verify that the sync_seqn is still the same for the second time.
                let agent_sync_seqn = rr.send_query_sync_seqn().await?;
                if self.state.committed.sync_seqn != agent_sync_seqn {
                    return Err(anyhow::anyhow!(
                        "Unexpected sync_seqn after failed operation with ENOSPC"
                    ));
                }
            }
            crate::message::Outcome::UnknownFailure(err) => {
                return Err(anyhow::anyhow!(
                    "Operation failed due to unknown reasons: {}",
                    err
                ));
            }
        }
        Ok(())
    }

    async fn schedule_rollback(&mut self, should_crash: bool) -> anyhow::Result<()> {
        let n_commits_to_rollback = self.state.rng.gen_range(1..self.max_rollback_commits) as usize;

        let last_snapshot = &self.state.committed;
        let rollback_sync_seqn = last_snapshot.sync_seqn + n_commits_to_rollback as u32;
        let scheduled_rollback = ScheduledRollback {
            sync_seqn: rollback_sync_seqn,
            n_commits: n_commits_to_rollback,
            snapshot: last_snapshot.clone(),
        };

        let maybe_crash_delay = if should_crash {
            // TODO: more complex crash delay evaluation for rollbacks.
            Some(Duration::from_millis(10))
        } else {
            None
        };

        self.scheduled_rollback = Some((scheduled_rollback, maybe_crash_delay));

        trace!(
            "scheduled rollback {}for sync_seqn: {} of {} commits",
            if should_crash { "crash " } else { "" },
            rollback_sync_seqn,
            n_commits_to_rollback,
        );

        Ok(())
    }

    async fn exercise_rollback(
        &mut self,
        rr: &comms::RequestResponse,
        scheduled_rollback: ScheduledRollback,
        should_crash: Option<Duration>,
    ) -> anyhow::Result<()> {
        let ScheduledRollback {
            n_commits: n_commits_to_rollback,
            snapshot,
            ..
        } = scheduled_rollback;

        let maybe_crash_text = if should_crash.is_some() { " crash" } else { "" };
        trace!(
            "exercising rollback{} of {} commits",
            maybe_crash_text,
            n_commits_to_rollback
        );

        let rollback_outcome = rr
            .send_request(crate::message::ToAgent::Rollback(
                crate::message::RollbackPayload {
                    n_commits: n_commits_to_rollback,
                    should_crash: should_crash.clone(),
                },
            ))
            .await?;

        if should_crash.is_some() {
            let ToSupervisor::Ack = rollback_outcome else {
                return Err(anyhow::anyhow!(
                    "RollbackCommit crash did not execute successfully"
                ));
            };

            self.wait_for_crash().await?;

            // During a rollback crash, every type of error could happen.
            // However the agent will be respawned, so it will just
            // make sure the rollback was correctly applied or not.
            self.spawn_new_agent().await?;
            let agent = self.agent.as_ref().unwrap();
            let rr = agent.rr().clone();

            let agent_sync_seqn = rr.send_query_sync_seqn().await?;
            let last_sync_seqn = self.state.committed.sync_seqn;
            if agent_sync_seqn == last_sync_seqn + 1 {
                // sync_seqn has increased, so the rollback is expected to be applied correctly
                self.state.rollback(snapshot);
            } else if agent_sync_seqn == last_sync_seqn {
                // The rollback successfully crashed.
                info!("rollback crashed, seqno: {}", last_sync_seqn);
            } else {
                return Err(anyhow::anyhow!(
                    "Unexpected sync_seqn after rollback{}",
                    maybe_crash_text
                ));
            }
        } else {
            let ToSupervisor::RollbackResponse { outcome } = rollback_outcome else {
                return Err(anyhow::anyhow!(
                    "RollbackCommit did not execute successfully"
                ));
            };

            let was_enospc_enabled = self.enabled_enospc;
            self.ensure_outcome_validity(rr, &outcome).await?;

            let agent_sync_seqn = rr.send_query_sync_seqn().await?;
            let last_sync_seqn = self.state.committed.sync_seqn;
            if agent_sync_seqn == last_sync_seqn + 1 {
                // sync_seqn has increased, so the rollback is expected to be applied correctly
                self.state.rollback(snapshot);
            } else if agent_sync_seqn == last_sync_seqn {
                if !was_enospc_enabled {
                    return Err(anyhow::anyhow!(
                        "Rollback was expexted to succeed. Unexpected sync_seqn",
                    ));
                }
                // The rollback successfully crashed.
                info!("rollback crashed, seqno: {}", last_sync_seqn);
            } else {
                return Err(anyhow::anyhow!(
                    "Unexpected sync_seqn after rollback{}",
                    maybe_crash_text
                ));
            }
        }

        self.ensure_snapshot_validity(&rr).await?;
        Ok(())
    }

    async fn wait_for_crash(&mut self) -> anyhow::Result<()> {
        // Wait until the agent dies. We give it a grace period of 5 seconds.
        //
        // Note that we don't "take" the agent from the `workload_agent` place. This is because there is
        // always a looming possibility of SIGINT arriving.
        const TOLERANCE: Duration = Duration::from_secs(5);
        let agent = self.agent.as_mut().unwrap();
        let agent_died_or_timeout = timeout(TOLERANCE, agent.died()).await;
        self.agent.take().unwrap().teardown().await;
        if let Err(Elapsed { .. }) = agent_died_or_timeout {
            return Err(anyhow::anyhow!("agent did not die"));
        }

        Ok(())
    }

    async fn ensure_changeset_applied(
        &self,
        rr: &comms::RequestResponse,
        changeset: &Vec<KeyValueChange>,
    ) -> anyhow::Result<()> {
        if !self.ensure_changeset {
            return Ok(());
        }

        for change in changeset {
            match change {
                KeyValueChange::Insert(key, value)
                    if rr.send_request_query(*key).await?.as_ref() != Some(&value) =>
                {
                    dump_changest(&changeset, self.workload_dir.path())?;
                    return Err(anyhow::anyhow!("Inserted item not present after commit"));
                }
                KeyValueChange::Delete(key) if rr.send_request_query(*key).await?.is_some() => {
                    dump_changest(&changeset, self.workload_dir.path())?;
                    return Err(anyhow::anyhow!("Deleted item still present after commit"));
                }
                _ => (),
            }
        }
        Ok(())
    }

    async fn ensure_changeset_reverted(
        &self,
        rr: &comms::RequestResponse,
        changeset: &Vec<KeyValueChange>,
    ) -> anyhow::Result<()> {
        if !self.ensure_changeset {
            return Ok(());
        }

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

    async fn ensure_snapshot_validity(
        &mut self,
        rr: &comms::RequestResponse,
    ) -> anyhow::Result<()> {
        if !self.ensure_snapshot && !self.sample_snapshot {
            return Ok(());
        }

        let expected_sync_seqn = self.state.committed.sync_seqn;
        let sync_seqn = rr.send_query_sync_seqn().await?;
        if expected_sync_seqn != sync_seqn {
            return Err(anyhow::anyhow!(
                "Unexpected sync_seqn while ensuring snapshot validity, expected: {}, found: {}",
                expected_sync_seqn,
                sync_seqn
            ));
        }

        if self.ensure_snapshot {
            return self.check_entire_snapshot(rr).await;
        }

        self.check_sampled_snapshot(rr).await
    }

    async fn check_entire_snapshot(&self, rr: &comms::RequestResponse) -> anyhow::Result<()> {
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

    async fn check_sampled_snapshot(&mut self, rr: &comms::RequestResponse) -> anyhow::Result<()> {
        let mut key = [0; 32];
        // The amount of items randomly sampled is equal to 5% of the entire state size.
        let sample_check_size = (self.state.committed.state.len() as f64 * 0.05) as usize;
        for _ in 0..sample_check_size {
            let (key, expected_value) = loop {
                self.state.rng.fill_bytes(&mut key);
                if let Some((next_key, Some(expected_value))) =
                    self.state.committed.state.get_next(&key)
                {
                    break (next_key, expected_value);
                }
            };

            let value = rr.send_request_query(*key).await?;
            if value.as_ref().map_or(true, |v| v != expected_value) {
                return Err(anyhow::anyhow!(
                    "Wrong key in snapshot,\n key: {:?},\n expected value: {:?},\n found value: {:?}",
                    hex::encode(key),
                    hex::encode(expected_value),
                    value.as_ref().map(hex::encode),
                ));
            }
        }
        Ok(())
    }

    async fn spawn_new_agent(&mut self) -> anyhow::Result<()> {
        assert!(self.agent.is_none());
        controller::spawn_agent_into(&mut self.agent).await?;
        let outcome = self
            .agent
            .as_mut()
            .unwrap()
            .init(self.workload_dir.path(), self.workload_id)
            .await?;
        if let InitOutcome::Success = outcome {
            ()
        } else {
            return Err(anyhow::anyhow!("Unexpected init outcome: {:?}", outcome));
        }

        // Finally, make the agent open the database.
        self.ensure_agent_open_db().await?;

        Ok(())
    }

    /// Ensure that the agent has opened the database.
    ///
    /// If the agent has run out of storage, we will turn off the `ENOSPC` error and try again.
    async fn ensure_agent_open_db(&mut self) -> anyhow::Result<()> {
        let rollback = if self.state.biases.rollback > 0.0 {
            Some(self.max_rollback_commits)
        } else {
            None
        };
        let outcome = self
            .agent
            .as_mut()
            .unwrap()
            .open(self.bitbox_seed, rollback)
            .await?;

        match outcome {
            OpenOutcome::Success => (),
            OpenOutcome::StorageFull => {
                // We got storage full and we know here that enospc is enabled.
                //
                // We assume this is trickfs and thus we can turn it off, otherwise,
                // we won't be able to continue.
                assert!(self.enabled_enospc);
                self.enabled_enospc = false;
                self.trick_handle
                    .as_ref()
                    .unwrap()
                    .set_trigger_enospc(false);

                let outcome = self
                    .agent
                    .as_mut()
                    .unwrap()
                    .open(self.bitbox_seed, rollback)
                    .await?;
                assert!(matches!(outcome, OpenOutcome::Success));
            }
            OpenOutcome::UnknownFailure(err) => {
                return Err(anyhow::anyhow!("unexpected open outcome: {:?}", err));
            }
        }

        Ok(())
    }

    /// Release potentially held resources.
    async fn teardown(&mut self) {
        if let Some(agent) = self.agent.take() {
            agent.teardown().await;
        }
        if let Some(trick_handle) = self.trick_handle.take() {
            tokio::task::block_in_place(move || {
                trick_handle.unmount_and_join();
            });
        }
    }

    /// Return the working directory.
    pub fn into_workload_dir(self) -> WorkloadDir {
        self.workload_dir
    }
}
