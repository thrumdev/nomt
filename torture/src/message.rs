//! This module declares messages that are exchanged between the supervisor and child processes.
//!
//! This is used only for inter-process communication and thus doesn't need to care about versioning
//! or compatibility.

use serde::{Deserialize, Serialize};

pub type Key = [u8; 32];
pub type Value = Vec<u8>;

/// A change in the key-value store.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum KeyValueChange {
    /// A key-value pair was inserted.
    Insert(Key, Value),
    /// A key-value pair was deleted.
    Delete(Key),
}

impl KeyValueChange {
    /// Returns the key that this changes pertains to.
    pub fn key(&self) -> &Key {
        match *self {
            KeyValueChange::Insert(ref key, _) | KeyValueChange::Delete(ref key) => key,
        }
    }

    pub fn value(&self) -> Option<Value> {
        match self {
            KeyValueChange::Insert(_, val) => Some(val.clone()),
            KeyValueChange::Delete(_) => None,
        }
    }
}

/// The parameters for the [`ToAgent::Init`] message.
#[derive(Debug, Serialize, Deserialize)]
pub struct InitPayload {
    /// ID string that can be used to identify the agent. This is used for logging and debugging.
    pub id: String,
    /// The directory where the child should store the data and other files (such as logs).
    ///
    /// The directory must exist.
    pub workdir: String,
    /// The seed that should be used for bitbox.
    ///
    /// Only used upon creation a new NOMT db.
    pub bitbox_seed: [u8; 16],
    /// Whether the agent is supposed to handle rollbacks.
    pub rollback: bool,
}

/// The supervisor sends this message to the child process to indicate that the child should
/// commit.
#[derive(Debug, Serialize, Deserialize)]
pub struct CommitPayload {
    /// The set of changes that the child should commit.
    ///
    /// There must be no duplicate keys in the set.
    pub changeset: Vec<KeyValueChange>,
    /// If Some the supervisor expects the commit to crash,
    /// the crash should happen after the specified amount of time.
    /// Time is specified in nanoseconds.
    pub should_crash: Option<u64>,
}

/// The maximum size of an envelope, in the serialized form.
pub const MAX_ENVELOPE_SIZE: usize = 128 * 1024 * 1024;

/// A wrapper around a message that adds a request number.
#[derive(Debug, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// The request number. This is used to match responses to requests.
    ///
    /// The request number is unique for each request. The response should have the same request
    /// number as the request that caused it.
    pub reqno: u64,
    /// The message itself.
    pub message: T,
}

/// Messages sent from the supervisor to the agent process.
#[derive(Debug, Serialize, Deserialize)]
pub enum ToAgent {
    /// The first message sent by the supervisor to the child process. Contains the parameters the
    /// supervisor expects the child to use.
    Init(InitPayload),
    /// The supervisor sends this message to the child process to indicate that the child should
    /// commit.
    Commit(CommitPayload),
    /// The supervisor sends this message to the child process to indicate that the child should
    /// perform a rollback of the number of specified blocks.
    Rollback(usize),
    /// The supervisor sends this message to the child process to query the value of a given key.
    Query(Key),
    /// The supervisor sends this message to the child process to query the current sequence number
    /// of the database.
    QuerySyncSeqn,
    /// The supervisor sends this message to the child process to indicate that the child should
    /// do a clean shutdown.
    GracefulShutdown,
}

/// Messages sent from the agent to the supervisor.
#[derive(Debug, Serialize, Deserialize)]
pub enum ToSupervisor {
    /// A generic acknowledgment message.
    Ack,
    /// The response to a successful commit, it contains the elapsed time to perform the commit.
    /// Time is measured in nanoseconds.
    CommitSuccessful(u64),
    /// The response to a query for a key-value pair.
    QueryValue(Option<Value>),
    /// The response to a query for the current sequence number of the database.
    SyncSeqn(u32),
}
