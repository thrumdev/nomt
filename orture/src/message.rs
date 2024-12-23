//! This module declares messages that are exchanged between the supervisor and child processes.
//!
//! This is used only for inter-process communication and thus doesn't need to care about versioning
//! or compatibility.

use serde::{Deserialize, Serialize};

pub type Key = [u8; 32];
pub type Value = Vec<u8>;

/// A change in the key-value store.
#[derive(Debug, Serialize, Deserialize)]
pub enum KeyValueChange {
    /// A key-value pair was inserted.
    Insert(Key, Value),
    /// A key-value pair was deleted.
    Delete(Key),
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
}

/// The supervisor sends this message to the child process to indicate that the child should
/// commit.
#[derive(Debug, Serialize, Deserialize)]
pub struct CommitPayload {
    /// The set of changes that the child should commit.
    ///
    /// There must be no duplicate keys in the set.
    pub changset: Vec<KeyValueChange>,
    /// Whether the supervisor expects the child to crash during the commit.
    pub should_crash: bool,
}

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
    /// do a clean shutdown.
    GracefulShutdown,
}

pub enum ToSupervisor {
    Ack,
}
