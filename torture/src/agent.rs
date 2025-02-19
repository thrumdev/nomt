use anyhow::{anyhow, bail, Result};
use futures::SinkExt as _;
use nomt::{hasher::Blake3Hasher, Nomt, SessionParams};
use std::future::Future;
use std::path::Path;
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::{
    io::{BufReader, BufWriter},
    net::{
        unix::{OwnedReadHalf, OwnedWriteHalf},
        UnixStream,
    },
    time::{error::Elapsed, sleep, timeout},
};
use tokio_serde::{formats::SymmetricalBincode, SymmetricallyFramed};
use tokio_stream::StreamExt as _;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};
use tracing::trace;

use crate::message::{
    self, CommitPayload, Envelope, InitOutcome, KeyValueChange, OpenOutcome, OpenPayload, Outcome,
    RollbackPayload, ToAgent, ToSupervisor, MAX_ENVELOPE_SIZE,
};

/// The entrypoint for the agent.
///
/// `input` is the UnixStream that the agent should use to communicate with its supervisor.
pub async fn run(input: UnixStream) -> Result<()> {
    let pid = std::process::id();
    trace!(pid, "Child process started");

    // Make the process non-dumpable.
    //
    // We expect this process to abort on a crash, so we don't want to leave lots of core dumps
    // behind.
    #[cfg(target_os = "linux")]
    nix::sys::prctl::set_dumpable(false)?;

    let mut stream = Stream::new(input);
    let workdir = initialize(&mut stream).await?;
    let mut agent = Agent::new();

    loop {
        // TODO: make the message processing non-blocking.
        //
        // That being said, I should probably note, that it doesn't necessarily mean we should allow
        // concurrent access to NOMT (only one session could be created at once anyway).
        let Envelope { reqno, message } = stream.recv().await?;
        match message {
            ToAgent::Commit(CommitPayload {
                changeset,
                should_crash: Some(crash_delay),
            }) => {
                // Ack first.
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::Ack,
                    })
                    .await?;

                let task = async move {
                    let start = std::time::Instant::now();
                    let _ = agent.commit(changeset).await;
                    let elapsed = start.elapsed();
                    tracing::info!("commit took {}ms", elapsed.as_millis());
                };

                crash_task(task, crash_delay, "commit").await;
                unreachable!();
            }
            ToAgent::Commit(CommitPayload {
                changeset,
                should_crash: None,
            }) => {
                let start = std::time::Instant::now();
                let outcome = agent.commit(changeset).await;
                let elapsed = start.elapsed();
                tracing::info!("commit took {}ms", elapsed.as_millis());
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::CommitResponse { elapsed, outcome },
                    })
                    .await?;
            }
            ToAgent::Rollback(RollbackPayload {
                n_commits,
                should_crash: Some(crash_delay),
            }) => {
                // Ack first.
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::Ack,
                    })
                    .await?;

                let task = async move {
                    let start = std::time::Instant::now();
                    let _ = agent.rollback(n_commits).await;
                    tracing::info!("rollback took {:?}", start.elapsed());
                };

                crash_task(task, crash_delay, "rollback").await;
                unreachable!();
            }
            ToAgent::Rollback(RollbackPayload {
                n_commits,
                should_crash: None,
            }) => {
                let start = std::time::Instant::now();
                let outcome = agent.rollback(n_commits).await;
                tracing::info!("rollback took {}ms", start.elapsed().as_millis());
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::RollbackResponse { outcome },
                    })
                    .await?;
            }
            ToAgent::Query(key) => {
                let value = agent.query(key)?;
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::QueryValue(value),
                    })
                    .await?;
            }
            ToAgent::QuerySyncSeqn => {
                let sync_seqn = agent.query_sync_seqn();
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::SyncSeqn(sync_seqn),
                    })
                    .await?;
            }
            ToAgent::GracefulShutdown => {
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::Ack,
                    })
                    .await?;
                drop(agent);
                break;
            }
            ToAgent::Open(open_params) => {
                tracing::info!("opening the database");
                let outcome = agent.perform_open(&workdir, Some(open_params)).await;
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::OpenResponse(outcome),
                    })
                    .await?;
            }
            ToAgent::Init(_) => {
                bail!("Unexpected Init message");
            }
        }
    }
    Ok(())
}

/// Execute the provided `task` and make it crash after the specified `crash_delay`.
async fn crash_task(
    task: impl Future<Output = ()> + Send + 'static,
    crash_delay: Duration,
    op: &'static str,
) {
    // TODO: implement this in the future.
    //
    // This seems to be a big feature. As of now, I envision it, as we either:
    //
    // 1. Launch a concurrent process that brings down the process (e.g. via `abort`) at
    //    some non-deterministic time in the future.
    // 2. Adjust the VFS settings so that it will crash at a specific moment. This might
    //    be some specific moment like writing to Meta or maybe writing to some part of HT.
    // 3. Introduce fail point to do the same.

    let barrier_1 = Arc::new(tokio::sync::Barrier::new(2));
    let barrier_2 = barrier_1.clone();
    let task_1 = tokio::spawn(async move {
        barrier_1.wait().await;
        sleep(crash_delay).await;
        tracing::info!("aborting {op} after {}ms", crash_delay.as_millis());
        std::process::abort();
    });
    let task_2 = tokio::spawn(async move {
        barrier_2.wait().await;
        task.await;
        std::process::abort();
    });
    let _ = tokio::join!(task_1, task_2);
    // This is unreachable because either the process will abort due to a timed
    // abort or due to the commit finishing successfully (and then aborting).
    unreachable!();
}

/// Performs the initialization of the agent.
///
/// Receives the [`ToAgent::Init`] message from the supervisor, initializes the logging, confirms
/// the receipt of the message, and returns the path to the working directory.
///
/// # Errors
///
/// Returns an error if the message is not an Init message or if the message is not received
/// within a certain time limit.
async fn initialize(stream: &mut Stream) -> Result<PathBuf> {
    const DEADLINE: Duration = Duration::from_secs(1);
    let Envelope { reqno, message } = match timeout(DEADLINE, stream.recv()).await {
        Ok(envelope) => envelope?,
        Err(Elapsed { .. }) => {
            anyhow::bail!("Timed out waiting for Init message");
        }
    };
    let ToAgent::Init(init) = message else {
        anyhow::bail!("Expected Init message");
    };

    crate::logging::init_agent(&init.id, &init.workdir);

    let workdir = PathBuf::from(&init.workdir);
    if !workdir.exists() {
        stream
            .send(Envelope {
                reqno,
                message: ToSupervisor::InitResponse(InitOutcome::WorkdirDoesNotExist),
            })
            .await?;
        bail!("Workdir does not exist");
    } else {
        stream
            .send(Envelope {
                reqno,
                message: ToSupervisor::InitResponse(InitOutcome::Success),
            })
            .await?;
        Ok(workdir)
    }
}

pub struct Agent {
    nomt: Option<Nomt<Blake3Hasher>>,
}

impl Agent {
    pub fn new() -> Self {
        Self { nomt: None }
    }

    pub async fn perform_open(
        &mut self,
        workdir: &Path,
        open_params: Option<OpenPayload>,
    ) -> OpenOutcome {
        if let Some(nomt) = self.nomt.take() {
            tracing::trace!("dropping the existing NOMT instance");
            drop(nomt);
        }

        let mut o = nomt::Options::new();
        o.path(workdir.join("nomt_db"));
        if let Some(open_params) = open_params {
            o.bitbox_seed(open_params.bitbox_seed);
            o.hashtable_buckets(500_000);
            if let Some(n_commits) = open_params.rollback {
                o.rollback(true);
                o.max_rollback_log_len(n_commits);
            } else {
                o.rollback(false);
            }
        }
        let nomt = match tokio::task::block_in_place(|| Nomt::open(o)) {
            Ok(nomt) => nomt,
            Err(ref err) if is_enospc(err) => return OpenOutcome::StorageFull,
            Err(ref err) => return OpenOutcome::UnknownFailure(err.to_string()),
        };
        self.nomt = Some(nomt);
        OpenOutcome::Success
    }

    pub async fn commit(&mut self, changeset: Vec<KeyValueChange>) -> Outcome {
        // UNWRAP: `nomt` is always `Some` except recreation.
        let nomt = self.nomt.as_ref().unwrap();
        let session = nomt.begin_session(SessionParams::default());
        let mut actuals = Vec::with_capacity(changeset.len());
        for change in changeset {
            match change {
                KeyValueChange::Insert(key, value) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(Some(value))));
                }
                KeyValueChange::Delete(key) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(None)));
                }
            }
        }

        // Perform the commit.
        let commit_result = tokio::task::block_in_place(|| session.finish(actuals)?.commit(&nomt));
        let commit_outcome = classify_result(commit_result);

        // Log the outcome if it was not successful.
        if !matches!(commit_outcome, Outcome::Success) {
            trace!("unsuccessful commit: {:?}", commit_outcome);
        }

        commit_outcome
    }

    async fn rollback(&mut self, n_commits: usize) -> Outcome {
        // UNWRAP: `nomt` is always `Some` except recreation.
        let nomt = self.nomt.as_ref().unwrap();

        // Perform the rollback.
        let rollback_result = tokio::task::block_in_place(|| nomt.rollback(n_commits));
        let rollback_outcome = classify_result(rollback_result);

        // Log the outcome if it was not successful.
        if !matches!(rollback_outcome, Outcome::Success) {
            trace!("unsuccessful rollback: {:?}", rollback_outcome);
        }

        rollback_outcome
    }

    pub fn query(&mut self, key: message::Key) -> Result<Option<message::Value>> {
        // UNWRAP: `nomt` is always `Some` except recreation.
        let nomt = self.nomt.as_ref().unwrap();
        let value = nomt.read(key)?;
        Ok(value)
    }

    fn query_sync_seqn(&mut self) -> u32 {
        // UNWRAP: `nomt` is always `Some` except recreation.
        let nomt = self.nomt.as_ref().unwrap();
        nomt.sync_seqn()
    }
}

/// Classify an operation result into one of the outcome.
fn classify_result(operation_result: anyhow::Result<()>) -> Outcome {
    match operation_result {
        Ok(()) => Outcome::Success,
        Err(ref err) if is_enospc(err) => Outcome::StorageFull,
        Err(err) => Outcome::UnknownFailure(err.to_string()),
    }
}

/// Examines the given error to determine if it is an `ENOSPC` IO error.
fn is_enospc(err: &anyhow::Error) -> bool {
    let Some(io_err) = err.downcast_ref::<std::io::Error>() else {
        return false;
    };
    let Some(errno) = io_err.raw_os_error() else {
        return false;
    };
    errno == libc::ENOSPC
}

/// Abstraction over the stream of messages from the supervisor.
struct Stream {
    rd_stream: SymmetricallyFramed<
        FramedRead<BufReader<OwnedReadHalf>, LengthDelimitedCodec>,
        Envelope<ToAgent>,
        SymmetricalBincode<Envelope<ToAgent>>,
    >,
    wr_stream: SymmetricallyFramed<
        FramedWrite<BufWriter<OwnedWriteHalf>, LengthDelimitedCodec>,
        Envelope<ToSupervisor>,
        SymmetricalBincode<Envelope<ToSupervisor>>,
    >,
}

impl Stream {
    /// Creates a stream wrapper for the given unix stream.
    fn new(command_stream: UnixStream) -> Self {
        let (rd, wr) = command_stream.into_split();
        let rd_stream = SymmetricallyFramed::new(
            FramedRead::new(
                BufReader::new(rd),
                LengthDelimitedCodec::builder()
                    .length_field_length(8)
                    .max_frame_length(MAX_ENVELOPE_SIZE)
                    .new_codec(),
            ),
            SymmetricalBincode::default(),
        );
        let wr_stream = SymmetricallyFramed::new(
            FramedWrite::new(
                BufWriter::new(wr),
                LengthDelimitedCodec::builder()
                    .length_field_length(8)
                    .max_frame_length(MAX_ENVELOPE_SIZE)
                    .new_codec(),
            ),
            SymmetricalBincode::default(),
        );
        Self {
            rd_stream,
            wr_stream,
        }
    }

    async fn recv(&mut self) -> Result<Envelope<ToAgent>> {
        match self.rd_stream.try_next().await {
            Ok(Some(envelope)) => Ok(envelope),
            Ok(None) => bail!("EOF"),
            Err(e) => Err(anyhow!(e)),
        }
    }

    async fn send(&mut self, message: Envelope<ToSupervisor>) -> Result<()> {
        self.wr_stream.send(message).await?;
        Ok(())
    }
}
