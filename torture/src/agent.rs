use anyhow::{anyhow, bail, Result};
use futures::SinkExt as _;
use nomt::{Blake3Hasher, Nomt};
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
    self, CommitPayload, Envelope, InitPayload, KeyValueChange, ToAgent, ToSupervisor,
    MAX_ENVELOPE_SIZE,
};

/// The entrypoint for the agent.
///
/// `input` is the UnixStream that the agent should use to communicate with its supervisor.
pub async fn run(input: UnixStream) -> Result<()> {
    // Make the process non-dumpable.
    //
    // We expect this process to abort on a crash, so we don't want to leave lots of core dumps
    // behind.
    #[cfg(target_os = "linux")]
    nix::sys::prctl::set_dumpable(false)?;

    let mut stream = Stream::new(input);
    let mut agent = recv_init(&mut stream).await?;

    crate::logging::init_agent(&agent.id, &agent.workdir);
    let pid = std::process::id();
    trace!(
        pid,
        bitbox_seed = hex::encode(&agent.bitbox_seed),
        "Child process started"
    );

    loop {
        // TODO: make the message processing non-blocking.
        //
        // That being said, I should probably note, that it doesn't necessarily mean we should allow
        // concurrent access to NOMT (only one session could be created at once anyway).
        let Envelope { reqno, message } = stream.recv().await?;
        match message {
            ToAgent::Commit(CommitPayload {
                changeset,
                should_crash: Some(crash_time),
            }) => {
                // TODO: implement this in the future.
                //
                // This seems to be a big feature. As of now, I envision it, as we either:
                //
                // 1. Launch a concurrent process that brings down the process (e.g. via `abort`) at
                //    some non-deterministic time in the future.
                // 2. Adjust the VFS settings so that it will crash at a specific moment. This might
                //    be some specific moment like writing to Meta or maybe writing to some part of HT.
                // 3. Introduce fail point to do the same.

                // Ack first.
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::Ack,
                    })
                    .await?;

                let barrier_1 = Arc::new(tokio::sync::Barrier::new(2));
                let barrier_2 = barrier_1.clone();
                let task_1 = tokio::spawn(async move {
                    barrier_1.wait().await;
                    // Wait for a random time and then abort.
                    // TODO: reduce non-determinism by using the supervisor generated seed.
                    let crash_time = Duration::from_nanos(crash_time);
                    sleep(crash_time).await;
                    tracing::info!("aborting after {}ms", crash_time.as_millis());
                    std::process::abort();
                });
                let task_2 = tokio::spawn(async move {
                    barrier_2.wait().await;
                    let start = std::time::Instant::now();
                    let _ = agent.commit(changeset).await;
                    let elapsed = start.elapsed();
                    tracing::info!("commit took {}ms", elapsed.as_millis());
                    std::process::abort();
                });
                let _ = tokio::join!(task_1, task_2);
                // This is unreachable because either the process will abort due to a timed
                // abort or due to the commit finishing successfully (and then aborting).
                unreachable!();
            }
            ToAgent::Commit(CommitPayload {
                changeset,
                should_crash: None,
            }) => {
                let start = std::time::Instant::now();
                agent.commit(changeset).await?;
                let elapsed = start.elapsed();
                tracing::info!("commit took {}ms", elapsed.as_millis());
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::CommitSuccessful(elapsed.as_nanos() as u64),
                    })
                    .await?;
            }
            ToAgent::Rollback(n_blocks) => {
                let start = std::time::Instant::now();
                agent.rollback(n_blocks)?;
                tracing::info!("rollback took {}ms", start.elapsed().as_millis());
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::Ack,
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
            ToAgent::Init(init) => bail!("unexpected init message, id={}", init.id),
        }
    }
    Ok(())
}

/// Receives the [`ToAgent::Init`] message from the supervisor and returns the initialized agent. Sends
/// an Ack message back to the supervisor.
///
/// # Errors
///
/// Returns an error if the message is not an Init message or if the message is not received
/// within a certain time limit.
async fn recv_init(stream: &mut Stream) -> Result<Agent> {
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
    let agent = Agent::new(init)?;
    stream
        .send(Envelope {
            reqno,
            message: ToSupervisor::Ack,
        })
        .await?;
    Ok(agent)
}

struct Agent {
    workdir: PathBuf,
    nomt: Nomt<Blake3Hasher>,
    id: String,
    bitbox_seed: [u8; 16],
}

impl Agent {
    fn new(init: InitPayload) -> Result<Self> {
        let workdir = PathBuf::from(&init.workdir);
        if !workdir.exists() {
            bail!("workdir does not exist: {:?}", workdir);
        }
        let mut o = nomt::Options::new();
        o.path(workdir.join("nomt_db"));
        o.bitbox_seed(init.bitbox_seed);
        o.hashtable_buckets(500_000);
        o.rollback(init.rollback);
        let nomt = Nomt::open(o)?;
        Ok(Self {
            workdir,
            nomt,
            id: init.id,
            bitbox_seed: init.bitbox_seed,
        })
    }

    async fn commit(&mut self, changeset: Vec<KeyValueChange>) -> Result<()> {
        let session = self.nomt.begin_session();
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

        tokio::task::block_in_place(|| self.nomt.update_and_commit(session, actuals))?;

        Ok(())
    }

    fn rollback(&mut self, n_blocks: usize) -> Result<()> {
        self.nomt.rollback(n_blocks)?;
        Ok(())
    }

    fn query(&mut self, key: message::Key) -> Result<Option<message::Value>> {
        let value = self.nomt.read(key)?;
        Ok(value)
    }

    fn query_sync_seqn(&mut self) -> u32 {
        self.nomt.sync_seqn()
    }
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
