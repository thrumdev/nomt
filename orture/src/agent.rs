use anyhow::{anyhow, bail, Result};
use futures::SinkExt as _;
use nomt::{Blake3Hasher, Nomt};
use rand::Rng;
use std::{path::PathBuf, time::Duration};
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
    trace!(pid, "Child process started");

    loop {
        // TODO: make the message processing non-blocking.
        //
        // That being said, I should probably note, that it doesn't necessarily mean we should allow
        // concurrent access to NOMT (only one session could be created at once anyway).
        let Envelope { reqno, message } = stream.recv().await?;
        match message {
            ToAgent::Commit(commit) => {
                if commit.should_crash {
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
                    tokio::spawn(async move {
                        let _ = agent.commit(commit);
                    })
                    .await?;
                    // Wait for a random time and then abort.
                    let millis = rand::thread_rng().gen_range(0..400);
                    sleep(Duration::from_millis(millis)).await;
                    std::process::abort();
                } else {
                    agent.commit(commit)?;
                    stream
                        .send(Envelope {
                            reqno,
                            message: ToSupervisor::Ack,
                        })
                        .await?;
                }
            }
            ToAgent::Query(key) => {
                trace!("query: {}", hex::encode(key));
                let value = agent.query(key)?;
                stream
                    .send(Envelope {
                        reqno,
                        message: ToSupervisor::QueryResponse(value),
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
        let nomt = Nomt::open(o)?;
        Ok(Self {
            workdir,
            nomt,
            id: init.id,
        })
    }

    fn commit(&mut self, commit: CommitPayload) -> Result<()> {
        let session = self.nomt.begin_session();
        let mut actuals = Vec::with_capacity(commit.changeset.len());
        for change in commit.changeset {
            match change {
                KeyValueChange::Insert(key, value) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(Some(value))));
                }
                KeyValueChange::Delete(key) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(None)));
                }
            }
        }
        // sort by ascending key
        actuals.sort_by(|(a, _), (b, _)| a.cmp(b));
        self.nomt.commit(session, actuals)?;
        Ok(())
    }

    fn query(&mut self, key: message::Key) -> Result<Option<message::Value>> {
        let value = self.nomt.read(key)?;
        Ok(value)
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
            FramedRead::new(BufReader::new(rd), LengthDelimitedCodec::new()),
            SymmetricalBincode::default(),
        );
        let wr_stream = SymmetricallyFramed::new(
            FramedWrite::new(BufWriter::new(wr), LengthDelimitedCodec::new()),
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
