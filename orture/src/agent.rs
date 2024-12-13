use anyhow::{bail, Result};
use nomt::{Blake3Hasher, Nomt};
use std::path::PathBuf;
use tokio::{io::BufStream, net::UnixStream};
use tokio_stream::StreamExt as _;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

use crate::message;

/// The entrypoint for the agent.
///
/// `input` is the UnixStream that the agent should use to communicate with its supervisor.
pub async fn run(input: UnixStream) -> Result<()> {
    let mut stream = Stream::new(input);
    // TODO: bail after on a timeout.
    let init = stream.recv_init().await?;
    let mut agent = Agent::new(init)?;
    loop {
        let message = stream.recv().await?;
        match message {
            message::ToAgent::Init(init) => bail!("unexpected init message, id={}", init.id),
            message::ToAgent::Commit(commit) => agent.commit(commit)?,
            message::ToAgent::GracefulShutdown => {
                println!("Received GracefulShutdown message");
                break;
            }
        }
    }
    Ok(())
}

struct Agent {
    workdir: PathBuf,
    nomt: Nomt<Blake3Hasher>,
}

impl Agent {
    fn new(init: message::InitPayload) -> Result<Self> {
        let workdir = PathBuf::from(&init.workdir);
        if !workdir.exists() {
            bail!("workdir does not exist: {:?}", workdir);
        }
        let mut o = nomt::Options::new();
        o.path(workdir.join("nomt_db"));
        o.bitbox_seed(init.bitbox_seed);
        let nomt = Nomt::open(o)?;
        Ok(Self { workdir, nomt })
    }

    fn commit(&mut self, commit: message::CommitPayload) -> Result<()> {
        let session = self.nomt.begin_session();
        let mut actuals = Vec::with_capacity(commit.changset.len());
        for change in commit.changset {
            match change {
                message::KeyValueChange::Insert(key, value) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(Some(value))));
                }
                message::KeyValueChange::Delete(key) => {
                    actuals.push((key, nomt::KeyReadWrite::Write(None)));
                }
            }
        }
        // sort by ascending key
        actuals.sort_by(|(a, _), (b, _)| a.cmp(b));
        self.nomt.commit(session, actuals)?;
        Ok(())
    }
}

/// Abstraction over the stream of messages from the supervisor.
struct Stream {
    framed: FramedRead<BufStream<UnixStream>, LengthDelimitedCodec>,
}

impl Stream {
    /// Creates a stream wrapper for the given unix stream.
    fn new(command_stream: UnixStream) -> Self {
        let framed = FramedRead::new(BufStream::new(command_stream), LengthDelimitedCodec::new());
        Self { framed }
    }

    /// Receives the [`message::Init`] message from the supervisor. If the message is not an Init
    /// message, returns an error.
    async fn recv_init(&mut self) -> Result<message::InitPayload> {
        match self.recv().await? {
            message::ToAgent::Init(init) => Ok(init),
            _ => Err(anyhow::anyhow!("Expected Init message")),
        }
    }

    async fn recv(&mut self) -> Result<message::ToAgent> {
        let bytes = self
            .framed
            .next()
            .await
            .ok_or_else(|| anyhow::anyhow!("EOF"))??;
        Ok(bincode::deserialize(&bytes)?)
    }
}
