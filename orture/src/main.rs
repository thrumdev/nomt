#![allow(dead_code)]

use anyhow::Result;
use futures::SinkExt;
use tokio::{io::BufWriter, net::UnixStream};
use tokio_util::codec::FramedWrite;

mod agent;
mod message;
mod spawn;

#[tokio::main]
async fn main() -> Result<()> {
    if let Some(chan) = spawn::am_spawned() {
        println!("orture agent");
        let chan = UnixStream::from_std(chan)?;
        agent::run(chan).await?;
    } else {
        println!("orture supervisor");
        let (child, chan) = spawn::spawn_child()?;
        let chan = UnixStream::from_std(chan)?;
        let mut framed = FramedWrite::new(
            BufWriter::new(chan),
            tokio_util::codec::LengthDelimitedCodec::new(),
        );

        std::fs::create_dir_all("/tmp/agent-1").unwrap();

        // Send an init message.
        let init_message = message::ToAgent::Init(message::InitPayload {
            id: "1".to_string(),
            workdir: "/tmp/agent-1".to_string(),
            bitbox_seed: [0; 16],
        });
        let buf = bincode::serialize(&init_message)?;
        framed.send(buf.into()).await?;

        // Perform a commit.
        let commit_message = message::ToAgent::Commit(message::CommitPayload {
            changset: vec![
                message::KeyValueChange::Insert([1; 32], vec![1, 2, 3]),
                message::KeyValueChange::Delete([2; 32]),
            ],
        });
        let buf = bincode::serialize(&commit_message)?;
        framed.send(buf.into()).await?;

        std::thread::sleep(std::time::Duration::from_secs(10));

        // Perform shutdown.
        let shutdown_message = message::ToAgent::GracefulShutdown;
        let buf = bincode::serialize(&shutdown_message)?;
        framed.send(buf.into()).await?;

        child.wait();

        std::fs::remove_dir_all("/tmp/agent-1").unwrap();
    }
    Ok(())
}
