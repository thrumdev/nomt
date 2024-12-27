#![allow(dead_code)]

use anyhow::Result;
use tokio::net::UnixStream;

mod agent;
mod logging;
mod message;
mod spawn;
mod supervisor;

#[tokio::main]
async fn main() -> Result<()> {
    if let Some(chan) = spawn::am_spawned() {
        println!("orture agent");
        let chan = UnixStream::from_std(chan)?;
        agent::run(chan).await?;
    } else {
        println!("orture supervisor");
        supervisor::run().await?;
    }
    Ok(())
}
