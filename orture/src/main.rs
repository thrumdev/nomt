#![allow(dead_code)]

use anyhow::Result;

mod spawn;

fn main() -> Result<()> {
    println!("Starting orture");

    if let Some(chan) = spawn::am_spawned() {
        println!("I am a child process");
        drop(chan);
    } else {
        let child = spawn::spawn_child()?;
        drop(child);

        // Perform shutdown.
        let shutdown_message = message::ToAgent::GracefulShutdown;
        let buf = bincode::serialize(&shutdown_message)?;
        framed.send(buf.into()).await?;

        child.wait();

        std::fs::remove_dir_all("/tmp/agent-1").unwrap();
    }

    Ok(())
}
