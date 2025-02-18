//! Collection of process backtraces.
//!
//! This uses the grug-brain developer approach: just invoke the LLDB or GDB to get the backtrace.

use std::path::Path;
use tokio::{fs, process::Command};
use which::which;

pub async fn collect_process_backtrace(filename: &Path, pid: u32) -> anyhow::Result<()> {
    // Determine which debugger tool to use.
    let command_str = if which("lldb").is_ok() {
        lldb(pid)
    } else if which("gdb").is_ok() {
        gdb(pid)
    } else {
        anyhow::bail!("no lldb or gdb in PATH")
    };

    // Run the command using a shell
    let output = Command::new("sh")
        .arg("-c")
        .arg(&command_str)
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("{} failed: {}", command_str, stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Write the backtrace into the file specified by filename.
    fs::write(&filename, stdout.as_ref()).await?;

    Ok(())
}

/// Generate the lldb command for obtaining the backtrace.
fn lldb(pid: u32) -> String {
    format!(
        "lldb -p {} -o \"thread backtrace all\" -o \"detach\" -o \"quit\"",
        pid
    )
}

/// Generate the gdb command for obtaining the backtrace.
fn gdb(pid: u32) -> String {
    format!(
        "gdb -p {} -batch -ex \"thread apply all bt\" -ex \"detach\" -ex \"quit\"",
        pid
    )
}
