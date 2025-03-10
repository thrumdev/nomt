use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Execute swarm testing. Multiple workloads will be executed at the same
    /// time, enabling and disabling different nomt features.
    Swarm(SwarmParams),
    /// Execute a single workload given a seed.
    Run(RunParams),
}

#[derive(Clone, Debug, Args)]
pub struct SwarmParams {
    /// The maximum number of failures before the supervisor stops.
    ///
    /// If not provided, the supervisor will stop after the first failure.
    #[arg(short, long, default_value_t = 1)]
    pub flag_limit: usize,

    /// Folder that will be used as the working directory by the Supervisor.
    /// It will contain all workload folders.
    #[arg(long = "workdir")]
    pub workdir: Option<String>,
}

#[derive(Clone, Debug, Args)]
pub struct RunParams {
    /// The 8-byte seed to use for the random number generator.
    pub seed: u64,

    /// Amount of disk space in bytes assigned to the workload. [Default: 20GiB]
    #[arg(short = 'd', long, default_value_t = 20 * 1024 * 1024 * 1024)]
    pub assigned_disk: u64,

    /// Amount of memory in bytes assigned to the workload. [Default: 3GiB]
    #[arg(short = 'm' ,long, default_value_t = 3 * 1024 * 1024 * 1024)]
    pub assigned_memory: u64,

    /// Folder that will be used as the working directory by the Supervisor.
    /// It will contain the folder of the workload that it is being executed.
    #[arg(long = "workdir")]
    pub workdir: Option<String>,
}
