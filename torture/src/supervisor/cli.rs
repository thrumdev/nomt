use clap::{Args, Parser};

#[derive(Parser, Debug)]
pub struct Cli {
    /// The 8-byte seed to use for the random number generator.
    ///
    /// If not provided, a random seed will be generated.
    pub seed: Option<u64>,

    /// The maximum number of failures before the supervisor stops.
    ///
    /// If not provided, the supervisor will stop after the first failure.
    #[arg(short, long, default_value_t = 1)]
    pub flag_limit: usize,

    #[clap(flatten)]
    pub workload_params: WorkloadParams,
}

#[derive(Clone, Debug, Args)]
pub struct WorkloadParams {
    /// The probability of a delete operation as opposed to an insert operation.
    ///
    /// Accepted values are in the range of 0 to 100
    #[clap(default_value = "1")]
    #[clap(value_parser=clap::value_parser!(u8).range(0..=100))]
    #[arg(long = "workload-delete-bias", short = 'd')]
    pub delete: u8,

    /// When generating a value, the probability of generating a value that will spill into the
    /// overflow pages.
    ///
    /// Accepted values are in the range of 0 to 100
    #[clap(default_value = "1")]
    #[clap(value_parser=clap::value_parser!(u8).range(0..=100))]
    #[arg(long = "workload-overflow-bias", short = 'o')]
    pub overflow: u8,

    /// When generating a key, whether it should be one that was appeared somewhere
    /// or a brand new key.
    ///
    /// Accepted values are in the range of 0 to 100
    #[clap(default_value = "50")]
    #[clap(value_parser=clap::value_parser!(u8).range(0..=100))]
    #[arg(long = "workload-new-key-bias", short = 'n')]
    pub new_key: u8,

    /// The number of times a workload will be executed.
    #[clap(default_value = "50")]
    #[arg(long = "workload-iterations", short = 'i')]
    pub iterations: usize,

    /// The size of a single workload iteration, the number of changesets per commit.
    #[clap(default_value = "5000")]
    #[arg(long = "workload-size", short = 's')]
    pub size: usize,

    /// Whether the size of each workload should be random or not.
    ///
    /// If specified, the size of each commit will be within 0..workload-size,
    /// otherwise it will always be workload-size.
    #[clap(default_value = "false")]
    #[arg(long = "workload-random-size")]
    pub random_size: bool,

    /// When exercising a new commit, the probability of causing it to crash.
    ///
    /// Accepted values are in the range of 0 to 100
    #[clap(default_value = "20")]
    #[clap(value_parser=clap::value_parser!(u8).range(0..=100))]
    #[arg(long = "workload-commit-crash")]
    pub crash: u8,
}
