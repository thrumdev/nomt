use crate::backend::Backend;
use clap::{Args, Parser, Subcommand};
use std::fmt::Display;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Benchmark different workloads against different backends
    ///
    /// It is a combination of Init and Exec with the ability to specify the stopping
    /// parameters of the execution of workloads over multiple backends.
    #[command(subcommand)]
    Bench(bench::BenchType),
    /// Initialize NOMT backend for the specified workload.
    ///
    /// The backend will be initialized with all the data required
    /// to execute the workload.
    Init(WorkloadParams),
    /// Execute a workload once over NOMT.
    ///
    /// If the NOMT's database is not there, it will start with an empty database;
    /// otherwise, it will use the already present one.
    Run(WorkloadParams),

    /// Check regression over multiple workloads
    ///
    /// Load a TOML file containing multiple workloads specifications
    /// and their mean execution time, re-execute all the workloads,
    /// and compare the results.
    ///
    /// Example of entry in the toml file:
    ///
    /// [workloads.<name_of_workload>] {n}
    /// name = "randr" {n}
    /// size = 25000 {n}
    /// initial_capacity = 20 {n}
    /// # then you can specify isolate {n}
    /// [workloads.random_read_20.isolate] {n}
    /// iterations = 10 {n}
    /// mean = 909901824 # mean in ns to be compared with {n}
    /// # or sequential (or both){n}
    /// [workloads.random_read_20.sequential] {n}
    /// time_limit = 100000 # in nanoseconds {n}
    /// op_limit = 100 {n}
    /// mean = 909901824 {n}
    Regression(regression::Params),
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Backend::SovDB => "sov-db",
            Backend::Nomt => "nomt",
            Backend::SpTrie => "sp-trie",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Debug, Args)]
pub struct WorkloadParams {
    /// Workload used by benchmarks.
    ///
    /// Possible values are: transfer, randr, randw, seqr and seqw
    ///
    /// `transfer` workload involves balancing transfer between two different accounts.
    ///
    /// `randr` and `randw` will perform randomly uniformly distributed reads and writes,
    /// respectively, over the key space.
    ///
    /// `seqr` and `seqw` will perform sequential reads and writes, respectively,
    /// starting from a random key.
    #[clap(default_value = "transfer")]
    #[arg(long = "workload-name", short)]
    pub name: String,

    /// Parameters avaiable only with workload "transfer".
    ///
    /// It is the percentage of transfers to a non-existing account,
    /// the remaining portion of transfers are to existing accounts
    ///
    /// Accepted values are in the range of 0 to 100
    #[clap(value_parser=clap::value_parser!(u8).range(0..=100))]
    #[arg(long = "workload-percentage-cold", short)]
    pub percentage_cold: Option<u8>,

    /// Amount of operations performed in the workload
    #[clap(default_value = "1000")]
    #[arg(long = "workload-size", short)]
    pub size: u64,

    /// Additional size of the database before starting the benchmarks.
    ///
    /// Some workloads operate over existing keys in the database,
    /// and this size is additional to those entries.
    ///
    /// The provided argument is the power of two exponent of the
    /// number of elements already present in the storage.
    ///
    /// Accepted values are in the range of 0 to 63
    ///
    /// Leave it empty to specify an initial empty storage
    #[arg(long = "workload-capacity", short = 'c')]
    #[clap(value_parser=clap::value_parser!(u8).range(0..64))]
    pub initial_capacity: Option<u8>,
}

pub mod bench {
    use super::{Args, Backend, WorkloadParams};

    #[derive(clap::Subcommand, Debug)]
    pub enum BenchType {
        /// Each Workload execution will be performed on a copy of the initialized backend
        Isolate(IsolateParams),

        /// All workloads will be performed on the same backend after being initialized
        Sequential(SequentialParams),
    }

    #[derive(Debug, Args)]
    pub struct CommonParams {
        /// Possible Backends to run benchmarks against
        ///
        /// Leave this flag empty to run benchmarks against all avaiable backends
        ///
        /// Use ',' to separate backends
        #[clap(default_values_t = Vec::<Backend>::new(), value_delimiter = ',')]
        #[arg(long, short)]
        pub backends: Vec<Backend>,

        // TODO: Change this argument to a vector to allow for specifying multiple workloads
        #[clap(flatten)]
        pub workload: WorkloadParams,
    }

    #[derive(Debug, Args)]
    pub struct SequentialParams {
        #[clap(flatten)]
        pub common_params: CommonParams,

        // TODO: better descriptions
        /// Repeat the Workload on the same backends until the total amount of
        /// operations performed by all workloads reach the specified amount.
        #[arg(long)]
        pub op_limit: Option<u64>,

        /// Repeat the Workload on the same backends until the specified duration is exeeded [ms]
        #[arg(long)]
        pub time_limit: Option<u64>,
    }

    #[derive(Debug, Args)]
    pub struct IsolateParams {
        #[clap(flatten)]
        pub common_params: CommonParams,

        /// Number of time the benchmark will be repeated on the same backend
        #[clap(default_value = "10")]
        #[arg(long, short)]
        pub iterations: u64,
    }
}

pub mod regression {
    use super::Args;

    #[derive(Debug, Args)]
    pub struct Params {
        /// Path to the toml file containing workloads info
        #[arg(long, short)]
        pub input_file: Option<String>,

        /// Optional path file where results will be stored
        #[arg(long, short)]
        pub output_file: Option<String>,
    }
}
