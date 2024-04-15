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
    /// It is a combination of Init and Exec with specifiable iterations
    /// over possible multiple backends
    Bench(bench::Params),
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
    /// Possible values are: transfer, set_balance/heavy_write, heavy_read, heavy_update, heavy_delete
    ///
    /// Transfer workload involves balancing transfer between two different accounts
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

    #[derive(Debug, Args)]
    pub struct Params {
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

        /// Number of time the benchmark will be repeated on the same backend
        #[clap(default_value = "100")]
        #[arg(long, short)]
        pub iteration: u64,
    }
}
