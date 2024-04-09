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
    Bench(bench::Params),
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Backend {
    SovDB,
    Nomt,
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Backend::SovDB => "sov-db",
            Backend::Nomt => "nomt",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Workload {
    /// Performs 1 write into the storage
    SetBalance,
    /// Performs 2 reads and 2 writes into the storage
    Transfer,
}

impl Display for Workload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Workload::Transfer => "transer",
            Workload::SetBalance => "set-balance",
        };
        f.write_str(name)
    }
}

pub mod bench {
    use super::{Args, Backend, Workload};

    #[derive(Debug, Args)]
    pub struct Params {
        /// Possible Backends to run benchmarks against
        ///
        /// Leave this flag empty to run benchmarks against all avaiable backends
        ///
        /// Use ',' to separate backends
        #[clap(default_values_t = Vec::<Backend>::new(), value_delimiter = ',')]
        #[arg(long, short)]
        pub backend: Vec<Backend>,

        /// Workloads used by benchmarks.
        #[clap(default_value = "transfer")]
        #[arg(long, short)]
        pub workload: Workload,

        /// Amount of actions performed in the workload
        #[clap(default_value = "1000")]
        #[arg(long, short = 's')]
        pub workload_size: u64,

        /// Size of the db before starting the benchmarks.
        ///
        /// The provided argument is the power of two exponent of the
        /// number of elements already present in the storage.
        ///
        /// Leave it empty to specify an initial empty storage
        #[clap(default_value = "0")]
        #[arg(long, short)]
        pub initial_size: Option<u8>,

        /// Number of time the benchmark will be repeated on the same backend
        #[clap(default_value = "100")]
        #[arg(long)]
        pub iteration: u64,
    }
}
