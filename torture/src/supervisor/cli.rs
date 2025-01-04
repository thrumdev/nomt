use clap::Parser;

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
}
