use clap::Parser;

#[derive(Parser, Debug)]
pub struct Cli {
    /// The 8-byte seed to use for the random number generator.
    ///
    /// If not provided, a random seed will be generated.
    pub seed: Option<u64>,
}
