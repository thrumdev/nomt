use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod meta_map;
mod store;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Init {
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
        #[arg(short, long)]
        num_pages: usize,
    },
    Run {
        #[arg(short, long, value_name = "FILE")]
        file: PathBuf,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Init { file, num_pages } => {
            if let Err(e) = store::create(file.clone(), num_pages) {
                println!("encountered error in creation {e:?}");
                let _ = std::fs::remove_file(file);
            }
        }
        Command::Run { file } => {
            let (_store, _meta_bytes) = match store::Store::open(file) {
                Ok(x) => x,
                Err(e) => {
                    println!("encountered error in opening store: {e:?}");
                    return;
                }
            };
        }
    }
}
