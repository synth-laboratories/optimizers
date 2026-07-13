use std::path::PathBuf;

use clap::Parser;
use synth_gepa::execute_gepa_from_toml;

#[derive(Debug, Parser)]
#[command(name = "gepa-baseline")]
#[command(about = "Run the public Rust GEPA baseline for a MARL profile")]
struct Args {
    #[arg(long)]
    config: PathBuf,
}

fn main() {
    let args = Args::parse();
    match execute_gepa_from_toml(&args.config) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("serialize GEPA baseline result")
            );
        }
        Err(error) => {
            eprintln!("GEPA baseline failed: {error}");
            std::process::exit(1);
        }
    }
}
