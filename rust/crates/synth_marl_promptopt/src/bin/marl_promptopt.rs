use std::path::PathBuf;

use clap::Parser;
use synth_marl_promptopt::execute_marl_promptopt_from_toml;

#[derive(Debug, Parser)]
#[command(name = "marl-promptopt")]
#[command(about = "Run a MARL-inspired prompt optimizer with public GEPA's proposer")]
struct Args {
    #[arg(long)]
    config: PathBuf,
}

fn main() {
    let args = Args::parse();
    match execute_marl_promptopt_from_toml(&args.config) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("serialize MARL promptopt result")
            );
        }
        Err(error) => {
            eprintln!("MARL prompt optimizer failed: {error}");
            std::process::exit(1);
        }
    }
}
