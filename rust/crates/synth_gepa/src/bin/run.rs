//! Native GEPA entry point for evaluator-owned, frozen TOML configurations.
use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() != 1 {
        eprintln!("Usage: synth-gepa-run CONFIG.toml");
        return ExitCode::from(2);
    }
    match synth_gepa::execute_gepa_from_toml(std::path::Path::new(&arguments[0])) {
        Ok(result) => match serde_json::to_string(&result) {
            Ok(value) => {
                println!("{value}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Native GEPA result serialization failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Native GEPA execution failed: {error}");
            ExitCode::FAILURE
        }
    }
}
