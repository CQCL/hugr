//! Validate serialized HUGR on the command line

use clap::Parser as _;

use hugr_cli::{CliArgs, mermaid, validate};

use clap_verbosity_flag::log::Level;

fn main() {
    match CliArgs::parse() {
        CliArgs::Validate(args) => run_validate(args),
        CliArgs::GenExtensions(args) => args.run_dump(&hugr::std_extensions::STD_REG),
        CliArgs::Mermaid(args) => run_mermaid(args),
        CliArgs::External(args) => {
            // External subcommand support: invoke `hugr-<subcommand>`
            if args.is_empty() {
                eprintln!("No external subcommand specified.");
                std::process::exit(1);
            }
            let subcmd = args[0].to_string_lossy();
            let exe = format!("hugr-{}", subcmd);
            let rest: Vec<_> = args[1..]
                .iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect();
            match std::process::Command::new(&exe).args(&rest).status() {
                Ok(status) => {
                    if !status.success() {
                        std::process::exit(status.code().unwrap_or(1));
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "error: no such subcommand: '{subcmd}'.\nCould not find '{exe}' in PATH."
                    );
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("error: failed to invoke '{exe}': {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    }
}

/// Run the `validate` subcommand.
fn run_validate(mut args: validate::ValArgs) {
    let result = args.run();

    if let Err(e) = result {
        if args.verbosity(Level::Error) {
            eprintln!("{e}");
        }
        std::process::exit(1);
    }
}

/// Run the `mermaid` subcommand.
fn run_mermaid(mut args: mermaid::MermaidArgs) {
    let result = args.run_print();

    if let Err(e) = result {
        if args.other_args.verbosity(Level::Error) {
            eprintln!("{e}");
        }
        std::process::exit(1);
    }
}
