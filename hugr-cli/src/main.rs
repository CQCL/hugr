//! Validate serialized HUGR on the command line

use std::ffi::OsString;

use anyhow::{Result, anyhow};
use clap::Parser as _;
use clap_verbosity_flag::VerbosityFilter;
use hugr_cli::{CliArgs, CliCommand};
use tracing::{error, metadata::LevelFilter};

fn main() {
    let cli_args = CliArgs::parse();

    let level = match cli_args.verbose.filter() {
        VerbosityFilter::Off => LevelFilter::OFF,
        VerbosityFilter::Error => LevelFilter::ERROR,
        VerbosityFilter::Warn => LevelFilter::WARN,
        VerbosityFilter::Info => LevelFilter::INFO,
        VerbosityFilter::Debug => LevelFilter::DEBUG,
        VerbosityFilter::Trace => LevelFilter::TRACE,
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(level)
        .pretty()
        .init();

    let result = match cli_args.command {
        CliCommand::Validate(mut args) => args.run(),
        CliCommand::GenExtensions(args) => args.run_dump(&hugr::std_extensions::STD_REG),
        CliCommand::Mermaid(mut args) => args.run_print(),
        CliCommand::Convert(mut args) => args.run_convert(),
        CliCommand::Describe(mut args) => args.run_describe(),
        CliCommand::External(args) => run_external(args),
        _ => Err(anyhow!("Unknown command")),
    };

    if let Err(err) = result {
        error!("{:?}", err);
        // TODO include description if verbosity is high enough
        std::process::exit(1);
    }
}

fn run_external(args: Vec<OsString>) -> Result<()> {
    // External subcommand support: invoke `hugr-<subcommand>`
    if args.is_empty() {
        eprintln!("No external subcommand specified.");
        std::process::exit(1);
    }
    let subcmd = args[0].to_string_lossy();
    let exe = format!("hugr-{subcmd}");
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
            eprintln!("error: no such subcommand: '{subcmd}'.\nCould not find '{exe}' in PATH.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("error: failed to invoke '{exe}': {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}
