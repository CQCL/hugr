//! Validate serialized HUGR on the command line

use clap::Parser as _;

use hugr_cli::{validate, CliArgs};

use clap_verbosity_flag::Level;
use hugr_core::extension::ExtensionRegistry;

fn main() {
    match CliArgs::parse() {
        CliArgs::Validate(args) => run_validate(args),
        CliArgs::GenExtensions(args) => args.run_dump(),
        CliArgs::External(_) => {
            // TODO: Implement support for external commands.
            // Running `hugr COMMAND` would look for `hugr-COMMAND` in the path
            // and run it.
            eprintln!("External commands are not supported yet.");
            std::process::exit(1);
        }
        _ => {
            eprintln!("Unknown command");
            std::process::exit(1);
        }
    };
}

/// Run the `validate` subcommand.
fn run_validate(args: validate::CliArgs) {
    let std_reg: ExtensionRegistry;
    let reg: &ExtensionRegistry = if args.no_std {
        &hugr_core::extension::PRELUDE_REGISTRY
    } else {
        std_reg = hugr_core::std_extensions::std_reg();
        &std_reg
    };

    let result = args.run(reg);

    if let Err(e) = result {
        if args.verbosity(Level::Error) {
            eprintln!("{}", e);
        }
        std::process::exit(1);
    }
}
