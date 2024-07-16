//! Validate serialized HUGR on the command line

use clap::Parser as _;
use hugr_core::std_extensions::arithmetic::{
    conversions::EXTENSION as CONVERSIONS_EXTENSION, float_ops::EXTENSION as FLOAT_OPS_EXTENSION,
    float_types::EXTENSION as FLOAT_TYPES_EXTENSION, int_ops::EXTENSION as INT_OPS_EXTENSION,
    int_types::EXTENSION as INT_TYPES_EXTENSION,
};
use hugr_core::std_extensions::logic::EXTENSION as LOGICS_EXTENSION;

use hugr_cli::{validate, CliArgs};
use hugr_core::extension::{ExtensionRegistry, PRELUDE};

use clap_verbosity_flag::Level;

fn main() {
    match CliArgs::parse() {
        CliArgs::Validate(args) => run_validate(args),
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
    // validate with all std extensions
    let reg = ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        INT_OPS_EXTENSION.to_owned(),
        INT_TYPES_EXTENSION.to_owned(),
        CONVERSIONS_EXTENSION.to_owned(),
        FLOAT_OPS_EXTENSION.to_owned(),
        FLOAT_TYPES_EXTENSION.to_owned(),
        LOGICS_EXTENSION.to_owned(),
    ])
    .unwrap();

    let result = args.run(&reg);

    if let Err(e) = result {
        if args.verbosity(Level::Error) {
            eprintln!("{}", e);
        }
        std::process::exit(1);
    }
}
