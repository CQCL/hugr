//! Validate serialized HUGR on the command line

use clap::Parser as _;
use hugr_core::std_extensions::arithmetic::{
    conversions::EXTENSION as CONVERSIONS_EXTENSION, float_ops::EXTENSION as FLOAT_OPS_EXTENSION,
    float_types::EXTENSION as FLOAT_TYPES_EXTENSION, int_ops::EXTENSION as INT_OPS_EXTENSION,
    int_types::EXTENSION as INT_TYPES_EXTENSION,
};
use hugr_core::std_extensions::logic::EXTENSION as LOGICS_EXTENSION;

use hugr_cli::{extensions::ExtArgs, validate, CliArgs};
use hugr_core::extension::{ExtensionRegistry, PRELUDE};

use clap_verbosity_flag::Level;

fn main() {
    match CliArgs::parse() {
        CliArgs::Validate(args) => run_validate(args),
        CliArgs::GenExtension(args) => run_dump(args),
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
    let reg = std_reg();

    let result = args.run(&reg);

    if let Err(e) = result {
        if args.verbosity(Level::Error) {
            eprintln!("{}", e);
        }
        std::process::exit(1);
    }
}

fn std_reg() -> ExtensionRegistry {
    ExtensionRegistry::try_new([
        PRELUDE.to_owned(),
        INT_OPS_EXTENSION.to_owned(),
        INT_TYPES_EXTENSION.to_owned(),
        CONVERSIONS_EXTENSION.to_owned(),
        FLOAT_OPS_EXTENSION.to_owned(),
        FLOAT_TYPES_EXTENSION.to_owned(),
        LOGICS_EXTENSION.to_owned(),
    ])
    .unwrap()
}

/// Write out the standard extensions in serialized form.
fn run_dump(args: ExtArgs) {
    let base_dir = args.outdir;

    for (name, ext) in std_reg().into_iter() {
        let mut path = base_dir.clone();
        for part in name.split('.') {
            path.push(part);
        }
        path.set_extension("json");

        std::fs::create_dir_all(path.clone().parent().unwrap()).unwrap();
        // file buffer
        let file = std::fs::File::create(&path).unwrap();

        serde_json::to_writer_pretty(file, &ext).unwrap();
    }
}
