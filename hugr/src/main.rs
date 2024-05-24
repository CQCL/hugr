//! Validate serialized HUGR on the command line

use hugr::std_extensions::arithmetic::{
    conversions::EXTENSION as CONVERSIONS_EXTENSION, float_ops::EXTENSION as FLOAT_OPS_EXTENSION,
    float_types::EXTENSION as FLOAT_TYPES_EXTENSION, int_ops::EXTENSION as INT_OPS_EXTENSION,
    int_types::EXTENSION as INT_TYPES_EXTENSION,
};
use hugr::std_extensions::logic::EXTENSION as LOGICS_EXTENSION;

use hugr::extension::{ExtensionRegistry, PRELUDE};

use clap::Parser;
use hugr_core::cli::CmdLineArgs;

fn main() {
    let opts = CmdLineArgs::parse();

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

    if let Err(e) = opts.run(&reg) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
