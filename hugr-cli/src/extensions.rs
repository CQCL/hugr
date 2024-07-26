//! Dump standard extensions in serialized form.
use clap::Parser;
use std::path::PathBuf;

/// Dump the standard extensions.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Write standard extensions.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct ExtArgs {
    /// Output directory
    #[arg(
        default_value = ".",
        short,
        long,
        value_name = "OUTPUT",
        help = "Output directory."
    )]
    pub outdir: PathBuf,
}
