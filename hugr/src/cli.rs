//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_stdin::FileOrStdin;

use crate::{extension::ExtensionRegistry, Hugr, HugrView};
/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
struct CmdLineArgs {
    input: FileOrStdin,
    /// Visualise with mermaid.
    #[arg(short, long, value_name = "MERMAID", help = "Visualise with mermaid.")]
    mermaid: bool,

    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    no_validate: bool,
    // TODO YAML extensions
}

/// Run the HUGR cli and validate against an extension registry.
pub fn run(registry: &ExtensionRegistry) -> Result<(), Box<dyn std::error::Error>> {
    let opts = CmdLineArgs::parse();

    let mut hugr: Hugr = serde_json::from_reader(opts.input.into_reader()?)?;
    if opts.mermaid {
        println!("{}", hugr.mermaid_string());
    }

    if !opts.no_validate {
        hugr.update_validate(registry)?;

        println!("HUGR valid!");
    }
    Ok(())
}
