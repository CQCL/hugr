//! Standard command line tools, used by the hugr binary.

use clap::Parser;
use clap_stdin::FileOrStdin;

use crate::{extension::ExtensionRegistry, Hugr, HugrView};
/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
pub struct CmdLineArgs {
    input: FileOrStdin,
    /// Visualise with mermaid.
    #[arg(short, long, value_name = "MERMAID", help = "Visualise with mermaid.")]
    mermaid: bool,
    /// Skip validation.
    #[arg(short, long, help = "Skip validation.")]
    no_validate: bool,
    // TODO YAML extensions
}

/// String to print when validation is successful.
pub const VALID_PRINT: &str = "HUGR valid!";

impl CmdLineArgs {
    /// Run the HUGR cli and validate against an extension registry.
    pub fn run(&self, registry: &ExtensionRegistry) -> Result<(), Box<dyn std::error::Error>> {
        let mut hugr: Hugr = serde_json::from_reader(self.input.into_reader()?)?;
        if self.mermaid {
            println!("{}", hugr.mermaid_string());
        }

        if !self.no_validate {
            hugr.update_validate(registry)?;

            println!("{}", VALID_PRINT);
        }
        Ok(())
    }
}
