//! Dump standard extensions in serialized form.
use anyhow::Result;
use clap::Parser;
use hugr::extension::ExtensionRegistry;
use std::{io::Write, path::PathBuf};

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

impl ExtArgs {
    /// Write out the standard extensions in serialized form.
    /// Qualified names of extensions used to generate directories under the specified output directory.
    /// E.g. extension "foo.bar.baz" will be written to "OUTPUT/foo/bar/baz.json".
    pub fn run_dump(&self, registry: &ExtensionRegistry) -> Result<()> {
        let base_dir = &self.outdir;

        for ext in registry {
            let mut path = base_dir.clone();
            for part in ext.name().split('.') {
                path.push(part);
            }
            path.set_extension("json");

            std::fs::create_dir_all(path.clone().parent().unwrap())?;
            // file buffer
            let mut file = std::fs::File::create(&path)?;

            serde_json::to_writer_pretty(&mut file, &ext)?;

            // write newline, for pre-commit end of file check that edits the file to
            // add newlines if missing.
            file.write_all(b"\n")?;
        }

        Ok(())
    }
}
