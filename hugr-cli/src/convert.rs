//! The `convert` subcommand.
use clap_stdin::FileOrStdin;
use hugr_model::v2::file::File;

/// Convert between HUGR file formats.
#[derive(clap::Parser, Debug)]
pub struct CliArgs {
    /// The input file to parse.
    pub input: FileOrStdin,
    /// The format to convert from.
    #[arg(long)]
    pub from: Format,
    /// The format to convert into.
    #[arg(long)]
    pub to: Format,
    /// Pretty print the output.
    #[arg(long, short)]
    pub pretty: bool,
}

impl CliArgs {
    /// Run the convert command.
    pub fn run(&self) -> Result<(), CliError> {
        let input: String = self.input.clone().contents()?;

        let hugr_file: File = match self.from {
            Format::Sexp => parens::from_str(&input).map_err(|err| CliError::SexpParse(err, input)),
            Format::Json => serde_json::from_str(&input).map_err(CliError::JsonParse),
        }?;

        let output = match (self.to, self.pretty) {
            (Format::Sexp, true) => parens::to_string_pretty(&hugr_file, 80),
            (Format::Sexp, false) => parens::to_string(&hugr_file),
            // converting to a JSON string should not fail
            (Format::Json, true) => serde_json::to_string_pretty(&hugr_file).unwrap(),
            (Format::Json, false) => serde_json::to_string(&hugr_file).unwrap(),
        };

        println!("{}", output);
        Ok(())
    }
}

/// File formats supported by hugr.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// S-expressions
    Sexp,
    /// JSON
    Json,
}

/// Error type for the CLI.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// Error reading input.
    #[error("Error reading input: {0}")]
    Input(#[from] clap_stdin::StdinError),
    /// Sexp parse error.
    #[error("Error while parsing input s-expression: {0}")]
    SexpParse(#[source] parens::parser::ParseError, String),
    /// JSON parse error.
    #[error("Error while parsing input json: {0}")]
    JsonParse(#[from] serde_json::Error),
}

impl CliError {
    /// Print the error.
    pub fn print(&self) {
        use ariadne::{Label, Source};
        match self {
            CliError::Input(err) => eprintln!("{}", err),
            CliError::SexpParse(err, input) => {
                let _ = ariadne::Report::build(ariadne::ReportKind::Error, (), err.span().start)
                    .with_message("parse error")
                    .with_label(
                        Label::new(err.span())
                            .with_message(err.to_string())
                            .with_color(ariadne::Color::Red),
                    )
                    .finish()
                    .eprint(Source::from(input));
            }
            CliError::JsonParse(err) => eprintln!("{}", err),
        }
    }
}
