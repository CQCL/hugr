//! Top level structure
use super::hugr::Hugr;
use super::keywords as kw;
use parens::{
    parser::{self, Parse, ParseError, Parser},
    printer::{Print, Printer},
};
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

/// A hugr file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    /// The format version of the file.
    pub version: Version,
    /// The items of the module.
    pub items: Vec<FileItem>,
}

impl Parse for File {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        let version: Version = parser.parse()?;

        if version.0 != "2" {
            return Err(ParseError::new(
                "only supports version 2",
                parser.parent_span(),
            ));
        }

        let mut items = Vec::new();
        while !parser.is_empty() {
            items.push(parser.list(FileItem::parse)?);
        }

        Ok(Self { version, items })
    }
}

impl Print for File {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.print(&self.version)?;

        for field in &self.items {
            printer.list(|printer| printer.print(field))?;
        }

        Ok(())
    }
}

/// A field in a hugr module file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileItem {
    /// A hugr graph.
    Hugr(Hugr),
}

impl Parse for FileItem {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        if parser.peek::<kw::hugr>() {
            Ok(Self::Hugr(parser.parse()?))
        } else {
            Err(parser.error("expected hugr"))
        }
    }
}

impl Print for FileItem {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        match self {
            FileItem::Hugr(hugr) => printer.print(hugr),
        }
    }
}

/// File format version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version(pub SmolStr);

impl Parse for Version {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.list(|parser| {
            parser.parse::<kw::version>()?;
            let version = parser.parse()?;
            Ok(Self(version))
        })
    }
}

impl Print for Version {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.list(|printer| {
            printer.print(kw::version)?;
            printer.print(&self.0)
        })
    }
}
