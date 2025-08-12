use indenter::indented;
use itertools::{Either, Itertools};
use pretty::{Arena, DocAllocator, RefDoc};

use crate::v0::{LinkName, RegionKind, SymbolName, Visibility, ast::Operation};

use super::{Node, Region, SeqPart, Symbol, Term};
use std::fmt::{self, Display, Write as _};

impl Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Wildcard => write!(f, "_"),
            Term::Var(var) => write!(f, "{}", var),
            Term::Apply(symbol, args) => {
                write!(f, "{}{}", symbol, OptSeq("[", args, "]"))
            }
            Term::List(seq_parts) => {
                write!(f, "[{}]", seq_parts.iter().format(", "))
            }
            Term::Literal(literal) => write!(f, "{}", literal),
            Term::Tuple(seq_parts) => {
                write!(f, "({})", seq_parts.iter().format(", "))
            }
            Term::Func(region) => todo!(),
        }
    }
}

impl Display for SeqPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeqPart::Item(term) => write!(f, "{}", term),
            SeqPart::Splice(term) => write!(f, "... {}", term),
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Write the types of the input and output links as well
        write!(
            f,
            "{meta}{outputs}",
            meta = Meta(&self.meta),
            outputs = OptSeq("", &self.outputs, " = ")
        )?;

        match &self.operation {
            Operation::Invalid => write!(f, "invalid")?,
            Operation::Dfg => write!(f, "dfg")?,
            Operation::Cfg => write!(f, "cfg")?,
            Operation::Block => write!(f, "block")?,
            Operation::DefineFunc(symbol) | Operation::DeclareFunc(symbol) => {
                write_symbol(f, "fn", symbol)?
            }
            Operation::Custom(term) => write!(f, "{}", term)?,
            Operation::DefineAlias(symbol, term) => todo!(),
            Operation::DeclareAlias(symbol) => todo!(),
            Operation::TailLoop => write!(f, "loop")?,
            Operation::Conditional => write!(f, "cond")?,
            Operation::DeclareConstructor(symbol) => write_symbol(f, "ctr", symbol)?,
            Operation::DeclareOperation(symbol) => write_symbol(f, "op", symbol)?,
            Operation::Import(symbol_name) => write!(f, "use {}", symbol_name)?,
        };

        write!(f, "{}", OptSeq(" ", &self.inputs, ""))?;

        match self.regions.as_ref() {
            [] => {}
            [region] if region.meta.is_empty() => write!(f, " {}", region)?,
            regions => {
                writeln!(f, " {{")?;
                for region in regions {
                    writeln!(&mut indented(f).ind(2), "{}", region)?;
                }
                write!(f, "}}")?;
            }
        }

        write!(f, ";")
    }
}

impl Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Print types for sources and targets!

        match self.kind {
            RegionKind::Module => {
                writeln!(f, "mod {{")?;
                for node in self.children.iter() {
                    writeln!(&mut indented(f).ind(2), "{}", node)?;
                }
                writeln!(f, "}}")
            }
            RegionKind::DataFlow => {
                writeln!(
                    f,
                    "({sources}) -> ({targets}) {{",
                    sources = self.sources.iter().format(", "),
                    targets = self.targets.iter().format(", ")
                )?;
                for node in self.children.iter() {
                    writeln!(&mut indented(f).ind(2), "{}", node)?;
                }
                writeln!(f, "}}")
            }
            RegionKind::ControlFlow => todo!(),
        }
    }
}

fn write_symbol(f: &mut fmt::Formatter<'_>, sort: &'static str, symbol: &Symbol) -> fmt::Result {
    match symbol.visibility {
        Some(Visibility::Public) => write!(f, "pub")?,
        Some(Visibility::Private) => {}
        None => {}
    }

    write!(
        f,
        "{sort} {name}{params} : {signature}{constraints}",
        name = symbol.name,
        params = OptSeq("[", &symbol.params, "]"),
        signature = symbol.signature,
        constraints = OptSeq("\nwhere ", &symbol.constraints, "")
    )
}

fn list_items(term: &Term) -> Option<Vec<&Term>> {
    let Term::List(parts) = term else {
        return None;
    };

    let mut items = Vec::new();

    for part in parts.iter() {
        match part {
            SeqPart::Item(item) => items.push(item),
            SeqPart::Splice(list) => items.extend(list_items(list)?),
        }
    }

    Some(items)
}

fn write_typed_links(f: &mut fmt::Formatter<'_>, links: &[LinkName], types: &Term) -> fmt::Result {
    let mut types = list_items(&types).unwrap_or_default().into_iter();
    let typed_links = links.iter().map(|link| TypedLink(link, types.next()));
    write!(f, "{}", typed_links.format(", "))
}

struct TypedLink<'a>(&'a LinkName, Option<&'a Term>);

impl<'a> Display for TypedLink<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;

        if let Some(typ) = &self.1 {
            write!(f, ": {}", typ)?;
        }

        Ok(())
    }
}

struct Meta<'a>(&'a [Term]);

impl<'a> Display for Meta<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for term in self.0.iter() {
            writeln!(f, "#[{}]", term)?;
        }
        Ok(())
    }
}

struct OptSeq<'a, T>(&'a str, &'a [T], &'a str);

impl<'a, T: Display> Display for OptSeq<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.1.is_empty() {
            write!(f, "{}{}{}", self.0, self.1.iter().format(", "), self.2)?;
        }
        Ok(())
    }
}
