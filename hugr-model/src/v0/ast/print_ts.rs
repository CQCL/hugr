use indenter::indented;
use itertools::{Either, Itertools};
use pretty::{Arena, DocAllocator, RefDoc};

use crate::v0::{RegionKind, SymbolName, Visibility, ast::Operation};

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
                write!(f, "[{}]", Seq(seq_parts))
            }
            Term::Literal(literal) => write!(f, "{}", literal),
            Term::Tuple(seq_parts) => {
                write!(f, "({})", Seq(seq_parts))
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
                    sources = Seq(&self.sources),
                    targets = Seq(&self.targets)
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
            write!(f, "{}{}{}", self.0, Seq(self.1), self.2)?;
        }
        Ok(())
    }
}

struct Seq<'a, T>(&'a [T]);

impl<'a, T: Display> Display for Seq<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for item in Itertools::intersperse(self.0.iter().map(Some), None) {
            match item {
                Some(item) => write!(f, "{}", item)?,
                None => write!(f, ", ")?,
            }
        }
        Ok(())
    }
}
