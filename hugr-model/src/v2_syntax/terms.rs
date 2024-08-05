//! Terms.

use parens::parser::{self, Parse, Parser, Peek};
use parens::printer::{Print, Printer};
use parens::util::Form;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use super::ident::{Label, Symbol, TermVar};
use super::keywords as kw;

/// A type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    /// Type constructor application.
    #[serde(rename = "ctr")]
    Ctr(TermCtr),
    /// Type variable.
    #[serde(rename = "var")]
    Var(TermVar),
    /// Row type.
    #[serde(rename = "row")]
    Row(TermRow),
    /// List type.
    #[serde(rename = "list")]
    List(TermList),
    /// Label.
    #[serde(rename = "label")]
    Label(Label),
    /// Scheme.
    #[serde(rename = "scheme")]
    Scheme(TermScheme),
    /// Literal integer.
    #[serde(rename = "int")]
    LitInt(i64),
    /// Literal string.
    #[serde(rename = "str")]
    LitStr(SmolStr),
}

impl Parse for Term {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        if parser.peek::<TermCtr>() {
            Ok(Self::Ctr(parser.parse()?))
        } else if parser.peek::<TermVar>() {
            Ok(Self::Var(parser.parse()?))
        } else if parser.peek::<TermRow>() {
            Ok(Self::Row(parser.parse()?))
        } else if parser.peek::<TermList>() {
            Ok(Self::List(parser.parse()?))
        } else if parser.peek::<Label>() {
            Ok(Self::Label(parser.parse()?))
        } else if parser.peek::<i64>() {
            Ok(Self::LitInt(parser.parse()?))
        } else if parser.peek::<Form<kw::str>>() {
            // Literal strings need to be contained in a `(str ...)` form at the moment.
            // This is due to `parens` not distinguishing between symbols and strings.
            // We might want to change this in the future.
            parser.list(|parser| {
                parser.parse::<kw::str>()?;
                Ok(Self::LitStr(parser.parse()?))
            })
        } else {
            Err(parser.error("expected type"))
        }
    }
}

impl Print for Term {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        match self {
            Term::Ctr(ctr) => printer.print(ctr),
            Term::Var(var) => printer.print(var),
            Term::Row(row) => printer.print(row),
            Term::List(list) => printer.print(list),
            Term::Label(label) => printer.print(label),
            Term::LitInt(int) => printer.print(int),
            Term::LitStr(str) => printer.list(|printer| {
                printer.print(kw::str)?;
                printer.print(str)
            }),
            Term::Scheme(_) => todo!(),
        }
    }
}

/// Type constructor applications.
///
/// # Syntax
///
/// The syntax of a type constructor is `(@name arg-0 ... arg-n)` where `@name` is a symbol
/// and `arg-0` to `arg-n` are the arguments to the type constructor. When the type constructor
/// does not have any arguments, it can also be written as `@name`. For example,
/// `(@hashmap @str @int)` may denote a hashmap type with string keys and integer values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermCtr {
    /// The name of the type constructor.
    pub ctr: Symbol,
    /// The arguments to the type constructor.
    pub args: Vec<Term>,
}

impl Parse for TermCtr {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        if parser.peek::<Symbol>() {
            let ctr = parser.parse()?;
            let args = Vec::new();
            Ok(Self { ctr, args })
        } else {
            parser.list(|parser| {
                let ctr = parser.parse()?;
                let args = parser.parse()?;
                Ok(Self { ctr, args })
            })
        }
    }
}

impl Peek for TermCtr {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek::<Symbol>() || cursor.peek_list(|cursor| cursor.peek::<Symbol>())
    }
}

impl Print for TermCtr {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        if self.args.is_empty() {
            printer.print(&self.ctr)
        } else {
            printer.list(|printer| {
                printer.print(&self.ctr)?;
                printer.print(&self.args)
            })
        }
    }
}

/// A row.
///
/// Rows have an optional tail.
/// If the tail is present the row is called *open*, and otherwise *closed*.
/// A row whose tail is again a row is considered equivalent to a single
/// concatenated row.
/// An open row is *proper* when its tail is either a proper row or a variable,
/// and *improper* otherwise. A closed row is always *proper*.
///
/// Two adjacent entries in a row may exchange place as long as their label is different.
/// Duplicate labels in a row are in general allowed, but their relative order is significant
/// (see [scoped labels]). Particular usages of rows may exclude rows with duplicate labels.
///
/// # Syntax
///
/// A closed row is a sequence of labels and types
/// delimited by braces: `{(:label-1 type-1) ... (:label-n type-n)}`.
/// Open rows are denoted as `{(:label-1 type-1) ... (:label-n type-n) . tail}` where `tail` is again a type.
///
/// # Examples
///
///  - Struct types with named fields: `(@struct {(:x @i64) (:y @i64)})`
///  - Variants in an enum: `(@enum {(:ok %i) (:err @parse-error)})`
///
/// [scoped labels]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/scopedlabels.pdf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermRow {
    /// The entries in the row type.
    pub entries: Vec<TermRowEntry>,
    /// The tail of the row or `None` if the row is closed.
    pub tail: Option<Box<Term>>,
}

impl Parse for TermRow {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.map(|parser| {
            let mut entries = Vec::new();
            let mut tail = None;

            while !parser.is_empty() {
                if parser.peek::<kw::dot>() {
                    parser.parse::<kw::dot>()?;
                    tail = Some(parser.parse()?);
                    break;
                } else {
                    entries.push(parser.parse()?);
                }
            }

            Ok(TermRow { entries, tail })
        })
    }
}

impl Peek for TermRow {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek_map(|_| true)
    }
}

impl Print for TermRow {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.print(&self.entries)?;

        if let Some(tail) = &self.tail {
            printer.print(kw::dot)?;
            printer.print(tail)?;
        }

        Ok(())
    }
}

/// An entry in a row type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermRowEntry {
    /// The label of the row entry.
    pub label: Label,
    /// The type associated to the row entry.
    pub value: Box<Term>,
}

impl Parse for TermRowEntry {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.list(|parser| {
            let label = parser.parse()?;
            let value = parser.parse()?;
            Ok(TermRowEntry { label, value })
        })
    }
}

impl Print for TermRowEntry {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.list(|printer| {
            printer.print(&self.label)?;
            printer.print(&self.value)
        })
    }
}

/// A list.
///
/// Lists have an optional tail.
/// If the tail is present the list is called *open*, and otherwise *closed*.
/// A list whose tail is again a list is considered equivalent to a single
/// concatenated list.
/// An open list is *proper* when its tail is either a proper list or a variable,
/// and *improper* otherwise. A closed list is always *proper*.
///
/// # Syntax
///
/// A closed list is a sequence of terms delimited by square brackets: `[item-1 ... item-n]`.
/// Open lists are denoted as `[item-1 ... item-n . tail]` where `tail` is again a term.
///
/// # Examples
///
///  - Inputs and outputs of a function: `(@fn [@f32 @f32] [@f32])`
///  - The type of an `eval` function, using open lists: `(@fn [(@fn %i %o) . %i] [%o])`
///  - The types in a tuple: `(@tuple [@int @str])`
///  - Dimensions of a tensor: `(@tensor @f32 [512 128 128])`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermList {
    /// The items in the list.
    pub items: Vec<Term>,
    /// The tail of the list, or `None` if the list is closed.
    pub tail: Option<Box<Term>>,
}

impl Parse for TermList {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.seq(|parser| {
            let mut items = Vec::new();
            let mut tail = None;

            while !parser.is_empty() {
                if parser.peek::<kw::dot>() {
                    parser.parse::<kw::dot>()?;
                    tail = Some(parser.parse()?);
                    break;
                } else {
                    items.push(parser.parse()?);
                }
            }

            Ok(TermList { items, tail })
        })
    }
}

impl Peek for TermList {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek_seq(|_| true)
    }
}

impl Print for TermList {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.seq(|printer| {
            printer.print(&self.items)?;

            if let Some(tail) = &self.tail {
                printer.print(kw::dot)?;
                printer.print(tail)?;
            }

            Ok(())
        })
    }
}

/// Term schemes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermScheme {
    /// List of universally quantified variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variables: Vec<TermVar>,
    /// Constraints imposed by the scheme.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<TermConstraint>,
    /// The monomorphic term.
    pub r#type: Box<Term>,
}

impl Parse for TermScheme {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        let mut variables = Vec::new();
        let mut constraints = Vec::new();

        while parser.peek::<Form<kw::forall>>() {
            parser.list(|parser| {
                parser.parse::<kw::forall>()?;
                variables.extend(parser.parse::<Vec<_>>()?);
                Ok(())
            })?;
        }

        while parser.peek::<Form<kw::r#where>>() {
            parser.list(|parser| {
                parser.parse::<kw::r#where>()?;
                constraints.extend(parser.parse::<Vec<_>>()?);
                Ok(())
            })?;
        }

        let r#type = parser.parse()?;

        Ok(Self {
            variables,
            constraints,
            r#type,
        })
    }
}

impl Print for TermScheme {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        if !self.variables.is_empty() {
            printer.list(|printer| {
                printer.print(kw::forall)?;
                printer.print(&self.variables)
            })?;
        }

        if !self.constraints.is_empty() {
            for constraint in &self.constraints {
                printer.list(|printer| {
                    printer.print(kw::r#where)?;
                    printer.print(constraint)
                })?;
            }
        }

        printer.print(&self.r#type)?;
        Ok(())
    }
}

/// Type constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermConstraint {
    /// The name of the type constraint.
    pub name: Symbol,
    /// The arguments to the type constraint.
    pub args: Vec<Term>,
}

impl Parse for TermConstraint {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.list(|parser| {
            let name = parser.parse()?;
            let args = parser.parse()?;
            Ok(Self { name, args })
        })
    }
}

impl Print for TermConstraint {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.list(|printer| {
            printer.print(&self.name)?;
            printer.print(&self.args)?;
            Ok(())
        })
    }
}
