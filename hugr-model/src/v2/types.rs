//! Types.

use parens::parser::{self, Parse, Parser, Peek};
use parens::printer::{Print, Printer};
use parens::util::Form;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use super::ident::{Label, Symbol, TypeVar};
use super::keywords as kw;

/// A type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Type {
    /// Type constructor application.
    #[serde(rename = "ctr")]
    Ctr(TypeCtr),
    /// Type variable.
    #[serde(rename = "var")]
    Var(TypeVar),
    /// Row type.
    #[serde(rename = "row")]
    Row(TypeRow),
    /// List type.
    #[serde(rename = "list")]
    List(TypeList),
    /// Label.
    #[serde(rename = "label")]
    Label(Label),
    /// Literal integer.
    #[serde(rename = "int")]
    LitInt(i64),
    /// Literal string.
    #[serde(rename = "str")]
    LitStr(SmolStr),
}

impl Parse for Type {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        if parser.peek::<TypeCtr>() {
            Ok(Self::Ctr(parser.parse()?))
        } else if parser.peek::<TypeVar>() {
            Ok(Self::Var(parser.parse()?))
        } else if parser.peek::<TypeRow>() {
            Ok(Self::Row(parser.parse()?))
        } else if parser.peek::<TypeList>() {
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

impl Print for Type {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        match self {
            Type::Ctr(ctr) => printer.print(ctr),
            Type::Var(var) => printer.print(var),
            Type::Row(row) => printer.print(row),
            Type::List(list) => printer.print(list),
            Type::Label(label) => printer.print(label),
            Type::LitInt(int) => printer.print(int),
            Type::LitStr(str) => printer.list(|printer| {
                printer.print(kw::str)?;
                printer.print(str)
            }),
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
pub struct TypeCtr {
    /// The name of the type constructor.
    pub ctr: Symbol,
    /// The arguments to the type constructor.
    pub args: Vec<Type>,
}

impl Parse for TypeCtr {
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

impl Peek for TypeCtr {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek::<Symbol>() || cursor.peek_list(|cursor| cursor.peek::<Symbol>())
    }
}

impl Print for TypeCtr {
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

/// A row type.
///
/// Rows have an optional tail.
/// If the tail is present the row is called *open*, and otherwise *closed*.
/// A row whose tail is again a row is considered equivalent to a single
/// concatenated row.
/// An open row is *proper* when its tail is either a proper row or a type variable,
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
pub struct TypeRow {
    /// The entries in the row type.
    pub entries: Vec<TypeRowEntry>,
    /// The tail of the row or `None` if the row is closed.
    pub tail: Option<Box<Type>>,
}

impl Parse for TypeRow {
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

            Ok(TypeRow { entries, tail })
        })
    }
}

impl Peek for TypeRow {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek_map(|_| true)
    }
}

impl Print for TypeRow {
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
pub struct TypeRowEntry {
    /// The label of the row entry.
    pub label: Label,
    /// The type associated to the row entry.
    pub value: Box<Type>,
}

impl Parse for TypeRowEntry {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.list(|parser| {
            let label = parser.parse()?;
            let value = parser.parse()?;
            Ok(TypeRowEntry { label, value })
        })
    }
}

impl Print for TypeRowEntry {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.list(|printer| {
            printer.print(&self.label)?;
            printer.print(&self.value)
        })
    }
}

/// A type level list.
///
/// Lists have an optional tail.
/// If the tail is present the list is called *open*, and otherwise *closed*.
/// A list whose tail is again a list is considered equivalent to a single
/// concatenated list.
/// An open list is *proper* when its tail is either a proper list or a type variable,
/// and *improper* otherwise. A closed list is always *proper*.
///
/// # Syntax
///
/// A closed list is a sequence of types delimited by square brackets: `[item-1 ... item-n]`.
/// Open lists are denoted as `[item-1 ... item-n . tail]` where `tail` is again a type.
///
/// # Examples
///
///  - Inputs and outputs of a function: `(@fn [@f32 @f32] [@f32])`
///  - The type of an `eval` function, using open lists: `(@fn [(@fn %i %o) . %i] [%o])`
///  - The types in a tuple: `(@tuple [@int @str])`
///  - Dimensions of a tensor: `(@tensor @f32 [512 128 128])`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeList {
    /// The items in the list.
    pub items: Vec<Type>,
    /// The tail of the list, or `None` if the list is closed.
    pub tail: Option<Box<Type>>,
}

impl Parse for TypeList {
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

            Ok(TypeList { items, tail })
        })
    }
}

impl Peek for TypeList {
    fn peek(cursor: parser::Cursor<'_>) -> bool {
        cursor.peek_seq(|_| true)
    }
}

impl Print for TypeList {
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

/// Type schemes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeScheme {
    /// List of universally quantified type variables.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variables: Vec<TypeVar>,
    /// Type constraints imposed by the type scheme.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<TypeConstraint>,
    /// The monomorphic type.
    pub r#type: Box<Type>,
}

impl Parse for TypeScheme {
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

impl Print for TypeScheme {
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
pub struct TypeConstraint {
    /// The name of the type constraint.
    pub name: Symbol,
    /// The arguments to the type constraint.
    pub args: Vec<Type>,
}

impl Parse for TypeConstraint {
    fn parse(parser: &mut Parser<'_>) -> parser::Result<Self> {
        parser.list(|parser| {
            let name = parser.parse()?;
            let args = parser.parse()?;
            Ok(Self { name, args })
        })
    }
}

impl Print for TypeConstraint {
    fn print<P: Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
        printer.list(|printer| {
            printer.print(&self.name)?;
            printer.print(&self.args)?;
            Ok(())
        })
    }
}
