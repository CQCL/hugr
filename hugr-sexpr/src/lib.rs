//! S-expressions as a data format.
//!
//! This crate provides a data type for s-expressions,
//! together with a reader and pretty printer.
//! Moreover, the crate provides derive macros to conveniently convert between
//! user defined types and s-expressions.
//!
//! # Derive Macros
//!
//! Converting between s-expressions and user-defined types can be tedious.
//! Since s-expressions do not cleanly map onto the serde data model, this crate
//! comes with its own derive macros instead.
//! In particular, the [`export::Export`] and [`import::Import`] traits can be derived
//! automatically for structs with named fields.
//!
//! ```
//! # use hugr_sexpr::import::Import;
//! # use hugr_sexpr::export::Export;
//! # use hugr_sexpr::read_values;
//! #[derive(Debug, PartialEq, Import, Export)]
//! pub struct Person {
//!     name: String,
//!     #[sexpr(required)]
//!     company: String,
//!     #[sexpr(optional)]
//!     birthday: Option<String>,
//!     #[sexpr(repeated)]
//!     #[sexpr(rename = "email")]
//!     email_addresses: Vec<String>,
//! }
//!
//! let person = Person {
//!     name: "John Doe".to_string(),
//!     company: "ACME".to_string(),
//!     birthday: None,
//!     email_addresses: vec![
//!       "john@doe.com".to_string(),
//!       "john.doe@acme.com".to_string()
//!     ],
//! };
//!
//! let sexpr = r#"
//!   "John Doe"
//!   (company "ACME")
//!   (email "john@doe.com")
//!   (email "john.doe@acme.com")
//! "#;
//!
//! let values = read_values(sexpr).unwrap();
//! let (_, imported) = Person::import(&values).unwrap();
//! assert_eq!(imported, person);
//! ```
use smol_str::SmolStr;
use std::fmt::Display;
pub mod export;
pub mod import;
mod pretty;
mod read;

pub use read::{read_values, read_values_with_span, ReadError, Span};

/// A value that can be encoded as an s-expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value<A = ()> {
    /// A list of values.
    List(Vec<Self>, A),
    /// A string.
    String(String, A),
    /// A symbol.
    Symbol(Symbol, A),
    /// A boolean.
    Bool(bool, A),
    /// An integer.
    Int(i64, A), // TODO: More flexible number types?
}

impl<A> Value<A> {
    /// Pretty prints a value into a string.
    pub fn pretty(&self, width: usize) -> String {
        pretty::value_to_string(self, width)
    }

    /// Returns the metadata attached to this value.
    pub fn meta(&self) -> &A {
        match self {
            Value::List(_, meta) => meta,
            Value::String(_, meta) => meta,
            Value::Symbol(_, meta) => meta,
            Value::Bool(_, meta) => meta,
            Value::Int(_, meta) => meta,
        }
    }

    /// Applies a function to the value's metadata.
    pub fn map_meta<F, B>(&self, mut f: F) -> Value<B>
    where
        F: FnMut(&A) -> B + Copy,
    {
        match self {
            Value::List(list, meta) => {
                let list = list.iter().map(|value| value.map_meta(f)).collect();
                let meta = f(meta);
                Value::List(list, meta)
            }
            Value::String(string, meta) => Value::String(string.clone(), f(meta)),
            Value::Symbol(symbol, meta) => Value::Symbol(symbol.clone(), f(meta)),
            Value::Bool(bool, meta) => Value::Bool(*bool, f(meta)),
            Value::Int(int, meta) => Value::Int(*int, f(meta)),
        }
    }

    /// Attempts to cast this value into a list.
    pub fn as_list(&self) -> Option<&[Self]> {
        match self {
            Value::List(list, _) => Some(list),
            _ => None,
        }
    }

    /// Attempts to cast this value into a list that begins with a symbol.
    pub fn as_list_with_head(&self) -> Option<(&Symbol, &[Self])> {
        let (head, values) = self.as_list()?.split_first()?;
        let head = head.as_symbol()?;
        Some((head, values))
    }

    /// Attempts to cast this value into a symbol.
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Value::Symbol(symbol, _) => Some(symbol),
            _ => None,
        }
    }

    /// Attempts to cast this value into a string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(string, _) => Some(string),
            _ => None,
        }
    }

    /// Attempts to cast this value into an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(int, _) => Some(*int),
            _ => None,
        }
    }
}

impl<A> Display for Value<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(80);
        pretty::write_value(self, width, f)
    }
}

impl<A> From<Vec<Value<A>>> for Value<A>
where
    A: Default,
{
    fn from(value: Vec<Value<A>>) -> Self {
        Value::List(value, A::default())
    }
}

/// A symbol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Symbol(SmolStr);

impl Symbol {
    /// Create a new [`Symbol`] from a string.
    pub fn new(string: impl AsRef<str>) -> Self {
        Self(SmolStr::new(string))
    }
}

impl From<SmolStr> for Symbol {
    fn from(value: SmolStr) -> Self {
        Self(value)
    }
}

impl From<Symbol> for SmolStr {
    fn from(value: Symbol) -> Self {
        value.0
    }
}

impl From<String> for Symbol {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<Symbol> for String {
    fn from(value: Symbol) -> Self {
        value.0.into()
    }
}

impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl From<&str> for Symbol {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_ref())
    }
}
