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
//! In particular, the [`output::Output`] and [`input::Input`] traits can be derived
//! automatically for structs with named fields.
//!
//! ```
//! # use hugr_sexpr::input::Input;
//! # use hugr_sexpr::output::Output;
//! #[derive(Debug, PartialEq, Input, Output)]
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
//! let imported = hugr_sexpr::from_str::<Person>(sexpr).unwrap();
//! assert_eq!(imported, person);
//! ```
use smol_str::SmolStr;
use std::fmt::Display;
pub mod input;
pub mod output;
pub mod pretty;
pub mod read;

pub use output::to_values;
pub use pretty::{to_fmt_pretty, to_string_pretty};
pub use read::from_str;

/// A value that can be encoded as an s-expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// A list of values.
    List(Vec<Self>),
    /// A string.
    String(SmolStr),
    /// A symbol.
    Symbol(Symbol),
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64), // TODO: More flexible number types?
}

impl Value {
    /// Attempts to cast this value into a list.
    pub fn as_list(&self) -> Option<&[Self]> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }

    /// Attempts to cast this value into a symbol.
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Value::Symbol(symbol) => Some(symbol),
            _ => None,
        }
    }

    /// Attempts to cast this value into a string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(string) => Some(string),
            _ => None,
        }
    }

    /// Attempts to cast this value into an integer.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(int) => Some(*int),
            _ => None,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = f.width().unwrap_or(80);
        to_fmt_pretty(self, width, f)
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Value::List(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value.into())
    }
}

impl From<SmolStr> for Value {
    fn from(value: SmolStr) -> Self {
        Value::String(value)
    }
}

impl From<Symbol> for Value {
    fn from(value: Symbol) -> Self {
        Value::Symbol(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
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
