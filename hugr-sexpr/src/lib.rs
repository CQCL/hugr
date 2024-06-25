//! S-expressions as a data format.
//!
//! This crate provides a data type for s-expressions,
//! together with a reader and pretty printer.
//! Moreover, the crate provides derive macros to conveniently convert between
//! user defined types and s-expressions.
//! See [`Value`] for the data model.
//!
//! # Syntax
//!
//! **Lists** are sequences of values, delimited on the outside by `(` and `)`
//! and separated by whitespace.
//!
//! **Strings** are delimited by double quotes `"` on both sides,
//! using the following escaping rules:
//!
//!  - `\"` and `\\` are used to escape `"` and `\`.
//!  - `\n`, `\r` and `\t` stand for the newline, carriage return and tab characters.
//!  - `\u{HEX}` stands in for any unicode character where `HEX` is a UTF-8 codepoint in hexadecimal notation.
//!
//! **Symbols** appear verbatim without delimiters, as long as it satisfies all of the following conditions:
//!
//!  - The symbol consists only of alphanumeric characters and of the special characters `!$%&*/:<=>?^_~+-.@`.
//!  - The symbol does not begin with a digit.
//!  - If the symbol begins with `+` or `-`, the following character (if any) is not a digit.
//!
//! Symbols that are not of this form are delimited by a pipe `|` on both sides.
//! For symbols that are delimited, the same escaping rules apply as for strings,
//! except that the double quote `"` is exchanged for the pipe `|`.
//! Notably the hash sign `#` is reserved and may not appear in a non-delimited symbol.
//! This is to allow for future extensibility if richer data types are required.
//!
//! **Booleans** are encoded by `#t` for true and `#f` for false.
//!
//! **Integers** are represented in text in decimal and with an optional sign,
//! following the format `[+-]?[0-9]+`.
//!
//! **Floats** follow the format
//! `[+-]?[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?`.
//! Positive and negative infinity are denoted by `#+inf` and `#-inf`,
//! while NaN is written as `#nan`.
//!
//! **Comments** begin with a `;` and extend to the end of the line.
//!
//! # Derive Macros
//!
//! Converting between s-expressions and user-defined types can be tedious.
//! Since s-expressions do not cleanly map onto the serde data model, this crate
//! comes with its own derive macros instead.
//! In particular, the [`output::Output`] and [`input::Input`] traits can be derived
//! automatically for structs with named fields.
//!
#[cfg_attr(
    feature = "derive",
    doc = r##"
```
# use hugr_sexpr::input::Input;
# use hugr_sexpr::output::Output;
#[derive(Debug, PartialEq, Input, Output)]
pub struct Person {
    name: String,
    #[sexpr(required)]
    company: String,
    #[sexpr(optional)]
    birthday: Option<String>,
    #[sexpr(repeated)]
    #[sexpr(rename = "email")]
    email_addresses: Vec<String>,
}

let person = Person {
    name: "John Doe".to_string(),
    company: "ACME".to_string(),
    birthday: None,
    email_addresses: vec![
      "john@doe.com".to_string(),
      "john.doe@acme.com".to_string()
    ],
};

let sexpr = r#"
  "John Doe"
  (company "ACME")
  (email "john@doe.com")
  (email "john.doe@acme.com")
"#;

let imported = hugr_sexpr::from_str::<Person>(sexpr).unwrap();
assert_eq!(imported, person);
```
"##
)]
use ordered_float::OrderedFloat;
use smol_str::SmolStr;
use std::fmt::Display;
pub(crate) mod escape;
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
    /// Lists are sequences of zero or more values.
    List(Vec<Self>),

    /// Strings can be any valid UTF-8 string.
    String(SmolStr),

    /// Symbols represent identifiers such as variables or field names.
    /// A symbol can be any valid UTF-8 string.
    Symbol(Symbol),

    /// Booleans.
    Bool(bool),

    /// Signed integers with 64bit precision.
    Int(i64),

    /// Floating point numbers with 64bit precision.
    Float(OrderedFloat<f64>),
}

impl Value {
    /// Attempts to cast this value into a list.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_sexpr::Value;
    /// assert_eq!(Value::List(vec![]).as_list(), Some(&vec![]));
    /// assert_eq!(Value::Int(3).as_list(), None);
    /// ```
    pub fn as_list(&self) -> Option<&Vec<Value>> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }

    /// Attempts to cast this value into a symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_sexpr::{Value, Symbol};
    /// assert_eq!(Value::Symbol("s".into()).as_symbol(), Some(&Symbol::new("s")));
    /// assert_eq!(Value::String("s".into()).as_symbol(), None);
    /// ```
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Value::Symbol(symbol) => Some(symbol),
            _ => None,
        }
    }

    /// Attempts to cast this value into a string.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_sexpr::{Value, Symbol};
    /// # use smol_str::SmolStr;
    /// assert_eq!(Value::String("s".into()).as_string(), Some(&SmolStr::new("s")));
    /// assert_eq!(Value::Symbol("s".into()).as_string(), None);
    /// ```
    pub fn as_string(&self) -> Option<&SmolStr> {
        match self {
            Value::String(string) => Some(string),
            _ => None,
        }
    }

    /// Attempts to cast this value into an integer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_sexpr::{Value, Symbol};
    /// assert_eq!(Value::Int(12).as_int(), Some(12));
    /// assert_eq!(Value::Float((12.5).into()).as_int(), None);
    /// ```
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(int) => Some(*int),
            _ => None,
        }
    }

    /// Attempts to cast this value into a float.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_sexpr::{Value, Symbol};
    /// assert_eq!(Value::Float((12.5).into()).as_float(), Some(12.5));
    /// assert_eq!(Value::Int(12).as_float(), None);
    /// ```
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(float) => Some(float.into_inner()),
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

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(value.into())
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

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(OrderedFloat(value))
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

impl proptest::arbitrary::Arbitrary for Symbol {
    type Parameters = ();
    type Strategy = proptest::strategy::SBoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;
        any::<String>().prop_map(Symbol::from).sboxed()
    }
}

impl proptest::arbitrary::Arbitrary for Value {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::prelude::*;

        let leaf = prop_oneof![
            any::<bool>().prop_map(Value::from),
            any::<i64>().prop_map(Value::from),
            any::<Symbol>().prop_map(Value::from),
            any::<String>().prop_map(Value::from),
            proptest::num::f64::ANY.prop_map(Value::from)
        ];

        leaf.prop_recursive(8, 256, 10, |inner| {
            proptest::collection::vec(inner, 0..10).prop_map(Value::List)
        })
        .boxed()
    }
}

#[cfg(test)]
mod test {
    use super::{from_str, to_string_pretty, Value};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn pretty_then_parse(values: Vec<Value>, width in 0..120usize) {
            let pretty = to_string_pretty(&values, width);
            let parsed: Vec<Value> = from_str(&pretty).unwrap();
            assert_eq!(values, parsed);
        }
    }
}
