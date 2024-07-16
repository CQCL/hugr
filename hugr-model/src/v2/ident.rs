//! Identifiers

use serde::{Deserialize, Serialize};

macro_rules! make_id {
    {
        $name:expr, $sigil:literal,
        $(#[$meta:meta])*
        $vis:vis struct $ident:ident;
    } => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        $vis struct $ident(::smol_str::SmolStr);

        impl $ident {
            /// Create a new identifier from a string.
            pub fn new(str: impl AsRef<str>) -> Self {
                Self(str.as_ref().into())
            }
        }

        impl ::std::fmt::Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str($sigil)?;
                f.write_str(self.0.as_ref())
            }
        }

        impl ::parens::parser::Parse for $ident {
            fn parse(parser: &mut ::parens::parser::Parser<'_>) -> ::parens::parser::Result<Self> {
                parser.step(|cursor| match cursor.atom() {
                    Some((atom, rest)) if atom.starts_with($sigil) => Ok((Self(atom[$sigil.len()..].into()), rest)),
                    _ => Err(cursor.error(format!("expected {} (starts with {})", $name, $sigil))),
                })
            }
        }

        impl ::parens::parser::Peek for $ident {
            fn peek(cursor: ::parens::parser::Cursor<'_>) -> bool {
                cursor.peek_atom(|atom| atom.starts_with($sigil))
            }
        }

        impl ::parens::printer::Print for $ident {
            fn print<P: ::parens::printer::Printer>(&self, printer: &mut P) -> Result<(), P::Error> {
                printer.atom(&self.to_string())
            }
        }
    }
}

make_id!(
    "variable",
    "%",
    /// Variable name, starting with a `%` sigil.
    #[derive(Serialize, Deserialize)]
    pub struct VarName;
);

make_id!(
    "type variable",
    "%",
    /// Type variable name, starting with a `%` sigil.
    #[derive(Serialize, Deserialize)]
    pub struct TypeVar;
);

make_id!(
    "label",
    ":",
    /// Label, starting with a `:` sigil.
    #[derive(Serialize, Deserialize)]
    pub struct Label;
);

make_id!(
    "symbol",
    "@",
    /// Symbol, starting with an `@` sigil.
    #[derive(Serialize, Deserialize)]
    pub struct Symbol;
);
