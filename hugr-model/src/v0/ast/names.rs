use std::{fmt::Display, str::FromStr, sync::LazyLock};

use regex::Regex;
use smol_str::{SmolStr, ToSmolStr};
use thiserror::Error;

use crate::v0::{LinkName, Literal, SymbolName, VarName};

use super::literals::{StringParseError, parse_string_literal};

/// Check if a name can be printed as a bare name or if it needs to be wrapped
/// in a string.
fn is_bare_name(name: &str) -> bool {
    static REGEX: LazyLock<Regex> = LazyLock::new(|| {
        let segment = r"[a-zA-Z_][a-zA-Z0-9_]*";
        let regex = format!(r"^([0-9]+)|({}(\.{})*)$", segment, segment);
        Regex::new(&regex).unwrap()
    });
    REGEX.is_match(name)
}

fn parse_name(sigil: char, str: &str) -> Result<SmolStr, NameParseError> {
    let Some(str) = str.strip_prefix(sigil) else {
        return Err(NameParseError::Sigil { expected: sigil });
    };

    if str.starts_with('"') {
        Ok(parse_string_literal(str)?)
    } else {
        Ok(str.to_smolstr())
    }
}

#[derive(Debug, Error)]
pub enum NameParseError {
    #[error("expected sigil `{expected}`")]
    Sigil { expected: char },
    #[error("error parsing string escaped id")]
    String(#[from] StringParseError),
}

macro_rules! impl_name {
    ($sigil:expr, $ident:ident) => {
        impl FromStr for $ident {
            type Err = NameParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let name = parse_name($sigil, s)?;
                Ok(Self::new(name))
            }
        }

        impl Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if is_bare_name(&self.0) {
                    write!(f, "{}{}", $sigil, self.0)
                } else {
                    write!(f, "{}{}", $sigil, Literal::Str(self.0.clone()))
                }
            }
        }
    };
}

impl_name!('@', SymbolName);
impl_name!('?', VarName);
impl_name!('%', LinkName);
