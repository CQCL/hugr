use lazy_static::lazy_static;
use regex::Regex;
use smol_str::SmolStr;
use thiserror::Error;
use derive_more::Display;

lazy_static! {
    pub static ref NAME_REGEX: Regex = Regex::new(r"^[\w--\d]\w*$").unwrap();
}

#[derive(Clone, Debug, Display, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// A well-formed identifier
pub struct Ident(SmolStr);

impl Ident {
    /// Makes an Ident, checking the supplied string is well-formed
    pub fn new(n: impl Into<SmolStr>) -> Result<Self, InvalidIdentifier> {
        let n = n.into();
        if NAME_REGEX.is_match(n.as_str()) {
            Ok(Ident(n))
        } else {
            Err(InvalidIdentifier(n))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("Invalid identifier {0}")]
/// Error indicating a string was not valid as an Ident
pub struct InvalidIdentifier(SmolStr);

#[cfg(test)]
mod test {
    use super::Ident;

    #[test]
    fn test_idents() {
        Ident::new("foo").unwrap();
        Ident::new("_foo").unwrap();
        Ident::new("Bar_xyz67").unwrap();

        Ident::new("foo.bar").unwrap_err();
        Ident::new("42").unwrap_err();
        Ident::new("xyz-5").unwrap_err();
    }
}
