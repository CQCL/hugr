use std::borrow::Borrow;

use derive_more::Display;
use lazy_static::lazy_static;
use regex::Regex;
use smol_str::SmolStr;
use thiserror::Error;

lazy_static! {
    pub static ref NAME_REGEX: Regex = Regex::new(r"^[\w--\d]\w*$").unwrap();
}

#[derive(
    Clone,
    Debug,
    Display,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
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

    /// Create a new Ident *without* doing the well-formedness check.
    /// Useful because we want to have static 'const'ants inside the crate,
    /// but to be used sparingly.
    pub(crate) const fn new_unchecked(n: &str) -> Self {
        Ident(SmolStr::new_inline(n))
    }
}

impl Borrow<str> for Ident {
    fn borrow(&self) -> &str {
        self.0.borrow()
    }
}

impl std::ops::Deref for Ident {
    type Target = str;

    fn deref(&self) -> &str {
        self.0.deref()
    }
}

impl PartialEq<str> for Ident {
    fn eq(&self, other: &str) -> bool {
        self.0.eq(other)
    }
}

impl TryInto<Ident> for &str {
    type Error = InvalidIdentifier;

    fn try_into(self) -> Result<Ident, Self::Error> {
        Ident::new(SmolStr::new(self))
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
