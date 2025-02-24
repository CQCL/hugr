use std::borrow::Borrow;

use derive_more::Display;
use lazy_static::lazy_static;
use regex::Regex;
use smol_str::SmolStr;
use thiserror::Error;

pub static PATH_COMPONENT_REGEX_STR: &str = r"[\w--\d]\w*";
lazy_static! {
    pub static ref PATH_REGEX: Regex =
        Regex::new(&format!(r"^{0}(\.{0})*$", PATH_COMPONENT_REGEX_STR)).unwrap();
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

/// A non-empty dot-separated list of valid identifiers
pub struct IdentList(SmolStr);

impl IdentList {
    /// Makes an IdentList, checking the supplied string is well-formed
    pub fn new(n: impl Into<SmolStr>) -> Result<Self, InvalidIdentifier> {
        let n = n.into();
        if PATH_REGEX.is_match(n.as_str()) {
            Ok(IdentList(n))
        } else {
            Err(InvalidIdentifier(n))
        }
    }

    /// Split off the last component of the path, returning the prefix and suffix.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr_core::hugr::IdentList;
    /// assert_eq!(
    ///     IdentList::new("foo.bar.baz").unwrap().split_last(),
    ///     Some((IdentList::new_unchecked("foo.bar"), "baz".into()))
    /// );
    /// assert_eq!(
    ///    IdentList::new("foo").unwrap().split_last(),
    ///    None
    /// );
    /// ```
    pub fn split_last(&self) -> Option<(IdentList, SmolStr)> {
        let (prefix, suffix) = self.0.rsplit_once('.')?;
        let prefix = Self::new_unchecked(prefix);
        let suffix = suffix.into();
        Some((prefix, suffix))
    }

    /// Create a new [IdentList] *without* doing the well-formedness check.
    /// This is a backdoor to be used sparingly, as we rely upon callers to
    /// validate names themselves. In tests, instead the [crate::const_extension_ids]
    /// macro is strongly encouraged as this ensures the name validity check
    /// is done properly.
    pub const fn new_unchecked(n: &str) -> Self {
        IdentList(SmolStr::new_inline(n))
    }
}

impl Borrow<str> for IdentList {
    fn borrow(&self) -> &str {
        self.0.borrow()
    }
}

impl std::ops::Deref for IdentList {
    type Target = str;

    fn deref(&self) -> &str {
        self.0.deref()
    }
}

impl TryInto<IdentList> for &str {
    type Error = InvalidIdentifier;

    fn try_into(self) -> Result<IdentList, InvalidIdentifier> {
        IdentList::new(SmolStr::new(self))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("Invalid identifier {0}")]
/// Error indicating a string was not valid as an [IdentList]
pub struct InvalidIdentifier(SmolStr);

#[cfg(test)]
mod test {

    mod proptest {
        use crate::hugr::ident::IdentList;
        use ::proptest::prelude::*;
        impl Arbitrary for super::IdentList {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                use crate::proptest::any_ident_string;
                use proptest::collection::vec;
                vec(any_ident_string(), 1..2)
                    .prop_map(|vs| {
                        IdentList::new(itertools::intersperse(vs, ".".into()).collect::<String>())
                            .unwrap()
                    })
                    .boxed()
            }
        }
        proptest! {
            #[test]
            fn arbitrary_identlist_valid((IdentList(ident_list)): IdentList) {
                assert!(IdentList::new(ident_list).is_ok())
            }
        }
    }

    use super::IdentList;

    #[test]
    fn test_idents() {
        IdentList::new("foo").unwrap();
        IdentList::new("_foo").unwrap();
        IdentList::new("Bar_xyz67").unwrap();
        IdentList::new("foo.bar").unwrap();
        IdentList::new("foo.bar.baz").unwrap();

        IdentList::new("42").unwrap_err();
        IdentList::new("foo.42").unwrap_err();
        IdentList::new("xyz-5").unwrap_err();
        IdentList::new("foo..bar").unwrap_err();
        IdentList::new(".foo").unwrap_err();
    }
}
