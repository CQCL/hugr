use std::convert::Infallible;

use super::{Apply, Term, Typed};
use hugr_model::v0::SymbolName;
use thiserror::Error;

macro_rules! make_ctr {
    ($ident:ident { $($arg:ident),* }, $symbol:expr) => {
        impl TryFrom<Term> for $ident {
            type Error = ViewError;

            fn try_from(value: Term) -> Result<Self, Self::Error> {
                let [$($arg),*] = value.view_apply(&Self::SYMBOL)?;
                $(let $arg = $arg.try_into()?;)*
                Ok(Self { $($arg),* })
            }
        }

        impl From<$ident> for Apply {
            fn from(value: $ident) -> Self {
                Apply::new(
                    $ident::SYMBOL,
                    [$(value.$arg.into()),*],
                    Term::default(),
                )
            }
        }

        impl From<&$ident> for Apply {
            fn from(value: &$ident) -> Self {
                value.clone().into()
            }
        }

        make_ctr!(@common $ident, $symbol);
    };

    ($ident:ident, $symbol:expr) => {
        impl TryFrom<Term> for $ident {
            type Error = ViewError;

            fn try_from(value: Term) -> Result<Self, Self::Error> {
                let [] = value.view_apply(&Self::SYMBOL)?;
                Ok(Self)
            }
        }

        impl From<$ident> for Apply {
            fn from($ident: $ident) -> Self {
                static TERM: ::once_cell::sync::Lazy<Apply> = ::once_cell::sync::Lazy::new(|| {
                    Apply::new($ident::SYMBOL, [], Term::Wildcard)
                });
                TERM.clone()
            }
        }

        impl From<&$ident> for Apply {
            fn from(value: &$ident) -> Self {
                value.into()
            }
        }

        make_ctr!(@common $ident, $symbol);
    };

    (@common $ident:ident, $symbol:expr) => {
        impl $ident {
            /// The symbol name of the constructor.
            pub const SYMBOL: SymbolName = SymbolName::new_static($symbol);
        }

        impl From<$ident> for Term {
            fn from(value: $ident) -> Self {
                Term::Apply(value.into())
            }
        }

        impl From<&$ident> for Term {
            fn from(value: &$ident) -> Self {
                Term::Apply(value.into())
            }
        }

        impl ::std::fmt::Display for $ident {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}", Term::from(self))
            }
        }
    }
}

macro_rules! impl_typed_static {
    ($ident:ident) => {
        impl Typed for $ident {
            #[allow(refining_impl_trait)]
            fn type_(&self) -> Term {
                Term::StaticType
            }
        }
    };
}

/// `core.fn`: The type of runtime functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreFn {
    pub inputs: Term,
    pub outputs: Term,
}

make_ctr!(CoreFn { inputs, outputs }, "core.fn");
impl_typed_static!(CoreFn);

/// `core.list`: The type of static lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreList {
    pub item_type: Term,
}

make_ctr!(CoreList { item_type }, "core.list");
impl_typed_static!(CoreList);

/// `core.tuple`: The type of static tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreTuple {
    pub item_types: Term,
}

make_ctr!(CoreTuple { item_types }, "core.tuple");
impl_typed_static!(CoreTuple);

/// `core.str`: The type of static strings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreStr;

make_ctr!(CoreStr, "core.str");
impl_typed_static!(CoreStr);

/// `core.nat`: The type of static natural numbers.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreNat;

make_ctr!(CoreNat, "core.nat");
impl_typed_static!(CoreNat);

/// `core.bytes`: The type of static byte strings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreBytes;

make_ctr!(CoreBytes, "core.bytes");
impl_typed_static!(CoreBytes);

/// `core.float`: The type of static floating point numbers.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreFloat;

make_ctr!(CoreFloat, "core.float");
impl_typed_static!(CoreFloat);

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ViewError {
    /// The viewed term does not match the pattern of the view.
    ///
    /// This is distinct from [`ViewError::Invalid`] in order to allow for
    /// views to be used in pattern matching.
    #[error("term mismatch")]
    Mismatch,
    /// The viewed term contains an uninferred subterm.
    ///
    /// This does not necessarily mean that the term is invalid since it
    /// may at a later stage be inferred to be a concrete term.
    #[error("uninferred term")]
    Uninferred,
    /// The viewed term contains a variable subterm.
    ///
    /// This does not necessarily mean that the term is invalid since the
    /// variable may at a later stage be substituted with a concrete term.
    #[error("term variable")]
    Variable,
    #[error("invalid term: {0}")]
    Invalid(String),
}

impl From<Infallible> for ViewError {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}
