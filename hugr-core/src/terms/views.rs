use std::convert::Infallible;

use super::{Apply, List, Term};
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

        impl $ident {
            pub const SYMBOL: SymbolName = SymbolName::new_static($symbol);
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

        impl From<$ident> for Term {
            fn from(value: $ident) -> Self {
                Term::Apply(value.into())
            }
        }
    };

    ($ident:ident, $symbol:expr) => {
        impl TryFrom<Term> for $ident {
            type Error = ViewError;

            fn try_from(value: Term) -> Result<Self, Self::Error> {
                let [] = value.view_apply(&Self::SYMBOL)?;
                Ok(Self)
            }
        }

        impl $ident {
            pub const SYMBOL: SymbolName = SymbolName::new_static($symbol);
        }

        impl From<$ident> for Apply {
            fn from($ident: $ident) -> Self {
                static TERM: ::once_cell::sync::Lazy<Apply> = ::once_cell::sync::Lazy::new(|| {
                    Apply::new($ident::SYMBOL, [], Term::Wildcard)
                });
                TERM.clone()
            }
        }

        impl From<$ident> for Term {
            fn from(value: $ident) -> Self {
                Term::Apply(value.into())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreFn {
    pub inputs: List,
    pub outputs: List,
}

make_ctr!(CoreFn { inputs, outputs }, "core.fn");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreList {
    pub item_type: Term,
}

make_ctr!(CoreList { item_type }, "core.list");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreTuple {
    pub item_types: List,
}

make_ctr!(CoreTuple { item_types }, "core.tuple");

#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreStr;

make_ctr!(CoreStr, "core.str");

#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreNat;

make_ctr!(CoreNat, "core.nat");

#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreBytes;

make_ctr!(CoreBytes, "core.bytes");

#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
pub struct CoreFloat;

make_ctr!(CoreFloat, "core.float");

// impl CoreFn {
//     pub const SYMBOL: SymbolName = SymbolName::new_static("core.fn");
// }

// impl TryFrom<Term> for CoreFn {
//     type Error = ViewError;

//     fn try_from(value: Term) -> Result<Self, Self::Error> {
//         let [inputs, outputs] = value.view_apply(&Self::SYMBOL)?;
//         let inputs = inputs.try_into()?;
//         let outputs = outputs.try_into()?;
//         Ok(CoreFn { inputs, outputs })
//     }
// }

// impl From<CoreFn> for Term {
//     fn from(value: CoreFn) -> Self {
//         Apply::new(
//             CoreFn::SYMBOL,
//             [value.inputs.into(), value.outputs.into()],
//             Term::default(),
//         )
//         .into()
//     }
// }

#[derive(Debug, Error)]
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
