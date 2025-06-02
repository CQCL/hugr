//! Term views for the `core` extension.

use hugr_model::v0::SymbolName;
use once_cell::sync::Lazy;
use servo_arc::ThinArc;

use super::{Term, View, ViewError};
use crate::{
    term_view_ctr,
    terms::{TermData, TermHeader},
};

/// `core.static`: The type of static types.
///
/// The type of `core.static` is `core.static` itself:
/// ```
/// # use hugr_core::terms::core::Static;
/// # use hugr_core::terms::Term;
/// assert_eq!(Term::from(Static).type_(), Term::from(Static));
/// ```
pub struct Static;

impl Static {
    /// The name of the type constructor.
    pub const CTR_NAME: SymbolName = SymbolName::new_static("core.static");
}

impl From<Static> for Term {
    fn from(_: Static) -> Self {
        static STATIC: Lazy<Term> = Lazy::new(|| {
            Term(ThinArc::from_header_and_iter_alloc(
                |layout| unsafe { std::alloc::alloc(layout) },
                TermHeader::new(TermData::Apply(Static::CTR_NAME), None, &[]),
                std::iter::empty(),
                0,
                true,
            ))
        });
        STATIC.clone()
    }
}

impl View for Static {
    fn view(term: &Term) -> Result<Self, ViewError> {
        let [] = term.view_apply(&Self::CTR_NAME)?;
        Ok(Self)
    }
}

term_view_ctr! {
    "core.fn";
    /// `core.fn`: The type of runtime functions.
    pub struct Fn {
        /// The list of input types of the function.
        pub inputs: Term,
        /// The list of output types of the function.
        pub outputs: Term,
    }
}

term_view_ctr! {
    "core.const";
    /// `core.const`: The type of runtime constants.
    pub struct Const {
        /// The runtime type of the constant.
        pub r#type: Term,
    }
}

term_view_ctr! {
    "core.type";
    /// `core.type`: The type of runtime types.
    pub struct Type;
}

term_view_ctr! {
    "core.type";
    /// `core.constraint`: The type of constraints.
    pub struct Constraint;
}

term_view_ctr! {
    "core.nonlinear";
    /// `core.nonlinear`: The non-linear constraint.
    pub struct NonLinear {
        /// The type being constrained.
        pub r#type: Term,
    }
}

term_view_ctr! {
    "core.meta";
    /// `core.meta`: The type of metadata.
    pub struct Meta;
}

term_view_ctr! {
    "core.adt";
    /// `core.adt`: Runtime algebraic data type.
    pub struct Adt {
        /// The list of variants.
        pub variants: Term,
    }
}

term_view_ctr! {
    "core.str";
    /// `core.str`: The type of string literals.
    pub struct Str;
}

term_view_ctr! {
    "core.nat";
    /// `core.nat`: The type of natural number literals.
    pub struct Nat;
}

term_view_ctr! {
    "core.bytes";
    /// `core.bytes`: The type of byte string literals.
    pub struct Bytes;
}

term_view_ctr! {
    "core.float";
    /// `core.float`: The type of floating point literals.
    pub struct Float;
}

term_view_ctr! {
    "core.list";
    /// `core.list`: The type of static lists.
    pub struct List {
        /// The static type of the list items.
        pub item_type: Term,
    }
}

term_view_ctr! {
    "core.tuple";
    /// `core.tuple`: The type of static tuples.
    pub struct Tuple {
        /// The list of the static types of the tuple items.
        pub item_types: Term,
    }
}

/// Term views for the `core.order_hint` extension.
pub mod order_hint {
    use crate::term_view_ctr;

    term_view_ctr! {
        "core.order_hint.order";
        /// `core.order_hint.order`: Metadata constructor for order hints.
        pub struct Order {
            /// The order key that has to be run first.
            pub before: u64,
            /// The order key that has to be run after.
            pub after: u64,
        }
    }

    term_view_ctr! {
        "core.order_hint.key";
        /// `core.order_hint.key`: Metadata constructor to attach order keys to nodes.
        pub struct Key {
            /// The order key to attach to the node.
            pub key: u64,
        }
    }
}
