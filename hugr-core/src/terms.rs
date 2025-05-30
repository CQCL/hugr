//! Terms in the static language that parameterises Hugrs.
use fxhash::FxHasher;
use hugr_model::v0::ast;
use hugr_model::v0::{Literal, SymbolName, VarName};
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use servo_arc::ThinArc;
use smol_str::SmolStr;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use thiserror::Error;
use util::ListIter;

use crate::Node;

pub mod core;
mod errors;
pub mod util;

/// A term in the static language that is used to parameterise `Hugr`s.
///
/// To pattern match the values of a term use [`Term::get`].
///
/// Terms are immutable and have no identity beyond structural equality. Two
/// structurally equal terms should be considered logically indistinguishable.
/// Equal terms can share their representation and this sharing can be observed
/// to enable optimisations.
///
/// Terms can be cloned cheaply without requiring a copy of the term's data.
/// Equality comparisons and hash codes are optimised so that [`Term`]s can be
/// used as keys in hash tables.
///
/// A term has a unique type within the context of a hugr. In particular, local
/// variables refer to their binding site. The same term may have different
/// types when interpreted in the context of different hugrs. This enables
/// caching types and constraints in a side table instead of requiring them to
/// be stored inline with the term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term(ThinArc<TermHeader, Term>);

/// Assert that [`Term`] is the same size as a pointer.
const _: () = assert!(std::mem::size_of::<Term>() == std::mem::size_of::<usize>());

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.header.hash);
    }
}

/// The constant-sized part of the [`Term`].
///
/// See [`TermKind`] for more information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
enum TermData {
    #[default]
    Wildcard,
    Apply(SymbolName),
    Literal(Literal),
    List(Term),
    ListConcat(Term),
    Tuple(Term),
    TupleConcat(Term),
    Var(Var),
    Func(Term, Node),
}

impl TermData {
    pub fn split<'a>(term: TermKind<'a>) -> (Self, &'a [Term]) {
        match term {
            TermKind::Wildcard => (TermData::Wildcard, &[]),
            TermKind::Apply(symbol, terms) => (TermData::Apply(symbol.clone()), terms),
            TermKind::Literal(literal) => (TermData::Literal(literal.clone()), &[]),
            TermKind::List(item_type, items) => (TermData::List(item_type.clone()), items),
            TermKind::ListConcat(item_type, lists) => {
                (TermData::ListConcat(item_type.clone()), lists)
            }
            TermKind::Tuple(item_types, items) => (TermData::Tuple(item_types.clone()), items),
            TermKind::TupleConcat(item_types, tuples) => {
                (TermData::TupleConcat(item_types.clone()), tuples)
            }
            TermKind::Var(var) => (TermData::Var(var.clone()), &[]),
            TermKind::Func(signature, node) => (TermData::Func(signature.clone(), node), &[]),
        }
    }
}

/// The constant-sized part of the [`Term`] together with derived data.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TermHeader {
    /// Cached hash code (derived).
    ///
    /// With a cached hash code we can calculate a [`Term`]s hash quickly
    /// without having to traverse the entire term. The cached hash code can
    /// also speed up the comparison of terms in the case where the compared
    /// terms are not equal.
    hash: u64,

    /// The constant-sized part without derived data.
    data: TermData,
}

impl TermHeader {
    /// Creates a new term header by computing the derived information.
    pub fn new(data: TermData, terms: &[Term]) -> Self {
        let hash = {
            let mut hasher = FxHasher::default();
            data.hash(&mut hasher);
            terms.hash(&mut hasher);
            hasher.finish()
        };

        Self { data, hash }
    }
}

impl Term {
    /// Create a new [`Term`].
    pub fn new(term: TermKind) -> Self {
        let (data, terms) = TermData::split(term);
        Self(ThinArc::from_header_and_iter(
            TermHeader::new(data, terms),
            terms.iter().cloned(),
        ))
    }

    /// Create a new static [`Term`] without reference counting.
    ///
    /// Terms created with this method will never be deallocated. This is useful
    /// for [`Term`]s that are known at Rust compile time.
    pub fn new_static(term: TermKind) -> Self {
        let (data, terms) = TermData::split(term);
        Self(ThinArc::from_header_and_iter_alloc(
            |layout| unsafe { std::alloc::alloc(layout) },
            TermHeader::new(data, terms),
            terms.iter().cloned(),
            terms.len(),
            true,
        ))
    }

    /// Borrows this term's data as a [`TermKind`].
    #[inline]
    pub fn get(&self) -> TermKind {
        match &self.0.header.data {
            TermData::Wildcard => TermKind::Wildcard,
            TermData::Apply(symbol) => TermKind::Apply(symbol, self.0.slice()),
            TermData::Literal(literal) => TermKind::Literal(literal),
            TermData::List(item_type) => TermKind::List(item_type, self.0.slice()),
            TermData::ListConcat(item_type) => TermKind::List(item_type, self.0.slice()),
            TermData::Tuple(item_types) => TermKind::Tuple(item_types, self.0.slice()),
            TermData::TupleConcat(item_types) => TermKind::TupleConcat(item_types, self.0.slice()),
            TermData::Var(var) => TermKind::Var(var),
            TermData::Func(signature, node) => TermKind::Func(signature, *node),
        }
    }

    /// Tries to apply a [`View`] to this term.
    #[inline]
    pub fn view<V: View>(&self) -> Result<V, ViewError> {
        V::view(self)
    }

    /// Tries to match this term with a symbol application, given a symbol name and statically known arity.
    ///
    /// Instead of using this method directly, consider implementing a view type for the symbol.
    ///
    /// # Errors
    ///
    /// - [`ViewError::Mismatch`] when the term is not a symbol application.
    /// - [`ViewError::Mismatch`] when the term is a symbol application for a different symbol.
    /// - [`ViewError::Mismatch`] when the term is a symbol application with more arguments than expected.
    /// - [`ViewError::Uninferred`] when the term is a wildcard.
    /// - [`ViewError::Variable`] when the term is a variable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_core::terms::{Term, TermKind, ViewError};
    /// # use hugr_model::v0::SymbolName;
    /// let this = SymbolName::new_static("this");
    /// let that = SymbolName::new_static("that");
    ///
    /// let arg0 = Term::from(0u64);
    /// let arg1 = Term::from(1u64);
    /// let term = Term::new(TermKind::Apply(&this, &[arg0.clone(), arg1.clone()]));
    ///
    /// assert_eq!(term.view_apply(&this), Ok([arg0.clone(), arg1.clone()]));
    /// assert_eq!(term.view_apply::<2>(&that), Err(ViewError::Mismatch));
    /// assert_eq!(Term::from(42u64).view_apply::<2>(&this), Err(ViewError::Mismatch));
    /// assert_eq!(Term::default().view_apply::<2>(&this), Err(ViewError::Uninferred));
    /// ```
    ///
    /// When the symbol is applied to fewer arguments than expected, the argument list is padded
    /// with wildcards from the front:
    /// ```
    /// # use hugr_core::terms::{Term, TermKind, ViewError};
    /// # use hugr_model::v0::SymbolName;
    /// let symbol = SymbolName::new_static("symbol");
    /// let arg = Term::from(42u64);
    /// let too_few = Term::new(TermKind::Apply(&symbol, &[arg.clone()]));
    /// assert_eq!(too_few.view_apply(&symbol), Ok([Term::default(), arg]));
    /// ```
    ///
    /// The view fails to match when the symbol is applied to more arguments than expected:
    /// ```
    /// # use hugr_core::terms::{Term, TermKind, ViewError};
    /// # use hugr_model::v0::SymbolName;
    /// let symbol = SymbolName::new_static("symbol");
    /// let args = [Term::from(0u64), Term::from(1u64), Term::from(2u64)];
    /// let too_many = Term::new(TermKind::Apply(&symbol, &args));
    /// assert_eq!(too_many.view_apply::<2>(&symbol), Err(ViewError::Mismatch));
    /// ```
    pub fn view_apply<const N: usize>(&self, symbol: &SymbolName) -> Result<[Term; N], ViewError> {
        let (term_symbol, term_args) = match self.get() {
            TermKind::Wildcard => return Err(ViewError::Uninferred),
            TermKind::Apply(term_symbol, term_args) => (term_symbol, term_args),
            TermKind::Var(_) => return Err(ViewError::Variable),
            _ => return Err(ViewError::Mismatch),
        };

        if symbol != term_symbol {
            return Err(ViewError::Mismatch);
        }

        if term_args.len() > N {
            return Err(ViewError::Mismatch);
        }

        let args = std::array::from_fn(|i| {
            (i + term_args.len())
                .checked_sub(N)
                .map(|i| term_args[i].clone())
                .unwrap_or_default()
        });

        Ok(args)
    }
}

impl From<Literal> for Term {
    fn from(value: Literal) -> Self {
        Term::new(TermKind::Literal(&value))
    }
}

impl From<u64> for Term {
    fn from(value: u64) -> Self {
        Literal::Nat(value).into()
    }
}

impl From<f64> for Term {
    fn from(value: f64) -> Self {
        Literal::Float(OrderedFloat(value)).into()
    }
}

impl From<OrderedFloat<f64>> for Term {
    fn from(value: OrderedFloat<f64>) -> Self {
        Literal::Float(value).into()
    }
}

impl From<SmolStr> for Term {
    fn from(value: SmolStr) -> Self {
        Literal::Str(value).into()
    }
}

impl From<Term> for ast::Term {
    fn from(value: Term) -> Self {
        ast::Term::from(&value)
    }
}

impl From<&Term> for ast::Term {
    fn from(value: &Term) -> Self {
        match value.get() {
            TermKind::Wildcard => ast::Term::Wildcard,
            TermKind::Apply(symbol, args) => {
                let symbol = symbol.clone();
                let args = args.iter().map(ast::Term::from).collect();
                ast::Term::Apply(symbol, args)
            }
            TermKind::Literal(literal) => ast::Term::Literal(literal.clone()),
            TermKind::Var(var) => ast::Term::Var(var.name().clone()),
            TermKind::Tuple(_, items) => ast::Term::Tuple(
                items
                    .iter()
                    .map(|item| ast::SeqPart::Item(item.into()))
                    .collect(),
            ),
            TermKind::TupleConcat(_, lists) => ast::Term::Tuple(
                lists
                    .iter()
                    .map(|list| ast::SeqPart::Splice(list.into()))
                    .collect(),
            ),
            TermKind::List(_, _) | TermKind::ListConcat(_, _) => {
                ast::Term::List(ListIter::new(value).map(ast::SeqPart::from).collect())
            }
            TermKind::Func(_, _) => {
                // TODO: Do we want to display regions somehow?
                ast::Term::Wildcard
            }
        }
    }
}

/// The default [`Term`] is a wildcard term.
///
/// ```
/// # use hugr_core::terms::{Term, TermKind};
/// assert_eq!(Term::default().get(), TermKind::Wildcard);
/// ```
impl Default for Term {
    fn default() -> Self {
        static WILDCARD: Lazy<Term> = Lazy::new(|| Term::new_static(TermKind::Wildcard));
        WILDCARD.clone()
    }
}

/// A borrowed view into a [`Term`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TermKind<'a> {
    #[default]
    Wildcard,
    Apply(&'a SymbolName, &'a [Term]),
    Literal(&'a Literal),
    List(&'a Term, &'a [Term]),
    ListConcat(&'a Term, &'a [Term]),
    Tuple(&'a Term, &'a [Term]),
    TupleConcat(&'a Term, &'a [Term]),
    Var(&'a Var),
    Func(&'a Term, Node),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var {
    name: VarName,
    node: Node,
    index: u16,
}

impl Var {
    #[inline]
    pub fn name(&self) -> &VarName {
        &self.name
    }

    #[inline]
    pub fn node(&self) -> Node {
        self.node
    }

    #[inline]
    pub fn index(&self) -> u16 {
        self.index
    }
}

impl From<Var> for Term {
    fn from(value: Var) -> Self {
        Term::new(TermKind::Var(&value))
    }
}

/// Trait for view types that can pattern match terms.
pub trait View: Sized {
    /// Attempts to match a term as an instance of this type.
    fn view(term: &Term) -> Result<Self, ViewError>;
}

impl View for Term {
    fn view(term: &Term) -> Result<Self, ViewError> {
        Ok(term.clone())
    }
}

impl View for Literal {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.get() {
            TermKind::Literal(literal) => Ok(literal.clone()),
            TermKind::Wildcard => Err(ViewError::Uninferred),
            TermKind::Var(_) => Err(ViewError::Variable),
            _ => Err(ViewError::Mismatch),
        }
    }
}

impl View for u64 {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.view()? {
            Literal::Nat(value) => Ok(value),
            _ => Err(ViewError::Mismatch),
        }
    }
}

impl View for f64 {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.view()? {
            Literal::Float(value) => Ok(value.into_inner()),
            _ => Err(ViewError::Mismatch),
        }
    }
}

impl View for OrderedFloat<f64> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.view()? {
            Literal::Float(value) => Ok(value),
            _ => Err(ViewError::Mismatch),
        }
    }
}

impl View for SmolStr {
    fn view(term: &Term) -> Result<Self, ViewError> {
        match term.view()? {
            Literal::Str(string) => Ok(string),
            _ => Err(ViewError::Mismatch),
        }
    }
}

/// A [`Term`] does not fit the pattern expected by a [`View`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ViewError {
    /// The term does not match the pattern of the view.
    #[error("the term does not match the pattern of the view")]
    Mismatch,

    /// The view did not match the term because relevant parts of the term are
    /// uninferred. Filling in the uninferred parts of the term can potentially
    /// lead to a match.
    #[error("the term has uninferred parts that prevent a match")]
    Uninferred,

    /// The view did not match the term because relevant parts of the term are
    /// abstracted behind a variable. Substituting more concrete terms for the
    /// variables can potentially lead to a match.
    #[error("the term contains variables that prevent a match")]
    Variable,
}

/// Macro to create a view type for a term constructor.
#[macro_export]
macro_rules! term_view_ctr {
    (
        $name:expr;
        $(#[$attr:meta])*
        $vis:vis struct $ident:ident {
            $($(#[$field_meta:meta])* pub $field_name:ident: $field_type:ty,)*
        }
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone)]
        pub struct $ident {
            $($(#[$field_meta])* pub $field_name: $field_type,)*
        }

        impl $ident {
            /// The name of the term constructor for this type.
            pub const CTR_NAME: ::hugr_model::v0::SymbolName = ::hugr_model::v0::SymbolName::new_static($name);
        }

        impl $crate::terms::View for $ident {
            fn view(term: &$crate::terms::Term) -> Result<Self, $crate::terms::ViewError> {
                let [$($field_name),*] = term.view_apply(&Self::CTR_NAME)?;
                $(let $field_name = $field_name.view()?;)*
                Ok(Self { $($field_name),* })
            }
        }

        impl From<$ident> for $crate::terms::Term {
            fn from(value: $ident) -> Self {
                $crate::terms::Term::new(
                    $crate::terms::TermKind::Apply(
                        &$ident::CTR_NAME,
                        &[$(value.$field_name.into()),*],
                    )
                )
            }
        }
    };

    (
        $name:expr;
        $(#[$attr:meta])*
        $vis:vis struct $ident:ident;
    ) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy)]
        pub struct $ident;

        impl $ident {
            /// The name of the term constructor for this type.
            pub const CTR_NAME: ::hugr_model::v0::SymbolName = ::hugr_model::v0::SymbolName::new_static($name);
        }

        impl $crate::terms::View for $ident {
            fn view(term: &$crate::terms::Term) -> Result<Self, $crate::terms::ViewError> {
                let [] = term.view_apply(&Self::CTR_NAME)?;
                Ok(Self)
            }
        }

        impl From<$ident> for $crate::terms::Term {
            fn from(_: $ident) -> Self {
                static TERM: ::once_cell::sync::Lazy<$crate::terms::Term> =
                    ::once_cell::sync::Lazy::new(|| {
                        $crate::terms::Term::new(
                            $crate::terms::TermKind::Apply(&$ident::CTR_NAME, &[])
                        )
                    });
                TERM.clone()
            }
        }
    };
}
