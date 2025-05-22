pub use apply::Apply;
use hugr_model::v0::ast;
pub use hugr_model::v0::{Literal, SymbolName, VarName};
pub use list::List;
use servo_arc::Arc;
use std::fmt::Display;
pub use views::ViewError;
use views::{CoreBytes, CoreFloat, CoreNat, CoreStr};

mod apply;
mod list;
pub mod views;

/// A term in the language of static parameters.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Term {
    #[default]
    Wildcard,
    Literal(Literal),
    List(List),
    Tuple(Tuple),
    Apply(Apply),
    Var(Var),
    StaticType,
}

impl Term {
    /// Attempt to view this term as an application of a particular symbol with a given arity.
    ///
    /// See [`Apply::view`] for more details.
    pub fn view_apply<const N: usize>(&self, symbol: &SymbolName) -> Result<[Term; N], ViewError> {
        match self {
            Term::Wildcard => Err(ViewError::Uninferred),
            Term::Literal(_) => Err(ViewError::Mismatch),
            Term::List(_) => Err(ViewError::Mismatch),
            Term::Tuple(_) => Err(ViewError::Mismatch),
            Term::Apply(apply) => apply.view(symbol),
            Term::Var(_) => Err(ViewError::Variable),
            Term::StaticType => Err(ViewError::Mismatch),
        }
    }

    fn has_vars(&self) -> bool {
        match self {
            Term::Wildcard => false,
            Term::Literal(_) => false,
            Term::List(list) => todo!(),
            Term::Tuple(tuple) => todo!(),
            Term::Apply(apply) => apply.has_vars(),
            Term::Var(var) => true,
            Term::StaticType => false,
        }
    }

    pub fn substitute(mut self, terms: &[Term]) -> Self {
        match self {
            Term::List(list) => todo!(),
            Term::Tuple(tuple) => todo!(),
            Term::Apply(apply) => todo!(),
            Term::Var(var) => todo!(),
            this => this,
        }
    }
}

impl From<&Term> for ast::Term {
    fn from(value: &Term) -> Self {
        match value {
            Term::Wildcard => ast::Term::Wildcard,
            Term::Literal(literal) => ast::Term::Literal(literal.clone()),
            Term::List(list) => list.into(),
            Term::Tuple(tuple) => tuple.into(),
            Term::Apply(apply) => apply.into(),
            Term::Var(var) => var.into(),
            Term::StaticType => {
                ast::Term::Apply(SymbolName::new_static("core.static"), Default::default())
            }
        }
    }
}

macro_rules! impl_from {
    ($name:ident) => {
        impl From<$name> for Term {
            fn from(value: $name) -> Self {
                Self::$name(value)
            }
        }
    };
}

impl_from!(Literal);
impl_from!(List);
impl_from!(Tuple);
impl_from!(Apply);
impl_from!(Var);

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

/// Trait for objects that have a type.
pub trait Typed {
    fn type_(&self) -> impl Into<Term>;
}

impl Typed for Term {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> Term {
        match self {
            Term::Wildcard => Term::Wildcard,
            Term::Literal(literal) => match literal {
                Literal::Str(_) => CoreStr.into(),
                Literal::Nat(_) => CoreNat.into(),
                Literal::Bytes(_) => CoreBytes.into(),
                Literal::Float(_) => CoreFloat.into(),
            },
            Term::List(list) => list.type_().into(),
            Term::Tuple(tuple) => todo!(),
            Term::Apply(apply) => apply.type_().into(),
            Term::Var(var) => var.type_().into(),
            Term::StaticType => Term::StaticType,
        }
    }
}

impl From<&Term> for Term {
    fn from(value: &Term) -> Self {
        value.clone()
    }
}

/// Part of a [`List`] or [`Tuple`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqPart {
    Item(Term),
    Splice(Term),
}

impl From<&SeqPart> for ast::SeqPart {
    fn from(value: &SeqPart) -> Self {
        match value {
            SeqPart::Item(term) => ast::SeqPart::Item(term.into()),
            SeqPart::Splice(term) => ast::SeqPart::Splice(term.into()),
        }
    }
}

impl Display for SeqPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::SeqPart::from(self))
    }
}

/// Heterogeneous sequences of [`Term`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tuple(Arc<[SeqPart]>);

impl From<&Tuple> for ast::Term {
    fn from(value: &Tuple) -> Self {
        todo!()
    }
}

impl Display for Tuple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

/// Variable [`Term`]s.
///
/// ```
/// # use hugr_core::terms::{Term, Var, VarName};
/// let x = Var::new(VarName::new("x"), 0, Term::Wildcard);
/// assert_eq!(x.to_string(), "?x");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Var(Arc<VarInner>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct VarInner {
    name: VarName,
    index: u16,
    type_: Term,
}

impl Var {
    pub fn new(name: VarName, index: u16, type_: Term) -> Self {
        Self(Arc::new(VarInner { name, index, type_ }))
    }

    pub fn name(&self) -> &VarName {
        &self.0.name
    }

    pub fn index(&self) -> u16 {
        self.0.index
    }
}

impl From<&Var> for ast::Term {
    fn from(value: &Var) -> Self {
        Self::Var(value.name().clone())
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl Typed for Var {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> &Term {
        &self.0.type_
    }
}
