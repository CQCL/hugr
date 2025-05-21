use hugr_model::v0::ast;
pub use hugr_model::v0::{Literal, SymbolName, VarName};
use std::fmt::Display;
use triomphe::{Arc, ThinArc};
pub use views::ViewError;
use views::{CoreBytes, CoreFloat, CoreList, CoreNat, CoreStr};

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

/// Homogeneous sequences of [`Term`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct List(ThinArc<ListHeader, SeqPart>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ListHeader {
    list_type: CoreList,
}

impl List {
    pub fn new<I>(parts: I, item_type: Term) -> Self
    where
        I: IntoIterator<Item = SeqPart>,
        I::IntoIter: ExactSizeIterator,
    {
        Self(ThinArc::from_header_and_iter(
            ListHeader {
                list_type: CoreList { item_type },
            },
            parts.into_iter(),
        ))
    }

    pub fn parts(&self) -> &[SeqPart] {
        &self.0.slice
    }

    pub fn item_type(&self) -> &Term {
        &self.type_().item_type
    }
}

impl Typed for List {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> &CoreList {
        &self.0.header.header.list_type
    }
}

impl From<&List> for ast::Term {
    fn from(value: &List) -> Self {
        ast::Term::List(value.parts().iter().map(ast::SeqPart::from).collect())
    }
}

impl Display for List {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl TryFrom<Term> for List {
    type Error = ViewError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            Term::List(list) => Ok(list),
            Term::Var(var) => {
                let type_ = var.type_().clone();
                Ok(List::new([SeqPart::Splice(Term::Var(var))], type_))
            }
            _ => Err(ViewError::Mismatch),
        }
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

/// [`Term`]s obtained by applying a symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Apply(ThinArc<ApplyHeader, Term>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ApplyHeader {
    name: SymbolName,
    type_: Term,
}

impl Apply {
    pub fn new<A>(name: SymbolName, args: A, type_: Term) -> Self
    where
        A: IntoIterator<Item = Term>,
        A::IntoIter: ExactSizeIterator,
    {
        Self(ThinArc::from_header_and_iter(
            ApplyHeader { name, type_ },
            args.into_iter(),
        ))
    }

    /// The name of the applied symbol.
    pub fn name(&self) -> &SymbolName {
        &self.0.header.header.name
    }

    /// The arguments to the symbol.
    pub fn args(&self) -> &[Term] {
        &self.0.slice
    }

    /// Attempt to view this term as an application of a particular symbol with a given arity.
    ///
    /// In the case that there are fewer than `N` arguments we still return a match. The returned
    /// argument sequence is padded from the front with [`Term::Wildcard`], indicating that the
    /// omitted arguments are intended to be implicit.
    ///
    /// # Errors
    ///
    /// - [`ViewError::Mismatch`] when the symbol name does not match.
    /// - [`ViewError::Invalid`] when the symbol is applied to too many arguments.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hugr_core::terms::{Term, Apply, SymbolName, ViewError, Literal};
    /// let this = SymbolName::new("this.name");
    /// let that = SymbolName::new("that.name");
    /// let arg1 = Term::Literal(Literal::Nat(1));
    /// let arg2 = Term::Literal(Literal::Nat(2));
    ///
    /// let apply = Apply::new(this.clone(), [arg1.clone(), arg2.clone()], Term::Wildcard);
    ///
    /// assert_eq!(apply.view(&this), Ok([arg1.clone(), arg2.clone()]));
    /// assert_eq!(apply.view(&this), Ok([Term::Wildcard, arg1.clone(), arg2.clone()]));
    /// assert!(matches!(apply.view::<1>(&this), Err(ViewError::Invalid(_))));
    /// assert_eq!(apply.view::<1>(&that), Err(ViewError::Mismatch));
    /// ```
    pub fn view<const N: usize>(&self, symbol: &SymbolName) -> Result<[Term; N], ViewError> {
        if self.name() != symbol {
            return Err(ViewError::Mismatch);
        }

        if self.args().len() > N {
            return Err(ViewError::Invalid(format!(
                "`{}` expects at most {} arguments",
                symbol, N
            )));
        }

        let result = std::array::from_fn(|i| {
            (i + self.args().len())
                .checked_sub(N)
                .map(|i| self.args()[i].clone())
                .unwrap_or_default()
        });

        Ok(result)
    }
}

impl Typed for Apply {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> &Term {
        &self.0.header.header.type_
    }
}

impl From<&Apply> for ast::Term {
    fn from(value: &Apply) -> Self {
        let name = value.name().clone();
        let args = value.args().iter().map(ast::Term::from).collect();
        ast::Term::Apply(name, args)
    }
}

impl Display for Apply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl TryFrom<Term> for Apply {
    type Error = ViewError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            Term::Apply(apply) => Ok(apply),
            Term::Wildcard => Err(ViewError::Uninferred),
            Term::Literal(_) => Err(ViewError::Mismatch),
            Term::List(_) => Err(ViewError::Mismatch),
            Term::Tuple(_) => Err(ViewError::Mismatch),
            Term::Var(_) => Err(ViewError::Variable),
            Term::StaticType => Err(ViewError::Mismatch),
        }
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
