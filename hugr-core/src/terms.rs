use hugr_model::v0::{Literal, SymbolName};
use hugr_model::v0::{VarName, ast};
use std::fmt::Display;
use std::sync::Arc;
use triomphe::ThinArc;
use views::{CoreBytes, CoreFloat, CoreNat, CoreStr, ViewError};

pub mod views;

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
    pub fn type_(&self) -> Term {
        match self {
            Term::Wildcard => Term::Wildcard,
            Term::Literal(literal) => match literal {
                Literal::Str(_) => CoreStr.into(),
                Literal::Nat(_) => CoreNat.into(),
                Literal::Bytes(_) => CoreBytes.into(),
                Literal::Float(_) => CoreFloat.into(),
            },
            Term::List(list) => todo!(),
            Term::Tuple(tuple) => todo!(),
            Term::Apply(apply) => apply.type_(),
            Term::Var(var) => todo!(),
            Term::StaticType => Term::StaticType,
        }
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqPart {
    Item(Term),
    Splice(Term),
}

impl From<&SeqPart> for ast::SeqPart {
    fn from(value: &SeqPart) -> Self {
        todo!()
    }
}

impl Display for SeqPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::SeqPart::from(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct List(Arc<[SeqPart]>);

impl List {
    pub fn new(parts: impl IntoIterator<Item = SeqPart>) -> Self {
        Self(parts.into_iter().collect())
    }
}

impl From<&List> for ast::Term {
    fn from(value: &List) -> Self {
        todo!()
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
            Term::Var(var) => Ok(List::new([SeqPart::Splice(Term::Var(var))])),
            _ => Err(ViewError::Mismatch),
        }
    }
}

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

    pub fn name(&self) -> &SymbolName {
        &self.0.header.header.name
    }

    pub fn args(&self) -> &[Term] {
        &self.0.slice
    }

    pub fn type_(&self) -> Term {
        self.0.header.header.type_.clone()
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Var {
    name: VarName,
    index: u16,
    type_: Arc<Term>,
}

impl From<&Var> for ast::Term {
    fn from(value: &Var) -> Self {
        Self::Var(value.name.clone())
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl Var {
    pub fn type_(&self) -> Term {
        self.type_.as_ref().clone()
    }
}
