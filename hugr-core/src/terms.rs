use hugr_model::v0::{Literal, SymbolName};
use hugr_model::v0::{VarName, ast};
use std::fmt::Display;
use std::sync::Arc;

mod views;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Term {
    #[default]
    Wildcard,
    Literal(Literal),
    List(List),
    Tuple(Tuple),
    Apply(Apply),
    Var(Var),
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
        }
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl From<Literal> for Term {
    fn from(value: Literal) -> Self {
        Self::Literal(value)
    }
}

impl From<List> for Term {
    fn from(value: List) -> Self {
        Self::List(value)
    }
}

impl From<Tuple> for Term {
    fn from(value: Tuple) -> Self {
        Self::Tuple(value)
    }
}

impl From<Apply> for Term {
    fn from(value: Apply) -> Self {
        Self::Apply(value)
    }
}

impl From<Var> for Term {
    fn from(value: Var) -> Self {
        Self::Var(value)
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
    type Error = ();

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            Term::List(list) => Ok(list),
            Term::Var(var) => Ok(List::new([SeqPart::Splice(Term::Var(var))])),
            _ => Err(()),
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
pub struct Apply {
    name: SymbolName,
    args: Arc<[Term]>,
}

impl Apply {
    pub fn new(name: SymbolName, args: impl IntoIterator<Item = Term>) -> Self {
        Self {
            name,
            args: args.into_iter().collect(),
        }
    }

    pub fn name(&self) -> &SymbolName {
        &self.name
    }

    pub fn args(&self) -> &[Term] {
        &self.args
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Var {
    name: VarName,
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
