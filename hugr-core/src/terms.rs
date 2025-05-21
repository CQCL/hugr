use hugr_model::v0::ast;
use hugr_model::v0::{Literal, SymbolName};
use std::fmt::Display;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Literal(Literal),
    List(List),
    Tuple(Tuple),
    Apply(Apply),
}

impl From<&Term> for ast::Term {
    fn from(value: &Term) -> Self {
        match value {
            Term::Literal(literal) => ast::Term::Literal(literal.clone()),
            Term::List(list) => list.into(),
            Term::Tuple(tuple) => tuple.into(),
            Term::Apply(apply) => apply.into(),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct List(Arc<[SeqPart]>);

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
