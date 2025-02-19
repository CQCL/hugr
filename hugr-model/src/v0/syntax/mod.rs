use ordered_float::OrderedFloat;
use smol_str::SmolStr;

use super::{view::View, NodeId, TermId, VarId};
use crate::v0 as model;

mod print;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub inputs: Vec<Link>,
    pub outputs: Vec<Link>,
    pub params: Vec<Term>,
    pub regions: Vec<Region>,
    pub meta: Vec<Term>,
    pub signature: Vec<Term>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Wildcard,
    Var(Var),
    Apply(Symbol, Vec<Term>),
    List(Vec<ListPart>),
    Str(SmolStr),
    Nat(u64),
    Bytes(Vec<u8>),
    Float(OrderedFloat<f64>),
    Tuple(Vec<TuplePart>),
}

impl<'a> View<'a> for Term {
    type Id = TermId;

    fn view(module: &'a super::Module<'a>, id: Self::Id) -> Option<Self> {
        let term = module.get_term(id)?;
        match term {
            model::Term::Wildcard => Some(Term::Wildcard),
            model::Term::Var(var) => Some(Term::Var(module.view(*var)?)),
            model::Term::Apply(symbol, terms) => {
                let symbol = module.view(*symbol)?;
                let terms = terms
                    .iter()
                    .map(|t| module.view(*t))
                    .collect::<Option<_>>()?;
                Some(Term::Apply(symbol, terms))
            }
            model::Term::ExtSet(ext_set_parts) => todo!(),
            model::Term::ConstFunc(region_id) => todo!(),
            model::Term::List(list_parts) => todo!(),
            model::Term::Str(_) => todo!(),
            model::Term::Nat(_) => todo!(),
            model::Term::Bytes(items) => todo!(),
            model::Term::Float(ordered_float) => todo!(),
            model::Term::Tuple(tuple_parts) => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(SmolStr);

impl<'a> View<'a> for Var {
    type Id = VarId;

    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(SmolStr);

impl<'a> View<'a> for Symbol {
    type Id = NodeId;

    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Link(SmolStr);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListPart {
    Item(Term),
    Spread(Term),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuplePart {
    Item(Term),
    Spread(Term),
}
