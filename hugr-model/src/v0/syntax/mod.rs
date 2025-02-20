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
    ExtSet,
}

impl<'a> View<'a> for Term {
    type Id = TermId;

    fn view(module: &'a super::Module<'a>, id: Self::Id) -> Option<Self> {
        let term = module.get_term(id)?;
        Some(match term {
            model::Term::Wildcard => Term::Wildcard,
            model::Term::Var(var) => Term::Var(module.view(*var)?),
            model::Term::Apply(symbol, terms) => {
                let symbol = module.view(*symbol)?;
                let terms = terms
                    .iter()
                    .map(|t| module.view(*t))
                    .collect::<Option<_>>()?;
                Term::Apply(symbol, terms)
            }
            model::Term::ExtSet(_) => Term::ExtSet,
            model::Term::ConstFunc(region_id) => todo!(),
            model::Term::List(list_parts) => {
                let list_parts = list_parts
                    .iter()
                    .map(|part| match part {
                        model::ListPart::Item(term) => Some(ListPart::Item(module.view(*term)?)),
                        model::ListPart::Splice(term) => {
                            Some(ListPart::Splice(module.view(*term)?))
                        }
                    })
                    .collect::<Option<_>>()?;
                Term::List(list_parts)
            }
            model::Term::Str(val) => Term::Str((*val).into()),
            model::Term::Nat(val) => Term::Nat(*val),
            model::Term::Bytes(bytes) => Term::Bytes((*bytes).into()),
            model::Term::Float(val) => Term::Float(*val),
            model::Term::Tuple(tuple_parts) => {
                let list_parts = tuple_parts
                    .iter()
                    .map(|part| match part {
                        model::TuplePart::Item(term) => Some(ListPart::Item(module.view(*term)?)),
                        model::TuplePart::Splice(term) => {
                            Some(ListPart::Splice(module.view(*term)?))
                        }
                    })
                    .collect::<Option<_>>()?;
                Term::List(list_parts)
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(SmolStr);

impl<'a> View<'a> for Var {
    type Id = VarId;

    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let node = module.get_node(id.0)?;

        let symbol = match node.operation {
            model::Operation::DefineFunc(symbol) => symbol,
            model::Operation::DeclareFunc(symbol) => symbol,
            model::Operation::DefineAlias(symbol) => symbol,
            model::Operation::DeclareAlias(symbol) => symbol,
            model::Operation::DeclareConstructor(symbol) => symbol,
            model::Operation::DeclareOperation(symbol) => symbol,
            _ => return None,
        };

        let param = &symbol.params[id.1 as usize];
        Some(Self(param.name.into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(SmolStr);

impl<'a> View<'a> for Symbol {
    type Id = NodeId;

    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let node = module.get_node(id)?;
        let name = node.operation.symbol()?;
        Some(Self(name.into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Link(SmolStr);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListPart {
    Item(Term),
    Splice(Term),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuplePart {
    Item(Term),
    Splice(Term),
}
