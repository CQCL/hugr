use std::sync::Arc;

use ordered_float::OrderedFloat;
use smol_str::SmolStr;

use super::{view::View, NodeId, RegionKind, ScopeClosure, TermId, VarId};
use crate::v0 as model;

mod parse;
mod print;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub operation: Operation,
    pub inputs: Box<[LinkName]>,
    pub outputs: Box<[LinkName]>,
    pub params: Box<[Term]>,
    pub regions: Box<[Region]>,
    pub meta: Box<[MetaItem]>,
    pub signature: Option<Signature>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Invalid,
    Dfg,
    Cfg,
    Block,
    DefineFunc(Arc<Symbol>),
    DeclareFunc(Arc<Symbol>),
    Custom(SymbolName),
    DefineAlias(Arc<Symbol>),
    DeclareAlias(Arc<Symbol>),
    TailLoop,
    Conditional,
    DeclareConstructor(Arc<Symbol>),
    DeclareOperation(Arc<Symbol>),
    Import(SymbolName),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: SymbolName,
    pub params: Arc<[Param]>,
    pub constraints: Arc<[Constraint]>,
    pub signature: Term,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: VarName,
    pub r#type: Arc<Term>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint(pub Term);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    pub sources: Box<[LinkName]>,
    pub targets: Box<[LinkName]>,
    pub children: Box<[Node]>,
    pub meta: Box<[MetaItem]>,
    pub signature: Option<Signature>,
    pub scope: ScopeClosure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaItem(pub Term);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature(pub Term);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    Wildcard,
    Var(VarName),
    Apply(SymbolName, Arc<[Term]>),
    List(Arc<[ListPart]>),
    Str(SmolStr),
    Nat(u64),
    Bytes(Arc<[u8]>),
    Float(OrderedFloat<f64>),
    Tuple(Arc<[TuplePart]>),
    Func(Arc<Region>),
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
pub struct VarName(SmolStr);

impl<'a> View<'a> for VarName {
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
pub struct SymbolName(SmolStr);

impl<'a> View<'a> for SymbolName {
    type Id = NodeId;

    fn view(module: &'a model::Module<'a>, id: Self::Id) -> Option<Self> {
        let node = module.get_node(id)?;
        let name = node.operation.symbol()?;
        Some(Self(name.into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkName(SmolStr);

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
