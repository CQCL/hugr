use std::sync::Arc;

use super::{LinkName, Node, Operation, Param, Region, SeqPart, Symbol, SymbolName, Term, VarName};
use crate::v0::view::View;
use crate::v0::{self as model, NodeId, ScopeClosure, TermId, VarId};

impl<'a> View<'a, TermId> for Term {
    fn view(module: &'a model::Module<'a>, id: TermId) -> Option<Self> {
        let term = module.get_term(id)?;
        Some(match term {
            model::Term::Wildcard => Term::Wildcard,
            model::Term::Var(var) => Term::Var(module.view(*var)?),
            model::Term::Apply(symbol, terms) => {
                let symbol = module.view(*symbol)?;
                let terms = module.view(*terms)?;
                Term::Apply(symbol, terms)
            }
            model::Term::ExtSet(_) => Term::ExtSet,
            model::Term::ConstFunc(region_id) => Term::Func(Arc::new(module.view(*region_id)?)),
            model::Term::List(list_parts) => Term::List(module.view(*list_parts)?),
            model::Term::Str(val) => Term::Str((*val).into()),
            model::Term::Nat(val) => Term::Nat(*val),
            model::Term::Bytes(bytes) => Term::Bytes((*bytes).into()),
            model::Term::Float(val) => Term::Float(*val),
            model::Term::Tuple(tuple_parts) => Term::List(module.view(*tuple_parts)?),
        })
    }
}

impl<'a> View<'a, NodeId> for Node {
    fn view(module: &'a model::Module<'a>, id: NodeId) -> Option<Self> {
        let node = module.get_node(id)?;

        let operation = match node.operation {
            model::Operation::Invalid => Operation::Invalid,
            model::Operation::Dfg => Operation::Dfg,
            model::Operation::Cfg => Operation::Cfg,
            model::Operation::Block => Operation::Block,
            model::Operation::DefineFunc(symbol) => {
                Operation::DefineFunc(Arc::new(module.view(*symbol)?))
            }
            model::Operation::DeclareFunc(symbol) => {
                Operation::DeclareFunc(Arc::new(module.view(*symbol)?))
            }
            model::Operation::Custom(node_id) => {
                let symbol = module.view(node_id)?;
                let params = module.view(node.params)?;
                Operation::Custom(Term::Apply(symbol, params))
            }
            model::Operation::DefineAlias(symbol) => {
                let [value] = node.params.try_into().ok()?;
                let value = module.view(value)?;
                Operation::DefineAlias(Arc::new(module.view(*symbol)?), value)
            }
            model::Operation::DeclareAlias(symbol) => {
                Operation::DeclareAlias(Arc::new(module.view(*symbol)?))
            }
            model::Operation::DeclareConstructor(symbol) => {
                Operation::DeclareConstructor(Arc::new(module.view(*symbol)?))
            }
            model::Operation::DeclareOperation(symbol) => {
                Operation::DeclareOperation(Arc::new(module.view(*symbol)?))
            }
            model::Operation::TailLoop => Operation::TailLoop,
            model::Operation::Conditional => Operation::Conditional,
            model::Operation::Import { name } => Operation::Import(SymbolName::new(name)),
        };

        let meta = module.view(node.meta)?;
        let signature = module.view(node.signature)?;
        let inputs = module.view(node.inputs)?;
        let outputs = module.view(node.outputs)?;
        let regions = module.view(node.regions)?;

        Some(Node {
            operation,
            inputs,
            outputs,
            regions,
            meta,
            signature,
        })
    }
}

impl<'a> View<'a, model::LinkIndex> for LinkName {
    fn view(_module: &'a model::Module<'a>, index: model::LinkIndex) -> Option<Self> {
        Some(LinkName::new_index(index))
    }
}

impl<'a> View<'a, model::ListPart> for SeqPart {
    fn view(module: &'a model::Module<'a>, part: model::ListPart) -> Option<Self> {
        Some(match part {
            model::ListPart::Item(term_id) => SeqPart::Item(module.view(term_id)?),
            model::ListPart::Splice(term_id) => SeqPart::Splice(module.view(term_id)?),
        })
    }
}

impl<'a> View<'a, model::TuplePart> for SeqPart {
    fn view(module: &'a model::Module<'a>, part: model::TuplePart) -> Option<Self> {
        Some(match part {
            model::TuplePart::Item(term_id) => SeqPart::Item(module.view(term_id)?),
            model::TuplePart::Splice(term_id) => SeqPart::Splice(module.view(term_id)?),
        })
    }
}

impl<'a> View<'a, model::Symbol<'a>> for Symbol {
    fn view(module: &'a model::Module<'a>, id: model::Symbol<'a>) -> Option<Self> {
        let name = SymbolName::new(id.name);
        let params = module.view(id.params)?;
        let constraints = module.view(id.constraints)?;
        let signature = module.view(id.signature)?;
        Some(Symbol {
            name,
            params,
            constraints,
            signature,
        })
    }
}

impl<'a> View<'a, model::Param<'a>> for Param {
    fn view(module: &'a model::Module<'a>, param: model::Param<'a>) -> Option<Self> {
        let name = VarName::new(param.name);
        let r#type = module.view(param.r#type)?;
        Some(Param { name, r#type })
    }
}

impl<'a> View<'a, model::RegionId> for Region {
    fn view(module: &'a model::Module<'a>, id: model::RegionId) -> Option<Self> {
        let region = module.get_region(id)?;
        let sources = module.view(region.sources)?;
        let targets = module.view(region.targets)?;
        let meta = module.view(region.meta)?;
        let children = module.view(region.children)?;
        let signature = module.view(region.signature)?;

        let scope = match region.scope {
            Some(_) => ScopeClosure::Closed,
            None => ScopeClosure::Open,
        };

        Some(Region {
            kind: region.kind,
            sources,
            targets,
            children,
            meta,
            signature,
            scope,
        })
    }
}

impl<'a> View<'a, VarId> for VarName {
    fn view(module: &'a model::Module<'a>, id: VarId) -> Option<Self> {
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

impl<'a> View<'a, NodeId> for SymbolName {
    fn view(module: &'a model::Module<'a>, id: NodeId) -> Option<Self> {
        let node = module.get_node(id)?;
        let name = node.operation.symbol()?;
        Some(Self(name.into()))
    }
}
