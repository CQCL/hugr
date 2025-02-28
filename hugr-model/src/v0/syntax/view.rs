use std::sync::Arc;

use super::{LinkName, Node, Region, SeqPart, SymbolName, Term, VarName};
use crate::v0::view::View;
use crate::v0::{self as model, NodeId, RegionId, ScopeClosure, TermId, VarId};

impl<'a> View<'a> for Term {
    type Id = TermId;

    fn view(module: &'a model::Module<'a>, id: &Self::Id) -> Option<Self> {
        let term = module.get_term(*id)?;
        Some(match term {
            model::Term::Wildcard => Term::Wildcard,
            model::Term::Var(var) => Term::Var(module.view(var)?),
            model::Term::Apply(symbol, terms) => {
                let symbol = module.view(symbol)?;
                let terms = terms
                    .iter()
                    .map(|t| module.view(t))
                    .collect::<Option<_>>()?;
                Term::Apply(symbol, terms)
            }
            model::Term::ExtSet(_) => Term::ExtSet,
            model::Term::ConstFunc(region_id) => {
                let region = module.view(region_id)?;
                Term::Func(Arc::new(region))
            }
            model::Term::List(list_parts) => {
                let list_parts = list_parts
                    .iter()
                    .map(|part| match part {
                        model::ListPart::Item(term) => Some(SeqPart::Item(module.view(term)?)),
                        model::ListPart::Splice(term) => Some(SeqPart::Splice(module.view(term)?)),
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
                        model::TuplePart::Item(term) => Some(SeqPart::Item(module.view(term)?)),
                        model::TuplePart::Splice(term) => Some(SeqPart::Splice(module.view(term)?)),
                    })
                    .collect::<Option<_>>()?;
                Term::List(list_parts)
            }
        })
    }
}

impl<'a> View<'a> for Node {
    type Id = NodeId;

    fn view(module: &'a model::Module<'a>, id: &'a Self::Id) -> Option<Self> {
        let node = module.get_node(*id)?;

        let meta = node
            .meta
            .iter()
            .map(|t| module.view(t))
            .collect::<Option<_>>()?;

        let signature = match node.signature {
            Some(signature) => Some(module.view(&signature)?),
            None => None,
        };

        let inputs = node
            .inputs
            .iter()
            .copied()
            .map(LinkName::new_index)
            .collect();

        let outputs = node
            .inputs
            .iter()
            .copied()
            .map(LinkName::new_index)
            .collect();

        let regions = node
            .regions
            .iter()
            .map(|r| module.view(r))
            .collect::<Option<_>>()?;

        Some(Node {
            operation: todo!(),
            inputs,
            outputs,
            regions,
            meta,
            signature,
        })
    }
}

impl<'a> View<'a> for Region {
    type Id = RegionId;

    fn view(module: &'a model::Module<'a>, id: &'a Self::Id) -> Option<Self> {
        let region = module.get_region(*id)?;

        let sources = region
            .sources
            .iter()
            .copied()
            .map(LinkName::new_index)
            .collect();

        let targets = region
            .targets
            .iter()
            .copied()
            .map(LinkName::new_index)
            .collect();

        let meta = region
            .meta
            .iter()
            .map(|t| module.view(t))
            .collect::<Option<_>>()?;

        let children = region
            .children
            .iter()
            .map(|id| module.view(id))
            .collect::<Option<_>>()?;

        let signature = match region.signature {
            Some(signature) => Some(module.view(&signature)?),
            None => None,
        };

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

impl<'a> View<'a> for VarName {
    type Id = VarId;

    fn view(module: &'a model::Module<'a>, id: &Self::Id) -> Option<Self> {
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

impl<'a> View<'a> for SymbolName {
    type Id = NodeId;

    fn view(module: &'a model::Module<'a>, id: &Self::Id) -> Option<Self> {
        let node = module.get_node(*id)?;
        let name = node.operation.symbol()?;
        Some(Self(name.into()))
    }
}
