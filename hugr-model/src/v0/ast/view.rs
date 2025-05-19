use std::sync::Arc;

use super::{LinkName, Node, Operation, Param, Region, SeqPart, Symbol, SymbolName, Term, VarName};
use crate::v0::table::{self, NodeId, TermId, VarId, View};

impl<'a> View<'a, TermId> for Term {
    fn view(module: &'a table::Module<'a>, id: TermId) -> Option<Self> {
        let term = module.get_term(id)?;
        Some(match term {
            table::Term::Wildcard => Term::Wildcard,
            table::Term::Var(var) => Term::Var(module.view(*var)?),
            table::Term::Apply(symbol, terms) => {
                let symbol = module.view(*symbol)?;
                let terms = module.view(*terms)?;
                Term::Apply(symbol, terms)
            }
            table::Term::Func(region_id) => Term::Func(Arc::new(module.view(*region_id)?)),
            table::Term::List(list_parts) => Term::List(module.view(*list_parts)?),
            table::Term::Literal(literal) => Term::Literal(literal.clone()),
            table::Term::Tuple(tuple_parts) => Term::List(module.view(*tuple_parts)?),
        })
    }
}

impl<'a> View<'a, NodeId> for Node {
    fn view(module: &'a table::Module<'a>, id: NodeId) -> Option<Self> {
        let node = module.get_node(id)?;

        let operation = match node.operation {
            table::Operation::Invalid => Operation::Invalid,
            table::Operation::Dfg => Operation::Dfg,
            table::Operation::Cfg => Operation::Cfg,
            table::Operation::Block => Operation::Block,
            table::Operation::DefineFunc(symbol) => {
                Operation::DefineFunc(Box::new(module.view(*symbol)?))
            }
            table::Operation::DeclareFunc(symbol) => {
                Operation::DeclareFunc(Box::new(module.view(*symbol)?))
            }
            table::Operation::Custom(operation) => Operation::Custom(module.view(operation)?),
            table::Operation::DefineAlias(symbol, value) => {
                let symbol = Box::new(module.view(*symbol)?);
                let value = module.view(value)?;
                Operation::DefineAlias(symbol, value)
            }
            table::Operation::DeclareAlias(symbol) => {
                Operation::DeclareAlias(Box::new(module.view(*symbol)?))
            }
            table::Operation::DeclareConstructor(symbol) => {
                Operation::DeclareConstructor(Box::new(module.view(*symbol)?))
            }
            table::Operation::DeclareOperation(symbol) => {
                Operation::DeclareOperation(Box::new(module.view(*symbol)?))
            }
            table::Operation::TailLoop => Operation::TailLoop,
            table::Operation::Conditional => Operation::Conditional,
            table::Operation::Import { name } => Operation::Import(SymbolName::new(name)),
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

impl<'a> View<'a, table::LinkIndex> for LinkName {
    fn view(_module: &'a table::Module<'a>, index: table::LinkIndex) -> Option<Self> {
        Some(LinkName::new_index(index))
    }
}

impl<'a> View<'a, table::SeqPart> for SeqPart {
    fn view(module: &'a table::Module<'a>, part: table::SeqPart) -> Option<Self> {
        Some(match part {
            table::SeqPart::Item(term_id) => SeqPart::Item(module.view(term_id)?),
            table::SeqPart::Splice(term_id) => SeqPart::Splice(module.view(term_id)?),
        })
    }
}

impl<'a> View<'a, table::Symbol<'a>> for Symbol {
    fn view(module: &'a table::Module<'a>, id: table::Symbol<'a>) -> Option<Self> {
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

impl<'a> View<'a, table::Param<'a>> for Param {
    fn view(module: &'a table::Module<'a>, param: table::Param<'a>) -> Option<Self> {
        let name = VarName::new(param.name);
        let r#type = module.view(param.r#type)?;
        Some(Param { name, r#type })
    }
}

impl<'a> View<'a, table::RegionId> for Region {
    fn view(module: &'a table::Module<'a>, id: table::RegionId) -> Option<Self> {
        let region = module.get_region(id)?;
        let sources = module.view(region.sources)?;
        let targets = module.view(region.targets)?;
        let meta = module.view(region.meta)?;
        let children = module.view(region.children)?;
        let signature = module.view(region.signature)?;

        Some(Region {
            kind: region.kind,
            sources,
            targets,
            children,
            meta,
            signature,
        })
    }
}

impl<'a> View<'a, VarId> for VarName {
    fn view(module: &'a table::Module<'a>, id: VarId) -> Option<Self> {
        let node = module.get_node(id.0)?;

        let symbol = match node.operation {
            table::Operation::DefineFunc(symbol) => symbol,
            table::Operation::DeclareFunc(symbol) => symbol,
            table::Operation::DefineAlias(symbol, _) => symbol,
            table::Operation::DeclareAlias(symbol) => symbol,
            table::Operation::DeclareConstructor(symbol) => symbol,
            table::Operation::DeclareOperation(symbol) => symbol,
            _ => return None,
        };

        let param = &symbol.params[id.1 as usize];
        Some(Self(param.name.into()))
    }
}

impl<'a> View<'a, NodeId> for SymbolName {
    fn view(module: &'a table::Module<'a>, id: NodeId) -> Option<Self> {
        let node = module.get_node(id)?;
        let name = node.operation.symbol()?;
        Some(Self(name.into()))
    }
}
