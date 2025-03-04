use bumpalo::{collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;
use thiserror::Error;

use super::{LinkName, Node, Region, SymbolName, Term, VarName};
use crate::v0::{self as model, RegionId, TermId, VarId};
use crate::v0::{
    scope::{LinkTable, SymbolTable, VarTable},
    NodeId,
};

struct Context<'a> {
    module: model::Module<'a>,
    bump: &'a Bump,
    vars: VarTable<'a>,
    links: LinkTable<LinkName>,
    symbols: SymbolTable<'a>,
    imports: FxHashMap<SymbolName, NodeId>,
}

impl<'a> Context<'a> {
    fn build_terms(&mut self, terms: &'a [Term]) -> BuildResult<&'a [TermId]> {
        let mut ids = BumpVec::with_capacity_in(terms.len(), self.bump);

        for term in terms {
            ids.push(self.build_term(term)?);
        }

        Ok(ids.into_bump_slice())
    }

    fn build_term(&mut self, term: &'a Term) -> BuildResult<TermId> {
        let term = match term {
            Term::Wildcard => model::Term::Wildcard,
            Term::Var(var_name) => model::Term::Var(self.resolve_var(var_name)?),
            Term::Apply(symbol_name, terms) => {
                let symbol_id = self.resolve_symbol(symbol_name)?;
                let terms = self.build_terms(terms)?;
                model::Term::Apply(symbol_id, terms)
            }
            Term::List(seq_parts) => todo!(),
            Term::Str(smol_str) => todo!(),
            Term::Nat(_) => todo!(),
            Term::Bytes(items) => todo!(),
            Term::Float(ordered_float) => todo!(),
            Term::Tuple(seq_parts) => todo!(),
            Term::Func(region) => todo!(),
            Term::ExtSet => todo!(),
        };

        Ok(self.module.insert_term(term))
    }

    fn build_nodes(&mut self, nodes: &'a [Node]) -> BuildResult<&'a [NodeId]> {
        let ids = {
            let mut ids = BumpVec::with_capacity_in(nodes.len(), self.bump);

            for node in nodes {
                let symbol_name = match &node.operation {
                    super::Operation::DefineFunc(symbol) => Some(&symbol.name),
                    super::Operation::DeclareFunc(symbol) => Some(&symbol.name),
                    super::Operation::DefineAlias(symbol, _) => Some(&symbol.name),
                    super::Operation::DeclareAlias(symbol) => Some(&symbol.name),
                    super::Operation::DeclareConstructor(symbol) => Some(&symbol.name),
                    super::Operation::DeclareOperation(symbol) => Some(&symbol.name),
                    super::Operation::Import(symbol_name) => Some(symbol_name),
                    _ => None,
                };

                let node_id = self.module.insert_node(model::Node::default());

                if let Some(symbol_name) = symbol_name {
                    let bump_symbol_name = self.bump.alloc_str(symbol_name.as_ref());
                    self.symbols
                        .insert(bump_symbol_name, node_id)
                        .map_err(|_| BuildError::DuplicateSymbol(symbol_name.clone()))?;
                }

                ids.push(node_id);
            }

            ids.into_bump_slice()
        };

        for (id, node) in ids.iter().zip(nodes) {
            self.build_node(*id, node)?;
        }

        Ok(ids)
    }

    fn build_node(&mut self, node_id: NodeId, node: &'a Node) -> BuildResult<()> {
        todo!()
    }

    fn build_regions(&mut self, regions: &'a [Region]) -> BuildResult<&'a [RegionId]> {
        let mut ids = BumpVec::with_capacity_in(regions.len(), self.bump);

        for region in regions {
            ids.push(self.build_region(region)?);
        }

        Ok(ids.into_bump_slice())
    }

    fn build_region(&mut self, region: &'a Region) -> BuildResult<RegionId> {
        todo!()
    }

    fn resolve_var(&self, var_name: &'a VarName) -> BuildResult<VarId> {
        self.vars
            .resolve(var_name.as_ref())
            .map_err(|_| BuildError::Var(var_name.clone()))
    }

    fn resolve_symbol(&self, symbol_name: &'a SymbolName) -> BuildResult<NodeId> {
        self.symbols
            .resolve(symbol_name.as_ref())
            .map_err(|_| BuildError::Symbol(symbol_name.clone()))
    }
}

#[derive(Debug, Clone, Error)]
pub enum BuildError {
    #[error("unknown var: {0}")]
    Var(VarName),
    #[error("unknown symbol: {0}")]
    Symbol(SymbolName),
    #[error("duplicate symbol: {0}")]
    DuplicateSymbol(SymbolName),
}

type BuildResult<T> = Result<T, BuildError>;
