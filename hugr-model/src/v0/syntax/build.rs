use bumpalo::{collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;
use thiserror::Error;

use super::{LinkName, Node, Param, Region, Symbol, SymbolName, Term, VarName};
use crate::v0::syntax::Operation;
use crate::v0::{self as model, LinkId, LinkIndex, RegionId, TermId, VarId};
use crate::v0::{
    scope::{LinkTable, SymbolTable, VarTable},
    NodeId,
};

struct Context<'a> {
    module: model::Module<'a>,
    bump: &'a Bump,
    vars: VarTable<'a>,
    links: LinkTable<&'a str>,
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
        let inputs = self.resolve_links(&node.inputs)?;
        let outputs = self.resolve_links(&node.outputs)?;
        let meta = self.build_terms(&node.meta)?;
        let regions = self.build_regions(&node.regions)?;

        let signature = match &node.signature {
            Some(signature) => Some(self.build_term(&signature)?),
            None => None,
        };

        let operation = match &node.operation {
            Operation::Invalid => model::Operation::Invalid,
            Operation::Dfg => model::Operation::Dfg,
            Operation::Cfg => model::Operation::Cfg,
            Operation::Block => model::Operation::Block,
            Operation::TailLoop => model::Operation::TailLoop,
            Operation::Conditional => model::Operation::Conditional,
            Operation::DefineFunc(symbol) => {
                let symbol = self.build_symbol(&symbol)?;
                model::Operation::DefineFunc(symbol)
            }
            Operation::DeclareFunc(symbol) => {
                let symbol = self.build_symbol(&symbol)?;
                model::Operation::DeclareFunc(symbol)
            }
            Operation::DefineAlias(symbol, term) => todo!(),
            Operation::DeclareAlias(symbol) => {
                let symbol = self.build_symbol(&symbol)?;
                model::Operation::DeclareAlias(symbol)
            }
            Operation::DeclareConstructor(symbol) => {
                let symbol = self.build_symbol(&symbol)?;
                model::Operation::DeclareConstructor(symbol)
            }
            Operation::DeclareOperation(symbol) => {
                let symbol = self.build_symbol(&symbol)?;
                model::Operation::DeclareOperation(symbol)
            }
            Operation::Import(symbol_name) => model::Operation::Import {
                name: symbol_name.as_ref(),
            },
            Operation::Custom(term) => todo!(),
        };

        self.module.nodes[node_id.index()] = model::Node {
            operation,
            inputs,
            outputs,
            params: todo!(),
            regions,
            meta,
            signature,
        };

        Ok(())
    }

    fn resolve_links(&mut self, links: &'a [LinkName]) -> BuildResult<&'a [LinkIndex]> {
        let mut indices = BumpVec::with_capacity_in(links.len(), self.bump);

        for link in links {
            indices.push(self.resolve_link(link)?);
        }

        Ok(indices.into_bump_slice())
    }

    fn resolve_link(&mut self, link: &'a LinkName) -> BuildResult<LinkIndex> {
        Ok(self.links.use_link(link.as_ref()))
    }

    fn build_regions(&mut self, regions: &'a [Region]) -> BuildResult<&'a [RegionId]> {
        let mut ids = BumpVec::with_capacity_in(regions.len(), self.bump);

        for region in regions {
            ids.push(self.build_region(region)?);
        }

        Ok(ids.into_bump_slice())
    }

    fn build_region(&mut self, region: &'a Region) -> BuildResult<RegionId> {
        let children = self.build_nodes(&region.children)?;
        let sources = self.resolve_links(&region.sources)?;
        let targets = self.resolve_links(&region.targets)?;
        let meta = self.build_terms(&region.meta)?;

        let signature = match &region.signature {
            Some(signature) => Some(self.build_term(&signature)?),
            None => None,
        };

        Ok(self.module.insert_region(model::Region {
            kind: region.kind,
            sources,
            targets,
            children,
            meta,
            signature,
            scope: todo!(),
        }))
    }

    fn build_symbol(&mut self, symbol: &'a Symbol) -> BuildResult<&'a model::Symbol<'a>> {
        let name = symbol.name.as_ref();
        let params = self.build_params(&symbol.params)?;
        let constraints = self.build_terms(&symbol.constraints)?;
        let signature = self.build_term(&symbol.signature)?;

        Ok(self.bump.alloc(model::Symbol {
            name,
            params,
            constraints,
            signature,
        }))
    }

    fn build_params(&mut self, params: &'a [Param]) -> BuildResult<&'a [model::Param<'a>]> {
        let mut result = BumpVec::with_capacity_in(params.len(), self.bump);

        for param in params {
            result.push(self.build_param(param)?);
        }

        Ok(result.into_bump_slice())
    }

    fn build_param(&mut self, param: &'a Param) -> BuildResult<model::Param<'a>> {
        let name = param.name.as_ref();
        let r#type = self.build_term(&param.r#type)?;
        Ok(model::Param { name, r#type })
    }

    fn resolve_var(&self, var_name: &'a VarName) -> BuildResult<VarId> {
        self.vars
            .resolve(var_name.as_ref())
            .map_err(|_| BuildError::Var(var_name.clone()))
    }

    fn resolve_symbol(&self, symbol_name: &'a SymbolName) -> BuildResult<NodeId> {
        // TODO: Instead of an error, add the symbol to the implicit imports

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
