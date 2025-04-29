use bumpalo::{Bump, collections::Vec as BumpVec};
use fxhash::FxHashMap;
use itertools::zip_eq;
use thiserror::Error;

use super::{
    LinkName, Module, Node, Operation, Param, Region, SeqPart, Symbol, SymbolName, Term, VarName,
};
use crate::v0::{RegionKind, ScopeClosure, table};
use crate::v0::{
    scope::{LinkTable, SymbolTable, VarTable},
    table::{LinkIndex, NodeId, RegionId, TermId, VarId},
};

pub struct Context<'a> {
    module: table::Module<'a>,
    bump: &'a Bump,
    vars: VarTable<'a>,
    links: LinkTable<&'a str>,
    symbols: SymbolTable<'a>,
    imports: FxHashMap<SymbolName, NodeId>,
    terms: FxHashMap<table::Term<'a>, TermId>,
}

impl<'a> Context<'a> {
    pub fn new(bump: &'a Bump) -> Self {
        Self {
            module: table::Module::default(),
            bump,
            vars: VarTable::new(),
            links: LinkTable::new(),
            symbols: SymbolTable::new(),
            imports: FxHashMap::default(),
            terms: FxHashMap::default(),
        }
    }

    pub fn resolve_module(&mut self, module: &'a Module) -> BuildResult<()> {
        self.module.root = self.module.insert_region(table::Region::default());
        self.symbols.enter(self.module.root);
        self.links.enter(self.module.root);

        let children = self.resolve_nodes(&module.root.children)?;
        let meta = self.resolve_terms(&module.root.meta)?;

        let (links, ports) = self.links.exit();
        self.symbols.exit();
        let scope = Some(table::RegionScope { links, ports });

        // Symbols that could not be resolved within the module still need to
        // be represented by a node. This is why we add import nodes.
        let all_children = {
            let mut all_children =
                BumpVec::with_capacity_in(children.len() + self.imports.len(), self.bump);
            all_children.extend(children);
            all_children.extend(self.imports.drain().map(|(_, node)| node));
            all_children.into_bump_slice()
        };

        self.module.regions[self.module.root.index()] = table::Region {
            kind: RegionKind::Module,
            sources: &[],
            targets: &[],
            children: all_children,
            meta,
            signature: None,
            scope,
        };

        Ok(())
    }

    fn resolve_terms(&mut self, terms: &'a [Term]) -> BuildResult<&'a [TermId]> {
        try_alloc_slice(self.bump, terms.iter().map(|term| self.resolve_term(term)))
    }

    fn resolve_term(&mut self, term: &'a Term) -> BuildResult<TermId> {
        let term = match term {
            Term::Wildcard => table::Term::Wildcard,
            Term::Var(var_name) => table::Term::Var(self.resolve_var(var_name)?),
            Term::Apply(symbol_name, terms) => {
                let symbol_id = self.resolve_symbol_name(symbol_name);
                let terms = self.resolve_terms(terms)?;
                table::Term::Apply(symbol_id, terms)
            }
            Term::List(parts) => table::Term::List(self.resolve_seq_parts(parts)?),
            Term::Literal(literal) => table::Term::Literal(literal.clone()),
            Term::Tuple(parts) => table::Term::Tuple(self.resolve_seq_parts(parts)?),
            Term::Func(region) => {
                let region = self.resolve_region(region, ScopeClosure::Closed)?;
                table::Term::Func(region)
            }
        };

        Ok(*self
            .terms
            .entry(term.clone())
            .or_insert_with(|| self.module.insert_term(term)))
    }

    fn resolve_seq_parts(&mut self, parts: &'a [SeqPart]) -> BuildResult<&'a [table::SeqPart]> {
        try_alloc_slice(
            self.bump,
            parts.iter().map(|part| self.resolve_seq_part(part)),
        )
    }

    fn resolve_seq_part(&mut self, part: &'a SeqPart) -> BuildResult<table::SeqPart> {
        Ok(match part {
            SeqPart::Item(term) => table::SeqPart::Item(self.resolve_term(term)?),
            SeqPart::Splice(term) => table::SeqPart::Splice(self.resolve_term(term)?),
        })
    }

    fn resolve_nodes(&mut self, nodes: &'a [Node]) -> BuildResult<&'a [NodeId]> {
        // Allocate ids for all nodes by introducing placeholders into the module.
        let ids: &[_] = self.bump.alloc_slice_fill_with(nodes.len(), |_| {
            self.module.insert_node(table::Node::default())
        });

        // For those nodes that introduce symbols, we then associate the symbol
        // with the id of the node. This serves as a form of forward declaration
        // so that the symbol is visible in the current region regardless of the
        // order of the nodes.
        for (id, node) in zip_eq(ids, nodes) {
            if let Some(symbol_name) = node.operation.symbol_name() {
                self.symbols
                    .insert(symbol_name.as_ref(), *id)
                    .map_err(|_| ResolveError::DuplicateSymbol(symbol_name.clone()))?;
            }
        }

        // Finally we can build the actual nodes.
        for (id, node) in zip_eq(ids, nodes) {
            self.resolve_node(*id, node)?;
        }

        Ok(ids)
    }

    fn resolve_node(&mut self, node_id: NodeId, node: &'a Node) -> BuildResult<()> {
        let inputs = self.resolve_links(&node.inputs)?;
        let outputs = self.resolve_links(&node.outputs)?;

        // When the node introduces a symbol it also introduces a new variable scope.
        if node.operation.symbol().is_some() {
            self.vars.enter(node_id);
        }

        let mut scope_closure = ScopeClosure::Open;

        let operation = match &node.operation {
            Operation::Invalid => table::Operation::Invalid,
            Operation::Dfg => table::Operation::Dfg,
            Operation::Cfg => table::Operation::Cfg,
            Operation::Block => table::Operation::Block,
            Operation::TailLoop => table::Operation::TailLoop,
            Operation::Conditional => table::Operation::Conditional,
            Operation::DefineFunc(symbol) => {
                let symbol = self.resolve_symbol(symbol)?;
                scope_closure = ScopeClosure::Closed;
                table::Operation::DefineFunc(symbol)
            }
            Operation::DeclareFunc(symbol) => {
                let symbol = self.resolve_symbol(symbol)?;
                table::Operation::DeclareFunc(symbol)
            }
            Operation::DefineAlias(symbol, term) => {
                let symbol = self.resolve_symbol(symbol)?;
                let term = self.resolve_term(term)?;
                table::Operation::DefineAlias(symbol, term)
            }
            Operation::DeclareAlias(symbol) => {
                let symbol = self.resolve_symbol(symbol)?;
                table::Operation::DeclareAlias(symbol)
            }
            Operation::DeclareConstructor(symbol) => {
                let symbol = self.resolve_symbol(symbol)?;
                table::Operation::DeclareConstructor(symbol)
            }
            Operation::DeclareOperation(symbol) => {
                let symbol = self.resolve_symbol(symbol)?;
                table::Operation::DeclareOperation(symbol)
            }
            Operation::Import(symbol_name) => table::Operation::Import {
                name: symbol_name.as_ref(),
            },
            Operation::Custom(term) => {
                let term = self.resolve_term(term)?;
                table::Operation::Custom(term)
            }
        };

        let meta = self.resolve_terms(&node.meta)?;
        let regions = self.resolve_regions(&node.regions, scope_closure)?;

        let signature = match &node.signature {
            Some(signature) => Some(self.resolve_term(signature)?),
            None => None,
        };

        // We need to close the variable scope if we have opened one before.
        if node.operation.symbol().is_some() {
            self.vars.exit();
        }

        self.module.nodes[node_id.index()] = table::Node {
            operation,
            inputs,
            outputs,
            regions,
            meta,
            signature,
        };

        Ok(())
    }

    fn resolve_links(&mut self, links: &'a [LinkName]) -> BuildResult<&'a [LinkIndex]> {
        try_alloc_slice(self.bump, links.iter().map(|link| self.resolve_link(link)))
    }

    fn resolve_link(&mut self, link: &'a LinkName) -> BuildResult<LinkIndex> {
        Ok(self.links.use_link(link.as_ref()))
    }

    fn resolve_regions(
        &mut self,
        regions: &'a [Region],
        scope_closure: ScopeClosure,
    ) -> BuildResult<&'a [RegionId]> {
        try_alloc_slice(
            self.bump,
            regions
                .iter()
                .map(|region| self.resolve_region(region, scope_closure)),
        )
    }

    fn resolve_region(
        &mut self,
        region: &'a Region,
        scope_closure: ScopeClosure,
    ) -> BuildResult<RegionId> {
        let meta = self.resolve_terms(&region.meta)?;
        let signature = match &region.signature {
            Some(signature) => Some(self.resolve_term(signature)?),
            None => None,
        };

        // We insert a placeholder for the region in order to allocate a region
        // id, which we need to track the region's scopes.
        let region_id = self.module.insert_region(table::Region::default());

        // Each region defines a new scope for symbols.
        self.symbols.enter(region_id);

        // If the region is closed, it also defines a new scope for links.
        if ScopeClosure::Closed == scope_closure {
            self.links.enter(region_id);
        }

        let sources = self.resolve_links(&region.sources)?;
        let targets = self.resolve_links(&region.targets)?;
        let children = self.resolve_nodes(&region.children)?;

        // Close the region's scopes.
        let scope = match scope_closure {
            ScopeClosure::Open => None,
            ScopeClosure::Closed => {
                let (links, ports) = self.links.exit();
                Some(table::RegionScope { links, ports })
            }
        };
        self.symbols.exit();

        self.module.regions[region_id.index()] = table::Region {
            kind: region.kind,
            sources,
            targets,
            children,
            meta,
            signature,
            scope,
        };

        Ok(region_id)
    }

    fn resolve_symbol(&mut self, symbol: &'a Symbol) -> BuildResult<&'a table::Symbol<'a>> {
        let name = symbol.name.as_ref();
        let params = self.resolve_params(&symbol.params)?;
        let constraints = self.resolve_terms(&symbol.constraints)?;
        let signature = self.resolve_term(&symbol.signature)?;

        Ok(self.bump.alloc(table::Symbol {
            name,
            params,
            constraints,
            signature,
        }))
    }

    /// Builds symbol parameters.
    ///
    /// This incrementally inserts the names of the parameters into the current
    /// variable scope, so that any parameter is in scope for each of its
    /// succeeding parameters.
    fn resolve_params(&mut self, params: &'a [Param]) -> BuildResult<&'a [table::Param<'a>]> {
        try_alloc_slice(
            self.bump,
            params.iter().map(|param| self.resolve_param(param)),
        )
    }

    /// Builds a symbol parameter.
    ///
    /// This inserts the name of the parameter into the current variable scope,
    /// making the parameter accessible as a variable.
    fn resolve_param(&mut self, param: &'a Param) -> BuildResult<table::Param<'a>> {
        let name = param.name.as_ref();
        let r#type = self.resolve_term(&param.r#type)?;

        self.vars
            .insert(param.name.as_ref())
            .map_err(|_| ResolveError::DuplicateVar(param.name.clone()))?;

        Ok(table::Param { name, r#type })
    }

    fn resolve_var(&self, var_name: &'a VarName) -> BuildResult<VarId> {
        self.vars
            .resolve(var_name.as_ref())
            .map_err(|_| ResolveError::UnknownVar(var_name.clone()))
    }

    /// Resolves a symbol name and returns the node that introduces the symbol.
    ///
    /// When there is no symbol with this name in scope, we create a new import
    /// node in the module and record that the symbol has been implicitly
    /// imported. At the end of the building process, these import nodes are
    /// inserted into the module's scope.
    fn resolve_symbol_name(&mut self, symbol_name: &'a SymbolName) -> NodeId {
        if let Ok(node) = self.symbols.resolve(symbol_name.as_ref()) {
            return node;
        }

        *self.imports.entry(symbol_name.clone()).or_insert_with(|| {
            self.module.insert_node(table::Node {
                operation: table::Operation::Import {
                    name: symbol_name.as_ref(),
                },
                ..Default::default()
            })
        })
    }

    pub fn finish(self) -> table::Module<'a> {
        self.module
    }
}

/// Error that may occur in [`Module::resolve`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ResolveError {
    /// Unknown variable.
    #[error("unknown var: {0}")]
    UnknownVar(VarName),
    /// Duplicate variable definition in the same symbol.
    #[error("duplicate var: {0}")]
    DuplicateVar(VarName),
    /// Duplicate symbol definition in the same region.
    #[error("duplicate symbol: {0}")]
    DuplicateSymbol(SymbolName),
}

type BuildResult<T> = Result<T, ResolveError>;

fn try_alloc_slice<T, E>(
    bump: &Bump,
    iter: impl IntoIterator<Item = Result<T, E>>,
) -> Result<&[T], E> {
    let iter = iter.into_iter();
    let mut vec = BumpVec::with_capacity_in(iter.size_hint().0, bump);
    for item in iter {
        vec.push(item?);
    }
    Ok(vec.into_bump_slice())
}
