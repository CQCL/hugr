//! Name resolution.
//!
//! Some nodes have the capability to introduce a named symbol. Symbols are
//! organised and scoped through regions. Within a region no two symbols can
//! introduce the same name. However, symbols can shadow other symbols from
//! outer regions by having identical names. When directly referencing a symbol
//! by the id of the node that introduced it, the symbol must be visible in the
//! current scope as if it was referred to by name.
//!
//! Nodes that introduce symbols may also include a parameter list. The
//! parameters of the node can be referenced by variables in terms. The type of
//! a parameter can contain variables that reference parameters that appear
//! earlier in the list. A node's variable scope is isolated from the variables
//! in the parameter list of any ancestor nodes. When directly referencing a
//! variable by the id of the node and the index in the node's parameter list,
//! the variable must be visible in the current scope as if it was referred to
//! by name.
//!
//! Ports of nodes and regions are connected when they have the same link. Links
//! are scoped through regions and a region's source and target ports are
//! considered to be within the region. Depending on the operation, regions may
//! be isolated in which case no link can cross the boundary of the region.

// TODO: Document that the regions passed to custom operations are currently isolated
// from the parent scope of links.

// TODO: We currently rely on terms with names not to be deduplicated, since in different
// scopes the same name may be resolved to different nodes. We should either document this,
// and make the deduplication process respect this; or we should figure out a solution
// on how to deal with this otherwise.

use super::{
    ExtSetPart, LinkRef, ListPart, MetaItem, ModelError, Module, Node, NodeId, Operation, Param,
    RegionId, SymbolIntroError, SymbolRef, SymbolRefError, Term, TermId, VarIndex, VarRef,
    VarRefError,
};
use bitvec::vec::BitVec;
use bumpalo::{collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;

mod bindings;
use bindings::Bindings;

/// Resolve all names in the module.
///
/// After successful resolution all [`SymbolRef`]s, [`VarRef`]s and [`LinkRef`]s
/// are replaced with their directly indexed variants. [`SymbolRef`]s that refer
/// to names which can not be found are resolved by inserting an import node
/// into the root region. Depending on the context, this may indicate an error or
/// provide an opportunity for linking.
pub fn resolve<'a>(module: &mut Module<'a>, bump: &'a Bump) -> Result<(), ModelError> {
    let mut resolver = Resolver::new(module, bump);
    resolver.resolve_region(resolver.module.root)?;
    resolver.add_imports_to_root();
    Ok(())
}

struct Resolver<'m, 'a> {
    /// The module to resolve.
    module: &'m mut Module<'a>,

    /// The bump allocator to use for allocations.
    bump: &'a Bump,

    /// Symbols that are visible in the current scope.
    symbol_scope: Bindings<&'a str, RegionId, NodeId>,

    /// Map from symbol names to newly created import nodes.
    ///
    /// When a symbol can not be resolved, we create an import node to be added to the root region
    /// at the end of the resolution process. This field is used to keep track of the import nodes
    /// that we have already created.
    symbol_import: FxHashMap<&'a str, NodeId>,

    /// Variables that are visible in the current scope.
    var_scope: Bindings<&'a str, NodeId, VarIndex>,

    // Links that are visible in the current scope.
    // link_scope: Bindings<&'a str, RegionId, LinkId>,
    /// Bit vector to keep track of visited terms so that we do not visit them multiple times.
    term_visited: BitVec,
}

impl<'m, 'a> Resolver<'m, 'a> {
    fn new(module: &'m mut Module<'a>, bump: &'a Bump) -> Self {
        Self {
            term_visited: BitVec::repeat(false, module.terms.len()),
            module,
            bump,
            symbol_scope: Bindings::new(),
            var_scope: Bindings::new(),
            symbol_import: FxHashMap::default(),
            // link_scope: Bindings::new(),
        }
    }

    /// Resolve a non-isolated region.
    fn resolve_region(&mut self, region: RegionId) -> Result<(), ModelError> {
        self.resolve_region_inner(region)?;
        Ok(())
    }

    /// Resolve an isolated region.
    fn resolve_isolated_region(&mut self, region: RegionId) -> Result<(), ModelError> {
        // self.link_scope.enter_isolated(region);
        self.resolve_region_inner(region)?;
        // self.link_scope.exit();
        Ok(())
    }

    fn resolve_region_inner(&mut self, region: RegionId) -> Result<(), ModelError> {
        let mut region_data = self
            .module
            .get_region(region)
            .ok_or_else(|| ModelError::RegionNotFound(region))?
            .clone();

        region_data.sources = self.resolve_links(region_data.sources);
        region_data.targets = self.resolve_links(region_data.targets);

        if let Some(signature) = region_data.signature {
            self.resolve_term(signature)?;
        }

        self.symbol_scope.enter(region);

        // In a first pass over the region's children, we resolve the symbols
        // that are defined by the nodes as well as the links in their inputs
        // and outputs. This is to ensure that resolution is independent of the
        // order of the nodes.
        for child in region_data.children {
            let mut child_data = self
                .module
                .get_node(*child)
                .ok_or_else(|| ModelError::NodeNotFound(*child))?
                .clone();

            if let Some(name) = child_data.operation.symbol() {
                self.symbol_scope
                    .try_insert(name, *child)
                    .map_err(|other| {
                        SymbolIntroError::DuplicateSymbol(other, *child, name.to_string())
                    })?;
            }

            child_data.inputs = self.resolve_links(child_data.inputs);
            child_data.outputs = self.resolve_links(child_data.outputs);
            self.module.nodes[child.index()] = child_data;
        }

        // In a second pass, we resolve the remaining properties of the nodes.
        for child in region_data.children {
            self.resolve_node(*child)?;
        }

        self.symbol_scope.exit();
        self.module.regions[region.index()] = region_data;

        Ok(())
    }

    fn resolve_node(&mut self, node: NodeId) -> Result<(), ModelError> {
        let mut node_data = self
            .module
            .get_node(node)
            .ok_or_else(|| ModelError::NodeNotFound(node))?
            .clone();

        self.resolve_terms(node_data.params)?;

        match node_data.operation {
            Operation::Invalid
            | Operation::Dfg
            | Operation::Cfg
            | Operation::Block
            | Operation::TailLoop
            | Operation::Conditional
            | Operation::Tag { .. } => {
                for region in node_data.regions {
                    self.resolve_region(*region)?;
                }
            }

            Operation::DefineFunc { decl } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                for region in node_data.regions {
                    self.resolve_isolated_region(*region)?;
                }
                self.var_scope.exit();
            }

            Operation::DeclareFunc { decl } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                self.var_scope.exit();
            }

            Operation::DefineAlias { decl, value } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_term(value)?;
                self.var_scope.exit();
            }

            Operation::DeclareAlias { decl } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.var_scope.exit();
            }

            Operation::DeclareConstructor { decl } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.var_scope.exit();
            }

            Operation::DeclareOperation { decl } => {
                self.var_scope.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.var_scope.exit();
            }

            Operation::CallFunc { func } => {
                self.resolve_term(func)?;
            }

            Operation::LoadFunc { func } => {
                self.resolve_term(func)?;
            }

            Operation::Custom { operation } => {
                let operation = self.resolve_symbol_ref(operation)?;
                node_data.operation = Operation::Custom { operation };

                for region in node_data.regions {
                    self.resolve_isolated_region(*region)?;
                }
            }

            Operation::CustomFull { operation } => {
                let operation = self.resolve_symbol_ref(operation)?;
                node_data.operation = Operation::CustomFull { operation };

                for region in node_data.regions {
                    self.resolve_isolated_region(*region)?;
                }
            }

            Operation::Import { .. } => {}
        };

        for meta in node_data.meta {
            self.resolve_meta(meta)?;
        }

        if let Some(signature) = node_data.signature {
            self.resolve_term(signature)?;
        }

        self.module.nodes[node.index()] = node_data;

        Ok(())
    }

    fn resolve_params(&mut self, params: &'a [Param<'a>]) -> Result<(), ModelError> {
        for (index, param) in params.iter().enumerate() {
            self.resolve_term(param.r#type)?;
            let node = *self.var_scope.scope().unwrap();

            self.var_scope
                .try_insert(param.name, index as u16)
                .map_err(|other| {
                    SymbolIntroError::DuplicateVar(
                        node,
                        other,
                        index as u16,
                        param.name.to_string(),
                    )
                })?;
        }

        Ok(())
    }

    fn resolve_links(&mut self, link_refs: &'a [LinkRef<'a>]) -> &'a [LinkRef<'a>] {
        // Short circuit if all links are already resolved.
        if link_refs.iter().all(|l| matches!(l, LinkRef::Index(_))) {
            return link_refs;
        }

        // There are non-resolved links. To resolve them we need to reallocate the links slice.
        let mut resolved = BumpVec::with_capacity_in(link_refs.len(), self.bump);

        for link_ref in link_refs {
            todo!();
        }

        resolved.into_bump_slice()
    }

    fn resolve_terms(&mut self, terms: &'a [TermId]) -> Result<(), ModelError> {
        terms.iter().try_for_each(|term| self.resolve_term(*term))
    }

    fn resolve_term(&mut self, term: TermId) -> Result<(), ModelError> {
        // Mark the term as visited and short circuit if it has already been visited before.
        if self.term_visited.replace(term.index(), true) {
            return Ok(());
        }

        let term_data = self
            .module
            .get_term(term)
            .ok_or_else(|| ModelError::TermNotFound(term))
            .copied()?;

        match term_data {
            super::Term::Wildcard
            | super::Term::Type
            | super::Term::StaticType
            | super::Term::Constraint
            | super::Term::Str(_)
            | super::Term::StrType
            | super::Term::Nat(_)
            | super::Term::NatType
            | super::Term::ExtSetType
            | super::Term::ControlType => {}

            super::Term::Var(var) => {
                let local = self.resolve_var_ref(var)?;
                self.module.terms[term.index()] = Term::Var(local);
            }

            super::Term::Apply {
                symbol: global,
                args,
            } => {
                let global = self.resolve_symbol_ref(global)?;
                self.module.terms[term.index()] = Term::Apply {
                    symbol: global,
                    args,
                };

                for arg in args {
                    self.resolve_term(*arg)?;
                }
            }

            super::Term::ApplyFull { symbol, args } => {
                let symbol = self.resolve_symbol_ref(symbol)?;
                self.module.terms[term.index()] = Term::ApplyFull { symbol, args };

                for arg in args {
                    self.resolve_term(*arg)?;
                }
            }

            super::Term::Quote { r#type } => {
                self.resolve_term(r#type)?;
            }

            super::Term::List { parts } => {
                for part in parts {
                    match part {
                        ListPart::Item(term) => self.resolve_term(*term)?,
                        ListPart::Splice(term) => self.resolve_term(*term)?,
                    }
                }
            }

            super::Term::ListType { item_type } => {
                self.resolve_term(item_type)?;
            }

            super::Term::ExtSet { parts } => {
                for part in parts {
                    match part {
                        ExtSetPart::Extension(_) => {}
                        ExtSetPart::Splice(term) => self.resolve_term(*term)?,
                    }
                }
            }

            super::Term::Adt { variants } => {
                self.resolve_term(variants)?;
            }

            super::Term::FuncType {
                inputs,
                outputs,
                extensions,
            } => {
                self.resolve_term(inputs)?;
                self.resolve_term(outputs)?;
                self.resolve_term(extensions)?;
            }

            super::Term::Control { values } => {
                self.resolve_term(values)?;
            }

            super::Term::NonLinearConstraint { term } => {
                self.resolve_term(term)?;
            }
        }

        Ok(())
    }

    fn resolve_var_ref(&self, local_ref: VarRef<'a>) -> Result<VarRef<'a>, VarRefError> {
        match local_ref {
            VarRef::Index(node, index) => {
                // Check whether this local would have been visible at this point.
                if self.var_scope.scope() != Some(&node) {
                    return Err(VarRefError::NotVisible(node, index));
                }

                Ok(local_ref)
            }
            VarRef::Named(name) => {
                let index = self
                    .var_scope
                    .get_value_by_key(name)
                    .ok_or_else(|| VarRefError::NameNotFound(name.to_string()))?;

                // UNWRAP: If the local is in scope, then the scope must be set.
                let node = self.var_scope.scope().unwrap();
                Ok(VarRef::Index(*node, index))
            }
        }
    }

    fn resolve_symbol_ref(&mut self, symbol_ref: SymbolRef<'a>) -> Result<SymbolRef<'a>, Error> {
        match symbol_ref {
            SymbolRef::Direct(node) => {
                // Check whether the global is visible at this point.
                if self.symbol_scope.get_key_by_value(node).is_none() {
                    return Err(SymbolRefError::NotVisible(node).into());
                }

                Ok(symbol_ref)
            }
            SymbolRef::Named(name) => {
                if let Some(node) = self.symbol_scope.get_value_by_key(name) {
                    return Ok(SymbolRef::Direct(node));
                }

                Ok(SymbolRef::Direct(self.make_symbol_import(name)))
            }
        }
    }

    /// Creates an import node for the given symbol name and returns its id.
    ///
    /// When there already exists an import node for the symbol name, the id of
    /// the existing node is returned. The import node is added to the root
    /// region of the module at the end of the resolution process via
    /// [`Self::add_imports_to_root`].
    fn make_symbol_import(&mut self, name: &'a str) -> NodeId {
        *self.symbol_import.entry(name).or_insert_with(|| {
            self.module.insert_node(Node {
                operation: Operation::Import { name },
                ..Node::default()
            })
        })
    }

    fn resolve_meta(&mut self, meta_item: &'a MetaItem<'a>) -> Result<(), Error> {
        self.resolve_term(meta_item.value)
    }

    fn add_imports_to_root(&mut self) {
        // Short circuit if there are no imports to add. This avoids allocating
        // a new slice for the children of the root region.
        if self.symbol_import.is_empty() {
            return;
        }

        let root_region = &mut self.module.regions[self.module.root.index()];
        let mut children = BumpVec::with_capacity_in(
            root_region.children.len() + self.symbol_import.len(),
            self.bump,
        );
        children.extend(self.symbol_import.drain().map(|(_, node)| node));
        children.extend(root_region.children.iter().copied());
        root_region.children = children.into_bump_slice();
    }
}

type Error = ModelError;
