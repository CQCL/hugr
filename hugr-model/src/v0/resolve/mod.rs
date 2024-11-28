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

// TODO: Document that the regions passed to custom operations are currently isolated
// from the parent scope of links.

// TODO: We currently rely on terms with names not to be deduplicated, since in different
// scopes the same name may be resolved to different nodes. We should either document this,
// and make the deduplication process respect this; or we should figure out a solution
// on how to deal with this otherwise.

use crate::v0::Link;

use super::{
    ExtSetPart, LinkId, LinkRef, ListPart, MetaItem, ModelError, Module, Node, NodeId, Operation,
    Param, RegionId, SymbolIntroError, SymbolRef, SymbolRefError, Term, TermId, VarRef,
    VarRefError,
};
use bitvec::vec::BitVec;
use bumpalo::{collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;

mod bindings;
use bindings::{Bindings, InsertBindingError};

/// Resolve all names in the module.
///
/// After successful resolution all [`SymbolRef`]s, [`VarRef`]s and [`LinkRef`]s
/// are replaced with their directly indexed variants. [`SymbolRef`]s that refer
/// to names which can not be found are resolved by inserting an import node
/// into the root region. Depending on the context, this may indicate an error or
/// provide an opportunity for linking.
pub fn resolve<'a>(module: &mut Module<'a>, bump: &'a Bump) -> Result<(), ModelError> {
    let term_visited = BitVec::repeat(false, module.terms.len());

    let mut resolver = Resolver {
        module,
        bump,
        scope_globals: Bindings::new(),
        scope_locals: Bindings::new(),
        global_import: FxHashMap::default(),
        links: FxHashMap::default(),
        term_visited,
    };

    resolver.resolve_region(resolver.module.root)?;
    resolver.add_imports_to_root();

    Ok(())
}

struct Resolver<'m, 'a> {
    module: &'m mut Module<'a>,
    bump: &'a Bump,

    /// Global scope.
    scope_globals: Bindings<&'a str, RegionId, NodeId>,

    global_import: FxHashMap<&'a str, NodeId>,

    /// Local scope.
    scope_locals: Bindings<&'a str, NodeId, u16>,

    links: FxHashMap<&'a str, LinkId>,

    /// Bit vector to keep track of visited terms so that we do not visit them multiple times.
    term_visited: BitVec,
}

impl<'m, 'a> Resolver<'m, 'a> {
    fn resolve_region(&mut self, region: RegionId) -> Result<(), ModelError> {
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

        // First collect all declarations in the region.
        self.scope_globals.enter(region);

        for child in region_data.children {
            let child_data = self
                .module
                .get_node(*child)
                .ok_or_else(|| ModelError::NodeNotFound(*child))?;

            if let Some(name) = child_data.operation.symbol() {
                self.scope_globals
                    .insert(name, *child)
                    .map_err(|err| match err {
                        InsertBindingError::NoScope => unreachable!(),
                        InsertBindingError::Duplicate(other) => {
                            SymbolIntroError::DuplicateSymbol(other, *child, name.to_string())
                        }
                    })?;
            }
        }

        // Then resolve all children.
        for child in region_data.children {
            self.resolve_node(*child)?;
        }

        // Finally reset the global scope
        self.scope_globals.exit();

        self.module.regions[region.index()] = region_data;

        Ok(())
    }

    fn resolve_isolated_region(&mut self, region: RegionId) -> Result<(), ModelError> {
        let links = std::mem::take(&mut self.links);
        self.resolve_region(region)?;
        self.links = links;
        Ok(())
    }

    fn resolve_node(&mut self, node: NodeId) -> Result<(), ModelError> {
        let mut node_data = self
            .module
            .get_node(node)
            .ok_or_else(|| ModelError::NodeNotFound(node))?
            .clone();

        node_data.inputs = self.resolve_links(node_data.inputs);
        node_data.outputs = self.resolve_links(node_data.outputs);
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
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                for region in node_data.regions {
                    self.resolve_region(*region)?;
                }
                self.scope_locals.exit();
            }

            Operation::DeclareFunc { decl } => {
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                self.scope_locals.exit();
            }

            Operation::DefineAlias { decl, value } => {
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_term(value)?;
                self.scope_locals.exit();
            }

            Operation::DeclareAlias { decl } => {
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.scope_locals.exit();
            }

            Operation::DeclareConstructor { decl } => {
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.scope_locals.exit();
            }

            Operation::DeclareOperation { decl } => {
                self.scope_locals.enter_isolated(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.scope_locals.exit();
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
            let node = *self.scope_locals.scope().unwrap();

            self.scope_locals
                .insert(param.name, index as u16)
                .map_err(|err| match err {
                    InsertBindingError::NoScope => unreachable!(),
                    InsertBindingError::Duplicate(other) => SymbolIntroError::DuplicateVar(
                        node,
                        other,
                        index as u16,
                        param.name.to_string(),
                    ),
                })?;
        }

        Ok(())
    }

    fn resolve_links(&mut self, link_refs: &'a [LinkRef<'a>]) -> &'a [LinkRef<'a>] {
        use std::collections::hash_map::Entry;

        // Short circuit if all links are already resolved.
        if link_refs.iter().all(|l| matches!(l, LinkRef::Id(_))) {
            return link_refs;
        }

        // There are non-resolved links. To resolve them we need to reallocate the links slice.
        let mut resolved = BumpVec::with_capacity_in(link_refs.len(), self.bump);

        for link_ref in link_refs {
            resolved.push(match link_ref {
                LinkRef::Id(_) => *link_ref,
                LinkRef::Named(name) => match self.links.entry(*name) {
                    Entry::Occupied(entry) => LinkRef::Id(*entry.get()),
                    Entry::Vacant(entry) => {
                        let link_id = self.module.insert_link(Link { name });
                        entry.insert(link_id);
                        LinkRef::Id(link_id)
                    }
                },
            });
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
                if self.scope_locals.scope() != Some(&node) {
                    return Err(VarRefError::NotVisible(node, index));
                }

                Ok(local_ref)
            }
            VarRef::Named(name) => {
                let index = self
                    .scope_locals
                    .get(name)
                    .ok_or_else(|| VarRefError::NameNotFound(name.to_string()))?;

                // UNWRAP: If the local is in scope, then the scope must be set.
                let node = self.scope_locals.scope().unwrap();
                Ok(VarRef::Index(*node, index))
            }
        }
    }

    fn resolve_symbol_ref(&mut self, symbol_ref: SymbolRef<'a>) -> Result<SymbolRef<'a>, Error> {
        match symbol_ref {
            SymbolRef::Direct(node) => {
                // Check whether the global is visible at this point.
                let name = self.module.symbol_name(symbol_ref)?;
                let visible = self.scope_globals.get(name);

                if visible != Some(node) {
                    return Err(SymbolRefError::NotVisible(node).into());
                }

                Ok(symbol_ref)
            }
            SymbolRef::Named(name) => {
                if let Some(node) = self.scope_globals.get(name) {
                    return Ok(SymbolRef::Direct(node));
                }

                Ok(SymbolRef::Direct(
                    *self.global_import.entry(name).or_insert_with(|| {
                        self.module.insert_node(Node {
                            operation: Operation::Import { name },
                            ..Node::default()
                        })
                    }),
                ))
            }
        }
    }

    fn resolve_meta(&mut self, meta_item: &'a MetaItem<'a>) -> Result<(), Error> {
        self.resolve_term(meta_item.value)
    }

    fn add_imports_to_root(&mut self) {
        // Short circuit if there are no imports to add. This avoids allocating
        // a new slice for the children of the root region.
        if self.global_import.is_empty() {
            return;
        }

        let root_region = &mut self.module.regions[self.module.root.index()];
        let mut children = BumpVec::with_capacity_in(
            root_region.children.len() + self.global_import.len(),
            self.bump,
        );
        children.extend(self.global_import.drain().map(|(_, node)| node));
        children.extend(root_region.children.iter().copied());
        root_region.children = children.into_bump_slice();
    }
}

type Error = ModelError;