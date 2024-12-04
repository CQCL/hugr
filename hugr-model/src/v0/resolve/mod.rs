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
    ExtSetPart, ListPart, MetaItem, ModelError, Module, Node, NodeId, Operation, Param, RegionId,
    SymbolRef, Term, TermId,
};
use bumpalo::{collections::Vec as BumpVec, Bump};
use fxhash::FxHashMap;

mod bindings;
mod symbols;
mod vars;
pub use symbols::SymbolTable;
pub use vars::VarTable;

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

    /// Map from symbol names to newly created import nodes.
    ///
    /// When a symbol can not be resolved, we create an import node to be added to the root region
    /// at the end of the resolution process. This field is used to keep track of the import nodes
    /// that we have already created.
    symbol_import: FxHashMap<&'a str, NodeId>,

    vars: VarTable<'a>,
    symbols: SymbolTable<'a>,

    term_to_region: FxHashMap<TermId, RegionId>,
}

macro_rules! set_max {
    ($name:ident, $e:expr) => {
        $name = ::std::cmp::max($name, $e);
    };
}

impl<'m, 'a> Resolver<'m, 'a> {
    fn new(module: &'m mut Module<'a>, bump: &'a Bump) -> Self {
        Self {
            term_to_region: FxHashMap::default(),
            symbols: SymbolTable::new(),
            vars: VarTable::new(),
            module,
            bump,
            symbol_import: FxHashMap::default(),
        }
    }

    fn resolve_region(&mut self, region: RegionId) -> Result<(), ModelError> {
        let region_data = self
            .module
            .get_region(region)
            .ok_or_else(|| ModelError::RegionNotFound(region))?
            .clone();

        if let Some(signature) = region_data.signature {
            self.resolve_term(signature)?;
        }

        self.symbols.enter(region);

        // In a first pass over the region's children, we resolve the symbols
        // that are defined by the nodes as well as the links in their inputs
        // and outputs. This is to ensure that resolution is independent of the
        // order of the nodes.
        for child in region_data.children {
            let child_data = self
                .module
                .get_node(*child)
                .ok_or_else(|| ModelError::NodeNotFound(*child))?;

            if let Some(name) = child_data.operation.symbol() {
                self.symbols.insert(name, *child)?;
            }
        }

        // In a second pass, we resolve the remaining properties of the nodes.
        for child in region_data.children {
            self.resolve_node(*child)?;
        }

        self.symbols.exit();

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
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                for region in node_data.regions {
                    self.resolve_region(*region)?;
                }
                self.vars.exit();
            }

            Operation::DeclareFunc { decl } => {
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.resolve_term(decl.signature)?;
                self.vars.exit();
            }

            Operation::DefineAlias { decl, value } => {
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.resolve_term(value)?;
                self.vars.exit();
            }

            Operation::DeclareAlias { decl } => {
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.vars.exit();
            }

            Operation::DeclareConstructor { decl } => {
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.vars.exit();
            }

            Operation::DeclareOperation { decl } => {
                self.vars.enter(node);
                self.resolve_params(decl.params)?;
                self.resolve_terms(decl.constraints)?;
                self.vars.exit();
            }

            Operation::CallFunc { func } => {
                self.resolve_term(func)?;
            }

            Operation::LoadFunc { func } => {
                self.resolve_term(func)?;
            }

            Operation::Custom { operation } => {
                let (operation, _) = self.resolve_symbol_ref(operation)?;
                node_data.operation = Operation::Custom { operation };

                for region in node_data.regions {
                    self.resolve_region(*region)?;
                }
            }

            Operation::CustomFull { operation } => {
                let (operation, _) = self.resolve_symbol_ref(operation)?;
                node_data.operation = Operation::CustomFull { operation };

                for region in node_data.regions {
                    self.resolve_region(*region)?;
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
        for param in params {
            self.resolve_term(param.r#type)?;
            self.vars.insert(param.name)?;
        }

        Ok(())
    }

    fn resolve_terms(&mut self, terms: &'a [TermId]) -> Result<(), ModelError> {
        terms.iter().try_for_each(|term| {
            self.resolve_term(*term)?;
            Ok(())
        })
    }

    fn resolve_term(&mut self, term: TermId) -> Result<ScopeDepth, ModelError> {
        if let Some(region) = self.term_to_region.get(&term) {
            match self.symbols.region_to_depth(*region) {
                Some(scope_depth) => return Ok(scope_depth),
                None => panic!("this should be an error"),
            }
        }

        let term_data = self
            .module
            .get_term(term)
            .ok_or_else(|| ModelError::TermNotFound(term))
            .copied()?;

        let mut scope_depth = 0;

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
                let var = self.vars.resolve(var)?;
                self.module.terms[term.index()] = Term::Var(var);
            }

            super::Term::Apply { symbol, args } => {
                let (symbol, symbol_depth) = self.resolve_symbol_ref(symbol)?;
                self.module.terms[term.index()] = Term::Apply { symbol, args };
                set_max!(scope_depth, symbol_depth);

                for arg in args {
                    set_max!(scope_depth, self.resolve_term(*arg)?);
                }
            }

            super::Term::ApplyFull { symbol, args } => {
                let (symbol, symbol_depth) = self.resolve_symbol_ref(symbol)?;
                self.module.terms[term.index()] = Term::ApplyFull { symbol, args };
                set_max!(scope_depth, symbol_depth);

                for arg in args {
                    set_max!(scope_depth, self.resolve_term(*arg)?);
                }
            }

            super::Term::Quote { r#type } => {
                set_max!(scope_depth, self.resolve_term(r#type)?);
            }

            super::Term::List { parts } => {
                for part in parts {
                    set_max!(
                        scope_depth,
                        match part {
                            ListPart::Item(term) => self.resolve_term(*term)?,
                            ListPart::Splice(term) => self.resolve_term(*term)?,
                        }
                    );
                }
            }

            super::Term::ListType { item_type } => {
                set_max!(scope_depth, self.resolve_term(item_type)?);
            }

            super::Term::ExtSet { parts } => {
                for part in parts {
                    set_max!(
                        scope_depth,
                        match part {
                            ExtSetPart::Extension(_) => 0,
                            ExtSetPart::Splice(term) => self.resolve_term(*term)?,
                        }
                    );
                }
            }

            super::Term::Adt { variants } => {
                set_max!(scope_depth, self.resolve_term(variants)?);
            }

            super::Term::FuncType {
                inputs,
                outputs,
                extensions,
            } => {
                set_max!(scope_depth, self.resolve_term(inputs)?);
                set_max!(scope_depth, self.resolve_term(outputs)?);
                set_max!(scope_depth, self.resolve_term(extensions)?);
            }

            super::Term::Control { values } => {
                set_max!(scope_depth, self.resolve_term(values)?);
            }

            super::Term::NonLinearConstraint { term } => {
                set_max!(scope_depth, self.resolve_term(term)?);
            }
        };

        let region = self.symbols.depth_to_region(scope_depth).unwrap();
        self.term_to_region.insert(term, region);
        Ok(scope_depth)
    }

    fn resolve_symbol_ref(
        &mut self,
        symbol_ref: SymbolRef<'a>,
    ) -> Result<(SymbolRef<'a>, ScopeDepth), Error> {
        let (mut symbol_ref, scope_depth) = self.symbols.try_resolve(symbol_ref)?;

        if let SymbolRef::Named(name) = symbol_ref {
            let node = self.make_symbol_import(name);
            symbol_ref = SymbolRef::Direct(node);
        }

        Ok((symbol_ref, scope_depth))
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
        self.resolve_term(meta_item.value)?;
        Ok(())
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
type ScopeDepth = u16;
