use fxhash::FxHasher;
use indexmap::IndexSet;
use std::hash::BuildHasherDefault;
use thiserror::Error;

use crate::v0::table::{NodeId, VarId};

type FxIndexSet<K> = IndexSet<K, BuildHasherDefault<FxHasher>>;

/// Table for keeping track of node parameters.
///
/// Variables refer to the parameters of a node which introduces a symbol.
/// Variables have an associated name and are scoped via nodes. The types of
/// parameters of a node may only refer to earlier parameters in the same node
/// in the order they are defined. A variable name must be unique within a
/// single node. Each node that introduces a symbol introduces a new isolated
/// scope for variables.
///
/// # Examples
///
/// ```
/// # pub use hugr_model::v0::table::{NodeId, VarId};
/// # pub use hugr_model::v0::scope::VarTable;
/// let mut vars = VarTable::new();
/// vars.enter(NodeId(0));
/// vars.insert("foo").unwrap();
/// assert_eq!(vars.resolve("foo").unwrap(), VarId(NodeId(0), 0));
/// assert!(!vars.is_visible(VarId(NodeId(0), 1)));
/// vars.insert("bar").unwrap();
/// assert!(vars.is_visible(VarId(NodeId(0), 1)));
/// assert_eq!(vars.resolve("bar").unwrap(), VarId(NodeId(0), 1));
/// vars.enter(NodeId(1));
/// assert!(vars.resolve("foo").is_err());
/// assert!(!vars.is_visible(VarId(NodeId(0), 0)));
/// vars.exit();
/// assert_eq!(vars.resolve("foo").unwrap(), VarId(NodeId(0), 0));
/// assert!(vars.is_visible(VarId(NodeId(0), 0)));
/// ```
#[derive(Debug, Clone, Default)]
pub struct VarTable<'a> {
    /// The set of variables in the currently active node and all its parent nodes.
    ///
    /// The order in this index set is the order in which variables were added to the table.
    /// This is used to efficiently remove all variables from the current node when exiting a scope.
    vars: FxIndexSet<(NodeId, &'a str)>,
    /// The stack of scopes that are currently open.
    scopes: Vec<VarScope>,
}

impl<'a> VarTable<'a> {
    /// Create a new empty variable table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enter a new scope for the given node.
    pub fn enter(&mut self, node: NodeId) {
        self.scopes.push(VarScope {
            node,
            var_count: 0,
            var_stack: self.vars.len(),
        });
    }

    /// Exit a previously entered scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no open scopes.
    pub fn exit(&mut self) {
        let scope = self.scopes.pop().unwrap();
        self.vars.drain(scope.var_stack..);
    }

    /// Resolve a variable name to a node and variable index.
    ///
    /// # Errors
    ///
    /// Returns an error if the variable is not defined in the current scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no open scopes.
    pub fn resolve(&self, name: &'a str) -> Result<VarId, UnknownVarError<'a>> {
        let scope = self.scopes.last().unwrap();
        let set_index = self
            .vars
            .get_index_of(&(scope.node, name))
            .ok_or(UnknownVarError(scope.node, name))?;
        let var_index = (set_index - scope.var_stack) as u16;
        Ok(VarId(scope.node, var_index))
    }

    /// Check if a variable is visible in the current scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no open scopes.
    #[must_use]
    pub fn is_visible(&self, var: VarId) -> bool {
        let scope = self.scopes.last().unwrap();
        scope.node == var.0 && var.1 < scope.var_count
    }

    /// Insert a new variable into the current scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the variable is already defined in the current scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no open scopes.
    pub fn insert(&mut self, name: &'a str) -> Result<VarId, DuplicateVarError<'a>> {
        let scope = self.scopes.last_mut().unwrap();
        let inserted = self.vars.insert((scope.node, name));

        if !inserted {
            return Err(DuplicateVarError(scope.node, name));
        }

        let var_index = scope.var_count;
        scope.var_count += 1;
        Ok(VarId(scope.node, var_index))
    }

    /// Reset the variable table to an empty state while preserving the allocations.
    pub fn clear(&mut self) {
        self.vars.clear();
        self.scopes.clear();
    }
}

#[derive(Debug, Clone)]
struct VarScope {
    /// The node that introduces this scope.
    node: NodeId,
    /// The number of variables in this scope.
    var_count: u16,
    /// The length of `VarTable::vars` when the scope was opened.
    var_stack: usize,
}

/// Error that occurs when a node defines two parameters with the same name.
#[derive(Debug, Clone, Error)]
#[error("node {0} already has a variable named `{1}`")]
pub struct DuplicateVarError<'a>(NodeId, &'a str);

/// Error that occurs when a variable is not defined in the current scope.
#[derive(Debug, Clone, Error)]
#[error("can not resolve variable `{1}` in node {0}")]
pub struct UnknownVarError<'a>(NodeId, &'a str);
