use fxhash::FxHasher;
use indexmap::IndexSet;
use std::hash::BuildHasherDefault;
use thiserror::Error;

use crate::v0::{NodeId, VarIndex};

type FxIndexSet<K> = IndexSet<K, BuildHasherDefault<FxHasher>>;

/// Table for keeping track of node parameters.
#[derive(Debug, Clone)]
pub struct VarTable<'a> {
    vars: FxIndexSet<(NodeId, &'a str)>,
    scopes: Vec<VarScope>,
}

impl<'a> VarTable<'a> {
    /// Create a new empty variable table.
    pub fn new() -> Self {
        Self {
            vars: FxIndexSet::default(),
            scopes: Vec::new(),
        }
    }

    /// Enter a new scope for the given node.
    pub fn enter(&mut self, node: NodeId) {
        self.scopes.push(VarScope {
            node,
            var_count: 0,
            var_stack: self.vars.len(),
        })
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
    pub fn resolve(&self, name: &'a str) -> Result<(NodeId, VarIndex), UnknownVarError<'a>> {
        let scope = self.scopes.last().unwrap();
        let (set_index, _) = self
            .vars
            .get_full(&(scope.node, name))
            .ok_or(UnknownVarError(scope.node, name))?;
        let var_index = (set_index - scope.var_stack) as u16;
        Ok((scope.node, var_index))
    }

    /// Insert a new variable into the current scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the variable is already defined in the current scope.
    pub fn insert(&mut self, name: &'a str) -> Result<(NodeId, VarIndex), DuplicateVarError<'a>> {
        let scope = self.scopes.last_mut().unwrap();
        let inserted = self.vars.insert((scope.node, name));

        if !inserted {
            return Err(DuplicateVarError(scope.node, name));
        }

        let var_index = scope.var_count;
        scope.var_count += 1;
        Ok((scope.node, var_index))
    }

    /// Reset the variable table to an empty state while preserving the allocations.
    pub fn clear(&mut self) {
        self.vars.clear();
        self.scopes.clear();
    }
}

impl<'a> Default for VarTable<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct VarScope {
    node: NodeId,
    var_count: u16,
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
