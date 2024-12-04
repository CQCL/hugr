use std::hash::BuildHasherDefault;

use crate::v0::{NodeId, SymbolIntroError, VarIndex, VarRef, VarRefError};
use fxhash::FxHasher;
use indexmap::IndexMap;

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Variable binding table that keeps track of variable resolution and scoping.
pub struct VarTable<'a> {
    scopes: Vec<VarScope>,
    table: FxIndexMap<(NodeId, &'a str), VarIndex>,
}

impl<'a> VarTable<'a> {
    /// Create a new variable table.
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            table: FxIndexMap::default(),
        }
    }

    /// Enter a new scope for the given node.
    pub fn enter(&mut self, node: NodeId) {
        self.scopes.push(VarScope { node, params: 0 })
    }

    /// Exit a previously entered scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no remaining open scopes.
    pub fn exit(&mut self) {
        let scope = self.scopes.pop().unwrap();

        // Remove the variable bindings that are no longer needed.
        // This is not necessary for correctness, but it helps to keep the table small.
        for _ in 0..scope.params {
            let last = self.table.pop();
            debug_assert_eq!(last.unwrap().0 .0, scope.node);
        }
    }

    /// Insert a new variable into the current scope.
    ///
    /// # Errors
    ///
    /// It is an error to insert a variable with the same name twice in the same scope.
    pub fn insert(&mut self, name: &'a str) -> Result<(), SymbolIntroError> {
        let var_scope = self.scopes.last_mut().unwrap();
        let node = var_scope.node;
        let index = var_scope.params;
        var_scope.params += 1;
        let other = self.table.insert((var_scope.node, name), index as u16);

        if let Some(other) = other {
            return Err(SymbolIntroError::DuplicateVar(
                node,
                other,
                index,
                name.to_string(),
            ));
        }

        Ok(())
    }

    /// Resolve a variable reference to a variable index.
    ///
    /// # Errors
    ///
    /// An error is returned if a named variable is not found in the current scope,
    /// or if an indexed variable from a different scope is referenced.
    pub fn resolve(&self, var_ref: VarRef<'a>) -> Result<VarRef<'a>, VarRefError> {
        match var_ref {
            VarRef::Index(node, index) => {
                // Check if this variable is in scope.
                self.scopes
                    .last()
                    .filter(|scope| scope.node == node)
                    .filter(|scope| index < scope.params)
                    .ok_or(VarRefError::NotVisible(node, index))?;

                Ok(var_ref)
            }
            VarRef::Named(name) => {
                let var_scope = self
                    .scopes
                    .last()
                    .ok_or_else(|| VarRefError::NameNotFound(name.to_string()))?;

                let index = *self
                    .table
                    .get(&(var_scope.node, name))
                    .ok_or_else(|| VarRefError::NameNotFound(name.to_string()))?;

                Ok(VarRef::Index(var_scope.node, index))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VarScope {
    /// The node that defines this scope.
    node: NodeId,
    /// The number of parameters that have been defined so far.
    /// This is initially zero and is incremented for each parameter that is added.
    /// In this way we ensure that the node's parameters come into scope gradually.
    params: u16,
}
