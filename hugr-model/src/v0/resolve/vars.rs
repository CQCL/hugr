use crate::v0::{NodeId, SymbolIntroError, VarIndex, VarRef, VarRefError};
use fxhash::FxHashMap;

pub struct Vars<'a> {
    scopes: Vec<VarScope>,
    table: FxHashMap<(NodeId, &'a str), VarIndex>,
}

impl<'a> Vars<'a> {
    pub fn new() -> Self {
        Self {
            scopes: Vec::new(),
            table: FxHashMap::default(),
        }
    }

    pub fn enter(&mut self, node: NodeId) {}

    pub fn exit(&mut self) {}

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
    node: NodeId,
    params: u16,
}
