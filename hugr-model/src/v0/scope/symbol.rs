use std::{borrow::Cow, hash::BuildHasherDefault};

use fxhash::FxHasher;
use indexmap::IndexMap;
use thiserror::Error;

use crate::v0::{NodeId, RegionId};

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Symbol binding table that keeps track of symbol resolution and scoping.
pub struct SymbolTable<'a> {
    symbols: FxIndexMap<&'a str, BindingIndex>,
    bindings: FxIndexMap<NodeId, Binding>,
    scopes: FxIndexMap<RegionId, Scope>,
}

impl<'a> SymbolTable<'a> {
    /// Create a new symbol table.
    pub fn new() -> Self {
        Self {
            symbols: FxIndexMap::default(),
            bindings: FxIndexMap::default(),
            scopes: FxIndexMap::default(),
        }
    }

    /// Enter a new scope for the given region.
    pub fn enter(&mut self, region: RegionId) {
        self.scopes.insert(
            region,
            Scope {
                binding_stack: self.bindings.len(),
            },
        );
    }

    /// Exit a previously entered scope.
    ///
    /// # Panics
    ///
    /// Panics if there are no remaining open scopes.
    pub fn exit(&mut self) {
        let (_, scope) = self.scopes.pop().unwrap();

        for _ in scope.binding_stack..self.bindings.len() {
            let (_, binding) = self.bindings.pop().unwrap();

            if let Some(shadows) = binding.shadows {
                self.symbols[binding.symbol_index] = shadows;
            } else {
                let last = self.symbols.pop();
                debug_assert_eq!(last.unwrap().1, self.bindings.len());
            }
        }
    }

    /// Insert a new symbol into the current scope.
    ///
    /// # Errors
    ///
    /// Returns an error if the symbol is already defined in the current scope.
    /// In the case of an error the table remains unchanged.
    ///
    /// # Panics
    ///
    /// Panics if there is no current scope.
    pub fn insert(&mut self, name: &'a str, node: NodeId) -> Result<(), SymbolIntroError> {
        let scope_depth = self.scopes.len() as u16 - 1;
        let (symbol_index, shadowed) = self.symbols.insert_full(name, self.bindings.len());

        if let Some(shadowed) = shadowed {
            let (shadowed_node, shadowed_binding) = self.bindings.get_index(shadowed).unwrap();
            if shadowed_binding.scope_depth == scope_depth {
                self.symbols.insert(name, shadowed);
                return Err(SymbolIntroError::Duplicate(
                    name.into(),
                    node,
                    *shadowed_node,
                ));
            }
        }

        self.bindings.insert(
            node,
            Binding {
                scope_depth,
                shadows: shadowed,
                symbol_index,
            },
        );

        Ok(())
    }

    /// Check whether a symbol is currently visible in the current scope.
    pub fn is_visible(&self, node: NodeId) -> bool {
        let Some(binding) = self.bindings.get(&node) else {
            return false;
        };

        // Check that the symbol has not been shadowed at this point.
        self.symbols[binding.symbol_index] == binding.symbol_index
    }

    /// Tries to resolve a [`SymbolRef`] in the current scope.
    pub fn resolve(&self, name: &'a str) -> Result<NodeId, SymbolResolveError> {
        // match symbol_ref {
        //     GlobalRef::Direct(node) => {

        //         Ok(node)
        //     }
        let index = *self
            .symbols
            .get(name)
            .ok_or(SymbolResolveError::NotFound(name.into()))?;

        // NOTE: The unwrap is safe because the `symbols` map
        // points to valid indices in the `bindings` map.
        let (node, _) = self.bindings.get_index(index).unwrap();
        Ok(*node)
    }

    /// Returns the depth of the given region, if it corresponds to a currently open scope.
    pub fn region_to_depth(&self, region: RegionId) -> Option<ScopeDepth> {
        Some(self.scopes.get_index_of(&region)? as _)
    }

    /// Returns the region corresponding to the scope at the given depth.
    pub fn depth_to_region(&self, depth: ScopeDepth) -> Option<RegionId> {
        let (region, _) = self.scopes.get_index(depth as _)?;
        Some(*region)
    }

    /// Resets the symbol table to its initial state while maintaining its
    /// allocated memory.
    pub fn clear(&mut self) {
        self.symbols.clear();
        self.bindings.clear();
        self.scopes.clear();
    }
}

impl<'a> Default for SymbolTable<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Binding {
    /// The depth of the scope in which this binding is defined.
    scope_depth: ScopeDepth,

    /// The index of the binding that is shadowed by this one, if any.
    shadows: Option<BindingIndex>,

    /// The index of this binding's symbol in the symbol table.
    ///
    /// The symbol table always points to the currently visible binding for a
    /// symbol. Therefore this index is only valid if this binding is not shadowed.
    /// In particular, we detect shadowing by checking if the entry in the symbol
    /// table at this index does indeed point to this binding.
    symbol_index: SymbolIndex,
}

#[derive(Debug, Clone, Copy)]
struct Scope {
    /// The length of the `bindings` stack when this scope was entered.
    binding_stack: usize,
}

type BindingIndex = usize;
type SymbolIndex = usize;

pub type ScopeDepth = u16;

/// Error while resolving a symbol.
#[derive(Debug, Clone, Error)]
pub enum SymbolResolveError<'a> {
    /// Symbol not found in the current scope.
    #[error("symbol name `{0}` not found in this scope")]
    NotFound(Cow<'a, str>),
}

/// Error while introducing a symbol.
#[derive(Debug, Clone, Error)]
pub enum SymbolIntroError<'a> {
    /// Duplicate symbol definition in the same scope.
    #[error("symbol `{0}` is already defined in this scope")]
    Duplicate(Cow<'a, str>, NodeId, NodeId),
}
