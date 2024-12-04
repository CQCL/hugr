use std::hash::BuildHasherDefault;

use fxhash::FxHasher;
use indexmap::IndexMap;

use crate::v0::{NodeId, RegionId, SymbolIntroError, SymbolRef, SymbolRefError};

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
    /// # Panics
    ///
    /// Panics if there is no current scope.
    pub fn insert(&mut self, name: &'a str, node: NodeId) -> Result<(), SymbolIntroError> {
        let scope_depth = self.scopes.len() as u16 - 1;
        let (position, shadowed) = self.symbols.insert_full(name, self.bindings.len());

        if let Some(shadowed) = shadowed {
            let (shadowed_node, shadowed_binding) = self.bindings.get_index(shadowed).unwrap();
            if shadowed_binding.scope_depth == scope_depth {
                return Err(SymbolIntroError::DuplicateSymbol(
                    node,
                    *shadowed_node,
                    name.to_string(),
                ));
            }
        }

        self.bindings.insert(
            node,
            Binding {
                scope_depth,
                shadows: shadowed,
                symbol_index: position,
            },
        );

        Ok(())
    }

    /// Tries to resolve a [`SymbolRef`] in the current scope.
    ///
    /// If the symbol is named but can not be found in the current scope, it is
    /// returned as is. This is to allow some alternative resolution strategy for
    /// top level symbols to be employed by the caller.
    ///
    /// # Errors
    ///
    /// Directly indexed symbols are checked for visibility and shadowing and an
    /// error is returned if the symbol is not visible.
    pub fn try_resolve(
        &self,
        symbol_ref: SymbolRef<'a>,
    ) -> Result<(SymbolRef<'a>, ScopeDepth), SymbolRefError> {
        match symbol_ref {
            SymbolRef::Direct(node) => {
                let binding = self
                    .bindings
                    .get(&node)
                    .ok_or(SymbolRefError::NotVisible(node))?;

                // Check if the symbol has been shadowed at this point.
                if self.symbols[binding.symbol_index] != binding.symbol_index {
                    return Err(SymbolRefError::NotVisible(node).into());
                }

                Ok((symbol_ref, binding.scope_depth))
            }
            SymbolRef::Named(name) => {
                if let Some(index) = self.symbols.get(name) {
                    let (node, binding) = self.bindings.get_index(*index).unwrap();
                    Ok((SymbolRef::Direct(*node), binding.scope_depth))
                } else {
                    Ok((symbol_ref, 0))
                }
            }
        }
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
