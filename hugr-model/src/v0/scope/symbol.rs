use std::{borrow::Cow, hash::BuildHasherDefault};

use fxhash::FxHasher;
use indexmap::{IndexMap, map::Entry};
use thiserror::Error;

use crate::v0::table::NodeId;

type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Symbol binding table that keeps track of symbol resolution.
///
/// Nodes in a module region may introduce a symbol so that other parts of the
/// IR can refer to the node. Symbols have an associated name which must be
/// unique within the module.
///
/// # Examples
///
/// ```
/// # pub use hugr_model::v0::table::{NodeId, RegionId};
/// # pub use hugr_model::v0::scope::SymbolTable;
/// let mut symbols = SymbolTable::new();
/// symbols.enter(RegionId(0));
/// symbols.insert("foo", NodeId(0)).unwrap();
/// assert_eq!(symbols.resolve("foo").unwrap(), NodeId(0));
/// symbols.enter(RegionId(1));
/// assert_eq!(symbols.resolve("foo").unwrap(), NodeId(0));
/// symbols.insert("foo", NodeId(1)).unwrap();
/// assert_eq!(symbols.resolve("foo").unwrap(), NodeId(1));
/// assert!(!symbols.is_visible(NodeId(0)));
/// symbols.exit();
/// assert_eq!(symbols.resolve("foo").unwrap(), NodeId(0));
/// assert!(symbols.is_visible(NodeId(0)));
/// assert!(!symbols.is_visible(NodeId(1)));
/// ```
#[derive(Debug, Clone, Default)]
pub struct SymbolTable<'a> {
    symbols: FxIndexMap<&'a str, NodeId>,
}

impl<'a> SymbolTable<'a> {
    /// Create a new symbol table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a new symbol into the table.
    ///
    /// # Errors
    ///
    /// Returns an error if the symbol is already defined.
    /// In the case of an error the table remains unchanged.
    ///
    /// # Panics
    ///
    /// Panics if there is no current scope.
    pub fn insert(&mut self, name: &'a str, node: NodeId) -> Result<(), DuplicateSymbolError> {
        match self.symbols.entry(name) {
            Entry::Occupied(prev) => Err(DuplicateSymbolError(name.into(), node, *prev.get())),
            Entry::Vacant(entry) => {
                entry.insert(node);
                Ok(())
            }
        }
    }

    /// Tries to resolve a symbol name.
    pub fn resolve(&self, name: &'a str) -> Result<NodeId, UnknownSymbolError> {
        self.symbols
            .get(name)
            .copied()
            .ok_or(UnknownSymbolError(name.into()))
    }

    /// Resets the symbol table to its initial state while maintaining its
    /// allocated memory.
    pub fn clear(&mut self) {
        self.symbols.clear();
    }
}

/// Error that occurs when trying to resolve an unknown symbol.
#[derive(Debug, Clone, Error)]
#[error("symbol name `{0}` not found in this scope")]
pub struct UnknownSymbolError<'a>(pub Cow<'a, str>);

/// Error that occurs when trying to introduce a symbol that is already defined in the current scope.
#[derive(Debug, Clone, Error)]
#[error("symbol `{0}` is already defined in this scope")]
pub struct DuplicateSymbolError<'a>(pub Cow<'a, str>, pub NodeId, pub NodeId);
