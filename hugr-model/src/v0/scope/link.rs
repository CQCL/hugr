use std::hash::{BuildHasherDefault, Hash};

use fxhash::FxHasher;
use indexmap::IndexSet;

use crate::v0::{LinkIndex, RegionId};

type FxIndexSet<K> = IndexSet<K, BuildHasherDefault<FxHasher>>;

/// Table for tracking links between ports.
#[derive(Debug, Clone)]
pub struct LinkTable<K> {
    links: FxIndexSet<(RegionId, K)>,
    scopes: Vec<LinkScope>,
}

impl<K> LinkTable<K>
where
    K: Copy + Eq + Hash,
{
    /// Create a new empty link table.
    pub fn new() -> Self {
        Self {
            links: FxIndexSet::default(),
            scopes: Vec::new(),
        }
    }

    /// Enter a new scope for the given region.
    pub fn enter(&mut self, region: RegionId) {
        self.scopes.push(LinkScope {
            link_stack: self.links.len(),
            link_count: 0,
            port_count: 0,
            region,
        });
    }

    /// Exit a previously entered scope, returning the number of links and ports in the scope.
    pub fn exit(&mut self) -> (u32, u32) {
        let scope = self.scopes.pop().unwrap();
        self.links.drain(scope.link_stack..);
        debug_assert_eq!(self.links.len(), scope.link_stack);
        (scope.link_count, scope.port_count)
    }

    /// Resolve a link key to a link index, adding one more port to the current scope.
    ///
    /// If the key has not been used in the current scope before, it will be added to the link table.
    ///
    /// # Panics
    ///
    /// Panics if there are no open scopes.
    pub fn use_link(&mut self, key: K) -> LinkIndex {
        let scope = self.scopes.last_mut().unwrap();
        let (map_index, inserted) = self.links.insert_full((scope.region, key));

        if inserted {
            scope.link_count += 1;
        }

        scope.port_count += 1;
        LinkIndex::new(map_index - scope.link_stack)
    }

    /// Reset the link table to an empty state while preserving allocated memory.
    pub fn clear(&mut self) {
        self.links.clear();
        self.scopes.clear();
    }
}

impl<K> Default for LinkTable<K>
where
    K: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct LinkScope {
    link_stack: usize,
    link_count: u32,
    port_count: u32,
    region: RegionId,
}
