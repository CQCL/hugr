use std::hash::{BuildHasherDefault, Hash};

use fxhash::FxHasher;
use indexmap::IndexSet;

use crate::v0::{LinkIndex, RegionId};

type FxIndexSet<K> = IndexSet<K, BuildHasherDefault<FxHasher>>;

pub struct LinkTable<K> {
    links: FxIndexSet<(RegionId, K)>,
    scopes: Vec<LinkScope>,
}

impl<K> LinkTable<K>
where
    K: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            links: FxIndexSet::default(),
            scopes: Vec::new(),
        }
    }

    pub fn enter(&mut self, region: RegionId) {
        self.scopes.push(LinkScope {
            link_stack: self.links.len(),
            link_count: 0,
            region,
        });
    }

    pub fn exit(&mut self) -> usize {
        let scope = self.scopes.pop().unwrap();
        self.links.drain(scope.link_stack..);
        debug_assert_eq!(self.links.len(), scope.link_stack);
        scope.link_count
    }

    pub fn resolve(&mut self, key: K) -> LinkIndex {
        let scope = self.scopes.last_mut().unwrap();
        let (map_index, inserted) = self.links.insert_full((scope.region, key));

        if inserted {
            scope.link_count += 1;
        }

        LinkIndex::new(map_index - scope.link_stack)
    }

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

struct LinkScope {
    link_stack: usize,
    link_count: usize,
    region: RegionId,
}
