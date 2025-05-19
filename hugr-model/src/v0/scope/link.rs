use std::hash::{BuildHasherDefault, Hash};

use fxhash::FxHasher;
use indexmap::IndexSet;

use crate::v0::table::{LinkIndex, RegionId};

type FxIndexSet<K> = IndexSet<K, BuildHasherDefault<FxHasher>>;

/// Table for tracking links between ports.
///
/// Two ports are connected when they share the same link. Links are named and
/// scoped via closed regions. Links from one closed region are not visible
/// in another. Open regions are considered to form the same scope as their
/// parent region. Links do not have a unique point of declaration.
///
/// The link table keeps track of an association between a key of type `K` and
/// the link indices within each closed region. When resolving links from a text format,
/// `K` is the name of the link as a string slice. However the link table might
/// is also useful in other contexts where the key is not a string when constructing
/// a module from a different representation.
///
/// # Examples
///
/// ```
/// # pub use hugr_model::v0::table::RegionId;
/// # pub use hugr_model::v0::scope::LinkTable;
/// let mut links = LinkTable::new();
/// links.enter(RegionId(0));
/// let foo_0 = links.use_link("foo");
/// let bar_0 = links.use_link("bar");
/// assert_eq!(foo_0, links.use_link("foo"));
/// assert_eq!(bar_0, links.use_link("bar"));
/// let (num_links, num_ports) = links.exit();
/// assert_eq!(num_links, 2);
/// assert_eq!(num_ports, 4);
/// ```
#[derive(Debug, Clone)]
pub struct LinkTable<K> {
    /// The set of links in the currently active region and all parent regions.
    ///
    /// The order in this index set is the order in which links were added to the table.
    /// This is used to efficiently remove all links from the current region when exiting a scope.
    links: FxIndexSet<(RegionId, K)>,

    /// The stack of scopes that are currently open.
    scopes: Vec<LinkScope>,
}

impl<K> LinkTable<K>
where
    K: Copy + Eq + Hash,
{
    /// Create a new empty link table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            links: FxIndexSet::default(),
            scopes: Vec::new(),
        }
    }

    /// Enter a new scope for the given closed region.
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
    /// The length of `LinkTable::links` when the scope was opened.
    link_stack: usize,
    /// The number of links in this scope.
    link_count: u32,
    /// The number of ports in this scope.
    port_count: u32,
    /// The region that introduces this scope.
    region: RegionId,
}
