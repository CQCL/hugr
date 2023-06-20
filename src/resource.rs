//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.
//!
//! [`OpDef`]: crate::ops::custom::OpDef

use std::collections::HashSet;
use std::fmt::{self, Display};

use smol_str::SmolStr;

use crate::ops::{OpDef, OpaqueOp};
use crate::types::custom::CustomType;

/// A unique identifier for a resource.
///
/// The actual [`Resource`] is stored externally.
pub type ResourceId = SmolStr;

/// A resource is a set of capabilities required to execute a graph.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Resource {
    /// Unique identifier for the resource.
    pub name: ResourceId,
    /// Set of resource dependencies required by this resource.
    pub resource_reqs: ResourceSet,
    /// Types defined by this resource.
    pub types: Vec<CustomType>,
    /// Operation declarations with serializable definitions.
    pub operations: Vec<OpDef>,
    /// Opaque operation declarations that do not expose their definitions.
    pub opaque_operations: Vec<OpaqueOp>,
}

impl Resource {
    /// Creates a new resource with the given name.
    pub fn new(name: ResourceId) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    /// Returns the name of the resource.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add an exported type to the resource.
    pub fn add_type(&mut self, ty: CustomType) {
        self.types.push(ty);
    }

    /// Add an operation definition to the resource.
    pub fn add_op(&mut self, op: OpDef) {
        self.operations.push(op);
    }

    /// Add an opaque operation declaration to the resource.
    pub fn add_opaque_op(&mut self, op: OpaqueOp) {
        self.opaque_operations.push(op);
    }
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// A set of resources identified by their unique [`ResourceId`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResourceSet(HashSet<ResourceId>);

impl ResourceSet {
    /// Creates a new empty resource set.
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    /// Adds a resource to the set.
    pub fn insert(&mut self, resource: &ResourceId) {
        self.0.insert(resource.clone());
    }

    /// Returns `true` if the set contains the given resource.
    pub fn contains(&self, resource: &ResourceId) -> bool {
        self.0.contains(resource)
    }

    /// Returns `true` if the set is a subset of `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns `true` if the set is a superset of `other`.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    /// Create a resource set with a single element.
    pub fn singleton(resource: &ResourceId) -> Self {
        let mut set = Self::new();
        set.insert(resource);
        set
    }

    /// Returns the union of two resource sets.
    pub fn union(mut self, other: &Self) -> Self {
        self.0.extend(other.0.iter().cloned());
        self
    }

    /// The things in other which are in not in self
    pub fn missing_from(&self, other: &Self) -> Self {
        ResourceSet(HashSet::from_iter(other.0.difference(&self.0).cloned()))
    }
}

impl Display for ResourceSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl FromIterator<ResourceId> for ResourceSet {
    fn from_iter<I: IntoIterator<Item = ResourceId>>(iter: I) -> Self {
        Self(HashSet::from_iter(iter))
    }
}
