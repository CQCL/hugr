//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.
//!
//! [`OpDef`]: crate::ops::custom::OpDef

use std::collections::HashSet;

use smol_str::SmolStr;

use crate::ops::{OpDef, OpaqueOp};
use crate::types::custom::CustomType;

/// A unique identifier for a resource.
///
/// The actual [`Resource`] is stored externally.
pub type ResourceId = SmolStr;

/// A resource is a set of capabilities required to execute a graph.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Resource {
    pub name: ResourceId,
    pub resource_reqs: ResourceSet,
    pub types: Vec<CustomType>,
    /// Operations with serializable definitions.
    pub operations: Vec<OpDef>,
    /// Opaque operations that do not expose their definitions.
    pub opaque_operations: Vec<OpaqueOp>,
}

impl Resource {
    pub fn new(name: ResourceId) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn add_type(&mut self, ty: CustomType) {
        self.types.push(ty);
    }

    pub fn add_op(&mut self, op: OpDef) {
        self.operations.push(op);
    }

    pub fn add_opaque_op(&mut self, op: OpaqueOp) {
        self.opaque_operations.push(op);
    }
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// A set of resources.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResourceSet(HashSet<ResourceId>);

impl ResourceSet {
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    pub fn insert(&mut self, resource: &Resource) {
        self.0.insert(resource.name.clone());
    }

    pub fn contains(&self, resource: &Resource) -> bool {
        self.0.contains(&resource.name)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    pub fn singleton(resource: &Resource) -> Self {
        let mut set = Self::new();
        set.insert(resource);
        set
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut set = self.clone();
        set.0.extend(other.0.iter().cloned());
        set
    }
}

impl FromIterator<ResourceId> for ResourceSet {
    fn from_iter<I: IntoIterator<Item = ResourceId>>(iter: I) -> Self {
        Self(HashSet::from_iter(iter))
    }
}
