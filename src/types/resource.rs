use std::collections::HashSet;

/// A resource is a set of capabilities required to execute a graph.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Resource {
    resources: HashSet<ResourceValue>,
}

impl Resource {
    pub fn new() -> Self {
        Self {
            resources: HashSet::new(),
        }
    }

    pub fn add(&mut self, resource: ResourceValue) {
        self.resources.insert(resource);
    }

    pub fn contains(&self, resource: &ResourceValue) -> bool {
        self.resources.contains(resource)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ResourceValue> {
        self.resources.iter()
    }
}

/// A resource value is a capability required to execute a graph.
///
/// TODO: The user should be able to define new custom resources, used on their
/// operations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceValue {
    pub name: String,
}
