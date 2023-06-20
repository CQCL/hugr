//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};

use smol_str::SmolStr;

use crate::ops::OpaqueOp;
use crate::types::{custom::CustomType, Signature, SignatureDescription, SimpleType};

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new CustomOps from a serialized definition.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// Unique identifier of the operation.
    ///
    /// This is used to compare two custom ops for equality.
    pub name: SmolStr,
    /// Human readable description of the operation.
    pub description: String,
    inputs: Vec<(Option<SmolStr>, SimpleType)>,
    outputs: Vec<(Option<SmolStr>, SimpleType)>,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub misc: HashMap<String, serde_yaml::Value>,
    /// (YAML?)-encoded definition of the operation.
    ///
    /// TODO: Define the format of this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub def: Option<String>,
    /// Resources required to execute this operation.
    pub resource_reqs: ResourceSet,

    /// Signature of the operation.
    ///
    /// Computed from the serialized `inputs` and `outputs`.
    #[serde(skip)]
    signature: OnceCell<Signature>,
    /// Optional port descriptions.
    ///
    /// Computed from the serialized `inputs` and `outputs`.
    #[serde(skip)]
    port_names: OnceCell<SignatureDescription>,
}

impl OpDef {
    /// Initialize a new operation definition with a fixed signature.
    pub fn new(name: SmolStr, signature: Signature) -> Self {
        Self::new_with_description(
            name,
            String::new(),
            signature,
            SignatureDescription::default(),
        )
    }

    /// Initialize a new operation definition with a fixed signature.
    pub fn new_with_description(
        name: SmolStr,
        description: String,
        signature: Signature,
        port_names: SignatureDescription,
    ) -> Self {
        let inputs: Vec<_> = port_names
            .input_zip(&signature)
            .chain(port_names.const_input_zip(&signature))
            .map(|(n, t)| (Some(n.clone()), t.clone()))
            .collect();

        let outputs = port_names
            .output_zip(&signature)
            .map(|(n, t)| (Some(n.clone()), t.clone()));
        Self {
            name,
            description,
            inputs,
            outputs: outputs.collect(),
            misc: HashMap::new(),
            def: None,
            resource_reqs: ResourceSet::new(),
            signature: OnceCell::from(signature),
            port_names: OnceCell::from(port_names),
        }
    }

    /// The signature of the operation.
    pub fn signature(&self) -> Signature {
        self.signature
            .get_or_init(|| {
                let inputs = self
                    .inputs
                    .iter()
                    .map(|(_, t)| t.clone())
                    .collect::<Vec<_>>();
                let outputs = self
                    .outputs
                    .iter()
                    .map(|(_, t)| t.clone())
                    .collect::<Vec<_>>();
                Signature::new_df(inputs, outputs)
            })
            .clone()
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self) -> Option<SignatureDescription> {
        Some(
            self.port_names
                .get_or_init(|| {
                    let inputs = self
                        .inputs
                        .iter()
                        .map(|(n, _)| n.clone().unwrap_or_default())
                        .collect::<Vec<_>>();
                    let outputs = self
                        .outputs
                        .iter()
                        .map(|(n, _)| n.clone().unwrap_or_default())
                        .collect::<Vec<_>>();
                    SignatureDescription::new_df(inputs, outputs)
                })
                .clone(),
        )
    }
}

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
