//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use smol_str::SmolStr;
use thiserror::Error;

use crate::ops::custom::OpaqueOp;
use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
use crate::types::CustomType;

mod opdef;
pub use opdef::{CustomSignatureFunc, OpDef};
mod type_def;
pub use type_def::{TypeDef, TypeDefTag};

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SignatureError {
    /// Name mismatch
    #[error("Definition name ({0}) and instantiation name ({1}) do not match.")]
    NameMismatch(SmolStr, SmolStr),
    /// Resource mismatch
    #[error("Definition resource ({0:?}) and instantiation resource ({1:?}) do not match.")]
    ResourceMismatch(ResourceId, ResourceId),
    /// When the type arguments of the node did not match the params declared by the OpDef
    #[error("Type arguments of node did not match params declared by definition: {0}")]
    TypeArgMismatch(#[from] TypeArgError),
}

/// Concrete instantiations of types and operations defined in resources.
trait CustomConcrete {
    fn def_name(&self) -> &SmolStr;
    fn type_args(&self) -> &[TypeArg];
    fn parent_resource(&self) -> &ResourceId;
}

impl CustomConcrete for OpaqueOp {
    fn def_name(&self) -> &SmolStr {
        self.name()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_resource(&self) -> &ResourceId {
        self.resource()
    }
}

impl CustomConcrete for CustomType {
    fn def_name(&self) -> &SmolStr {
        self.name()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_resource(&self) -> &ResourceId {
        self.resource()
    }
}

/// Type-parametrised functionality shared between [`TypeDef`] and [`OpDef`].
trait TypeParametrised {
    /// The concrete object built by binding type arguments to parameters
    type Concrete: CustomConcrete;
    /// The resource-unique name.
    fn name(&self) -> &SmolStr;
    /// Type parameters.
    fn params(&self) -> &[TypeParam];
    /// The parent resource.
    fn resource(&self) -> &ResourceId;
    /// Check provided type arguments are valid against parameters.
    fn check_args_impl(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        if args.len() != self.params().len() {
            return Err(SignatureError::TypeArgMismatch(TypeArgError::WrongNumber(
                args.len(),
                self.params().len(),
            )));
        }
        for (a, p) in args.iter().zip(self.params().iter()) {
            a.check_type(p).map_err(SignatureError::TypeArgMismatch)?;
        }
        Ok(())
    }

    /// Check custom instance is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    fn check_concrete_impl(&self, custom: &Self::Concrete) -> Result<(), SignatureError> {
        if self.resource() != custom.parent_resource() {
            return Err(SignatureError::ResourceMismatch(
                self.resource().clone(),
                custom.parent_resource().clone(),
            ));
        }
        if self.name() != custom.def_name() {
            return Err(SignatureError::NameMismatch(
                self.name().clone(),
                custom.def_name().clone(),
            ));
        }

        self.check_args_impl(custom.type_args())?;

        Ok(())
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
    /// Other resources defining types used by this resource.
    /// That is, an upper-bound on the types that can be returned by
    /// computing the signature of any operation in this resource,
    /// for any possible [TypeArg].
    pub resource_reqs: ResourceSet,
    /// Types defined by this resource.
    types: HashMap<SmolStr, type_def::TypeDef>,
    /// Operation declarations with serializable definitions.
    // Note: serde will serialize this because we configure with `features=["rc"]`.
    // That will clone anything that has multiple references, but each
    // OpDef should appear exactly once in this map (keyed by its name),
    // and the other references to the OpDef are from ExternalOp's in the Hugr
    // (which are serialized as OpaqueOp's i.e. Strings).
    operations: HashMap<SmolStr, Arc<opdef::OpDef>>,
}

impl Resource {
    /// Creates a new resource with the given name.
    pub fn new(name: ResourceId) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    /// Allows read-only access to the operations in this Resource
    pub fn get_op(&self, op_name: &str) -> Option<&Arc<opdef::OpDef>> {
        self.operations.get(op_name)
    }

    /// Allows read-only access to the types in this Resource
    pub fn get_type(&self, type_name: &str) -> Option<&type_def::TypeDef> {
        self.types.get(type_name)
    }

    /// Returns the name of the resource.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Iterator over the operations of this [`Resource`].
    pub fn operations(&self) -> impl Iterator<Item = (&SmolStr, &Arc<OpDef>)> {
        self.operations.iter()
    }

    /// Iterator over the types of this [`Resource`].
    pub fn types(&self) -> impl Iterator<Item = (&SmolStr, &TypeDef)> {
        self.types.iter()
    }
}

impl PartialEq for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ResourceBuildError {
    /// Existing [`OpDef`]
    #[error("Resource already has an op called {0}.")]
    OpDefExists(SmolStr),
    /// Existing [`TypeDef`]
    #[error("Resource already has an type called {0}.")]
    TypeDefExists(SmolStr),
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

    /// Iterate over the contained ResourceIds
    pub fn iter(&self) -> impl Iterator<Item = &ResourceId> {
        self.0.iter()
    }
}

impl Display for ResourceSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl FromIterator<ResourceId> for ResourceSet {
    fn from_iter<I: IntoIterator<Item = ResourceId>>(iter: I) -> Self {
        Self(HashSet::from_iter(iter))
    }
}
