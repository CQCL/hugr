//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};

use smol_str::SmolStr;
use thiserror::Error;

use crate::types::type_arg::check_arg;
use crate::types::TypeRow;
use crate::types::{
    custom::CustomType,
    type_arg::{TypeArgValue, TypeParam},
    Signature, SignatureDescription,
};
use crate::Hugr;

/// Trait for resources to provide custom binary code for computing signature.
pub trait CustomSignatureFunc {
    /// Compute signature of node given the operation name,
    /// values for the type parameters,
    /// and 'misc' data from the resource definition YAML
    fn compute_signature(
        &self,
        name: &SmolStr,
        arg_values: &Vec<TypeArgValue>,
        misc: &HashMap<String, serde_yaml::Value>,
    ) -> Result<(TypeRow, TypeRow, ResourceSet), SignatureError>;

    /// Describe the signature of a node, given the operation name,
    /// values for the type parameters,
    /// and 'misc' data from the resource definition YAML.
    fn describe_signature(
        &self,
        _name: &SmolStr,
        _arg_values: &Vec<TypeArgValue>,
        _misc: &HashMap<String, serde_yaml::Value>,
    ) -> SignatureDescription {
        SignatureDescription::default()
    }
}

/// An error that can occur in computing the signature of a node.
/// TODO: there's no real way to plumb this out of OpType::signature()...
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SignatureError {
    /// When the type arguments of the node did not match the params declared by the OpDef
    #[error("Type arguments of node did not match params declared by OpDef: {0}")]
    TypeArgMismatch(String),
}

/// Trait for resources to provide custom binary code for lowering a node to a Hugr.
pub trait CustomLowerFunc {
    /// Return a Hugr that implements the node using only the specified available resources;
    /// may fail.
    /// TODO: some error type to indicate Resources required?
    fn try_lower(
        &self,
        name: &SmolStr,
        arg_values: &Vec<TypeArgValue>,
        misc: &HashMap<String, serde_yaml::Value>,
        available_resources: &ResourceSet,
    ) -> Option<Hugr>;
}

#[derive(serde::Deserialize, serde::Serialize)]
enum SignatureFunc {
    // Note: I'd prefer to make the YAML version just implement the same CustomSignatureFunc trait,
    // and then just have a Box<dyn CustomSignatureFunc> instead of this enum, but that seems less likely
    // to serialize well.
    /// TODO: these types need to be whatever representation we want of a type scheme encoded in the YAML
    #[serde(rename = "signature")]
    FromYAML { inputs: String, outputs: String },
    #[serde(skip)]
    CustomFunc(Box<dyn CustomSignatureFunc>),
}

impl Debug for SignatureFunc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FromYAML { inputs, outputs } => f
                .debug_struct("signature")
                .field("inputs", inputs)
                .field("outputs", outputs)
                .finish(),
            Self::CustomFunc(_) => f.write_str("<custom sig>"),
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
enum LowerFunc {
    #[serde(skip)]
    None,
    #[serde(rename = "hugr")]
    FixedHugr(ResourceSet, Hugr),
    #[serde(skip)]
    CustomFunc(Box<dyn CustomLowerFunc>),
}

impl Debug for LowerFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::FixedHugr(_, _) => write!(f, "FixedHugr"),
            Self::CustomFunc(_) => write!(f, "<custom lower>"),
        }
    }
}

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new OpDef's from a serialized definition.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// Unique identifier of the operation. Used to look up OpDefs in the registry
    /// when deserializing nodes (which store only the name).
    pub name: SmolStr,
    /// Human readable description of the operation.
    pub description: String,
    /// Declared type parameters, values must be provided for each operation node
    pub args: Vec<TypeParam>,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub misc: HashMap<String, serde_yaml::Value>,

    // TODO YAML directive: only serialize "FromYAML" version, skip CustomFunc
    #[serde(flatten)]
    signature_func: SignatureFunc,
    #[serde(flatten)]
    lower_func: LowerFunc,
}

impl OpDef {
    /// Create an OpDef with a signature (inputs+outputs) read from the YAML
    pub fn new_with_yaml_types(
        name: SmolStr,
        description: String,
        args: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        inputs: String, // TODO this is likely the wrong type
        outputs: String, // TODO similarly
                        // resources: Option<String> -- if mentioned in YAML?
    ) -> Self {
        Self {
            name,
            description,
            args,
            misc,
            signature_func: SignatureFunc::FromYAML { inputs, outputs },
            lower_func: LowerFunc::None,
        }
    }

    /// Create an OpDef with custom binary code to compute the signature
    pub fn new_with_custom_sig(
        name: SmolStr,
        description: String,
        args: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        sig_func: impl CustomSignatureFunc + 'static,
    ) -> Self {
        Self {
            name,
            description,
            args,
            misc,
            signature_func: SignatureFunc::CustomFunc(Box::new(sig_func)),
            lower_func: LowerFunc::None,
        }
    }

    /// Modifies the OpDef with the ability to lower every operation to a
    /// fixed Hugr. Only applicable if the OpDef cannot currently lower itself.
    pub fn lowering_to_hugr(
        mut self,
        h: Hugr,
        required_resources: ResourceSet, // TODO can we figure these out from 'h' ?
    ) -> Result<Self, Self> {
        if let LowerFunc::None = self.lower_func {
            self.lower_func = LowerFunc::FixedHugr(required_resources, h);
            Ok(self)
        } else {
            Err(self)
        }
    }

    /// Add custom binary code that may try to lower the operation.
    /// Only applicable if the OpDef currently has no way to lower itself.
    pub fn with_custom_lower_func<F: CustomLowerFunc + 'static>(
        mut self,
        func: F,
    ) -> Result<Self, Self> {
        if let LowerFunc::None = self.lower_func {
            self.lower_func = LowerFunc::CustomFunc(Box::new(func));
            Ok(self)
        } else {
            Err(self)
        }
    }
    /// The signature of the operation.
    pub fn signature(
        &self,
        args: &Vec<TypeArgValue>,
        resources_in: &ResourceSet,
    ) -> Result<Signature, SignatureError> {
        if args.len() != self.args.len() {
            return Err(SignatureError::TypeArgMismatch(
                "Node provided wrong number of args".to_string(),
            ));
        }
        for (a, p) in args.iter().zip(self.args.iter()) {
            check_arg(a, p).map_err(SignatureError::TypeArgMismatch)?;
        }
        let (ins, outs, res) = match &self.signature_func {
            SignatureFunc::FromYAML { .. } => {
                // Sig should be computed solely from inputs + outputs + args.
                // Resources used should be at least the resource containing this OpDef.
                // (TODO Consider - should we identify that _in_ the OpDef?)
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.compute_signature(&self.name, args, &self.misc)?,
        };
        let mut sig = Signature::new_df(ins, outs);
        sig.input_resources = resources_in.clone();
        sig.output_resources = res.union(resources_in); // Pass input requirements through
        Ok(sig)
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self, args: &Vec<TypeArgValue>) -> SignatureDescription {
        match &self.signature_func {
            SignatureFunc::FromYAML { .. } => {
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.describe_signature(&self.name, args, &self.misc),
        }
    }

    /// Fallibly returns a Hugr that may replace an instance of this OpDef
    /// given a set of available resources that may be used in the Hugr.
    pub fn try_lower(
        &self,
        args: &Vec<TypeArgValue>,
        available_resources: &ResourceSet,
    ) -> Option<Hugr> {
        match &self.lower_func {
            LowerFunc::None => None,
            LowerFunc::FixedHugr(req_res, h) => {
                if available_resources.is_superset(req_res) {
                    Some(h.clone())
                } else {
                    None
                }
            }
            LowerFunc::CustomFunc(f) => {
                f.try_lower(&self.name, args, &self.misc, available_resources)
            }
        }
    }
}

/// A unique identifier for a resource.
///
/// The actual [`Resource`] is stored externally.
pub type ResourceId = SmolStr;

/// A resource is a set of capabilities required to execute a graph.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Resource {
    /// Unique identifier for the resource.
    pub name: ResourceId,
    // Set of resource dependencies required by this resource.
    // TODO I haven't seen where these are required yet. If they are,
    // we'll probably need some way to
    // pub resource_reqs: ResourceSet,
    /// Types defined by this resource.
    pub types: Vec<CustomType>,
    /// Operation declarations with serializable definitions.
    pub operations: Vec<OpDef>,
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
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl FromIterator<ResourceId> for ResourceSet {
    fn from_iter<I: IntoIterator<Item = ResourceId>>(iter: I) -> Self {
        Self(HashSet::from_iter(iter))
    }
}
