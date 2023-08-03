//! Resources
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use smol_str::SmolStr;
use thiserror::Error;

use crate::ops::custom::OpaqueOp;
use crate::types::type_param::{check_type_arg, TypeArgError};
use crate::types::CustomType;
use crate::types::TypeTag;
use crate::types::{
    type_param::{TypeArg, TypeParam},
    AbstractSignature, SignatureDescription, SimpleRow,
};
use crate::Hugr;

/// Trait for resources to provide custom binary code for computing signature.
pub trait CustomSignatureFunc: Send + Sync {
    /// Compute signature of node given the operation name,
    /// values for the type parameters,
    /// and 'misc' data from the resource definition YAML
    fn compute_signature(
        &self,
        name: &SmolStr,
        arg_values: &[TypeArg],
        misc: &HashMap<String, serde_yaml::Value>,
        // TODO: Make return type an AbstractSignature
    ) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError>;
    /// Describe the signature of a node, given the operation name,
    /// values for the type parameters,
    /// and 'misc' data from the resource definition YAML.
    fn describe_signature(
        &self,
        _name: &SmolStr,
        _arg_values: &[TypeArg],
        _misc: &HashMap<String, serde_yaml::Value>,
    ) -> SignatureDescription {
        SignatureDescription::default()
    }
}

impl<F> CustomSignatureFunc for F
where
    F: Fn(&[TypeArg]) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> + Send + Sync,
{
    fn compute_signature(
        &self,
        _name: &SmolStr,
        arg_values: &[TypeArg],
        _misc: &HashMap<String, serde_yaml::Value>,
    ) -> Result<(SimpleRow, SimpleRow, ResourceSet), SignatureError> {
        self(arg_values)
    }
}

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SignatureError {
    /// Name mismatch
    #[error("Definition name ({0}) and instantiation name ({1}) do not match.")]
    NameMismatch(SmolStr, SmolStr),
    /// Resource mismatch
    #[error("Definition resource ({0:?}) and instantiation resource ({1:?}) do not match.")]
    ResourceMismatch(Option<ResourceId>, Option<ResourceId>),
    /// When the type arguments of the node did not match the params declared by the OpDef
    #[error("Type arguments of node did not match params declared by definition: {0}")]
    TypeArgMismatch(#[from] TypeArgError),
}

/// Trait for Resources to provide custom binary code that can lower an operation to
/// a Hugr using only a limited set of other resources. That is, trait
/// implementations can return a Hugr that implements the operation using only
/// those resources and that can be used to replace the operation node. This may be
/// useful for third-party Resources or as a fallback for tools that do not support
/// the operation natively.
///
/// This trait allows the Hugr to be varied according to the operation's [TypeArg]s;
/// if this is not necessary then a single Hugr can be provided instead via
/// [LowerFunc::FixedHugr].
pub trait CustomLowerFunc: Send + Sync {
    /// Return a Hugr that implements the node using only the specified available resources;
    /// may fail.
    /// TODO: some error type to indicate Resources required?
    fn try_lower(
        &self,
        name: &SmolStr,
        arg_values: &[TypeArg],
        misc: &HashMap<String, serde_yaml::Value>,
        available_resources: &ResourceSet,
    ) -> Option<Hugr>;
}

/// The two ways in which an OpDef may compute the Signature of each operation node.
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

/// Different ways that an [OpDef] can lower operation nodes i.e. provide a Hugr
/// that implements the operation using a set of other resources.
#[derive(serde::Deserialize, serde::Serialize)]
pub enum LowerFunc {
    /// Lowering to a fixed Hugr. Since this cannot depend upon the [TypeArg]s,
    /// this will generally only be applicable if the [OpDef] has no [TypeParam]s.
    #[serde(rename = "hugr")]
    FixedHugr(ResourceSet, Hugr),
    /// Custom binary function that can (fallibly) compute a Hugr
    /// for the particular instance and set of available resources.
    #[serde(skip)]
    CustomFunc(Box<dyn CustomLowerFunc>),
}

impl Debug for LowerFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedHugr(_, _) => write!(f, "FixedHugr"),
            Self::CustomFunc(_) => write!(f, "<custom lower>"),
        }
    }
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
    /// The parent resource. if any.
    fn resource(&self) -> Option<&ResourceId>;
    /// Check provided type arguments are valid against parameters.
    fn check_args_impl(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        if args.len() != self.params().len() {
            return Err(SignatureError::TypeArgMismatch(TypeArgError::WrongNumber(
                args.len(),
                self.params().len(),
            )));
        }
        for (a, p) in args.iter().zip(self.params().iter()) {
            check_type_arg(a, p).map_err(SignatureError::TypeArgMismatch)?;
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
        if self.resource() != Some(custom.parent_resource()) {
            return Err(SignatureError::ResourceMismatch(
                self.resource().cloned(),
                Some(custom.parent_resource().clone()),
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

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new OpDef's from a serialized definition.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// The unique Resource, if any, owning this OpDef (of which this OpDef is a member)
    pub resource: Option<ResourceId>,
    /// Unique identifier of the operation. Used to look up OpDefs in the registry
    /// when deserializing nodes (which store only the name).
    pub name: SmolStr,
    /// Human readable description of the operation.
    pub description: String,
    /// Declared type parameters, values must be provided for each operation node
    pub params: Vec<TypeParam>,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub misc: HashMap<String, serde_yaml::Value>,

    #[serde(flatten)]
    signature_func: SignatureFunc,
    // Some operations cannot lower themselves and tools that do not understand them
    // can only treat them as opaque/black-box ops.
    #[serde(flatten)]
    lower_funcs: Vec<LowerFunc>,
}

impl OpDef {
    /// Check provided type arguments are valid against parameters.
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        self.check_args_impl(args)
    }

    /// Check [`OpaqueOp`] is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    pub fn check_opaque(&self, opaque: &OpaqueOp) -> Result<(), SignatureError> {
        self.check_concrete_impl(opaque)
    }

    /// Instantiate a concrete [`OpaqueOp`] by providing type arguments.
    ///
    /// # Errors
    ///
    /// This function will return an error if the provided arguments are not
    /// valid instances of the type parameters.
    pub fn instantiate_opaque(
        &self,
        args: impl Into<Vec<TypeArg>>,
    ) -> Result<OpaqueOp, SignatureError> {
        let args = args.into();
        self.check_args(&args)?;

        Ok(OpaqueOp::new(
            self.resource().expect("Resource not set.").clone(),
            self.name().clone(),
            // TODO add description
            "".to_string(),
            args,
            None,
        ))
    }
}

impl TypeParametrised for OpDef {
    type Concrete = OpaqueOp;

    fn params(&self) -> &[TypeParam] {
        &self.params
    }

    fn name(&self) -> &SmolStr {
        &self.name
    }

    fn resource(&self) -> Option<&ResourceId> {
        self.resource.as_ref()
    }
}

impl OpDef {
    /// Create an OpDef with a signature (inputs+outputs) read from the YAML
    pub fn new_with_yaml_types(
        name: SmolStr,
        description: String,
        params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        inputs: String, // TODO this is likely the wrong type
        outputs: String, // TODO similarly
                        // resources: Option<String> -- if mentioned in YAML?
    ) -> Self {
        Self {
            resource: Default::default(), // Currently overwritten when OpDef added to Resource
            name,
            description,
            params,
            misc,
            signature_func: SignatureFunc::FromYAML { inputs, outputs },
            lower_funcs: Vec::new(),
        }
    }

    /// Create an OpDef with custom binary code to compute the signature
    pub fn new_with_custom_sig(
        name: SmolStr,
        description: String,
        params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        sig_func: impl CustomSignatureFunc + 'static,
    ) -> Self {
        Self {
            resource: Default::default(), // Currently overwritten when OpDef added to Resource
            name,
            description,
            params,
            misc,
            signature_func: SignatureFunc::CustomFunc(Box::new(sig_func)),
            lower_funcs: Vec::new(),
        }
    }

    /// Provides a (new) way for the OpDef to fallibly lower operations. Each
    /// LowerFunc will be attempted in [Self::try_lower] only if previous methods failed.
    pub fn with_lowering(mut self, func: LowerFunc) {
        self.lower_funcs.push(func);
    }

    /// Computes the signature of a node, i.e. an instantiation of this
    /// OpDef with statically-provided [TypeArg]s.
    pub fn compute_signature(&self, args: &[TypeArg]) -> Result<AbstractSignature, SignatureError> {
        self.check_args(args)?;
        let (ins, outs, res) = match &self.signature_func {
            SignatureFunc::FromYAML { .. } => {
                // Sig should be computed solely from inputs + outputs + args.
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.compute_signature(&self.name, args, &self.misc)?,
        };
        let resource = self
            .resource
            .as_ref()
            .expect("OpDef does not belong to a Resource.");
        assert!(res.contains(resource));
        Ok(AbstractSignature::new_df(ins, outs).with_resource_delta(&res))
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self, args: &[TypeArg]) -> SignatureDescription {
        match &self.signature_func {
            SignatureFunc::FromYAML { .. } => {
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.describe_signature(&self.name, args, &self.misc),
        }
    }

    pub(crate) fn should_serialize_signature(&self) -> bool {
        match self.signature_func {
            SignatureFunc::CustomFunc(_) => true,
            SignatureFunc::FromYAML { .. } => false,
        }
    }

    /// Fallibly returns a Hugr that may replace an instance of this OpDef
    /// given a set of available resources that may be used in the Hugr.
    pub fn try_lower(&self, args: &[TypeArg], available_resources: &ResourceSet) -> Option<Hugr> {
        self.lower_funcs
            .iter()
            .flat_map(|f| match f {
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
            })
            .next()
    }
}

/// The type tag of a [`TypeDef`]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeDefTag {
    /// Defined by an explicit tag.
    Explicit(TypeTag),
    /// Derived as the tag containing all marked type parameters.
    FromParams(Vec<usize>),
}

impl From<TypeTag> for TypeDefTag {
    fn from(tag: TypeTag) -> Self {
        Self::Explicit(tag)
    }
}
/// A declaration of an opaque type.
/// Note this does not provide any way to create instances
/// - typically these are operations also provided by the Resource.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TypeDef {
    /// The unique name of the type
    pub name: SmolStr,
    /// Declaration of type parameters. The TypeDef must be instantiated
    /// with the same number of [`TypeArg`]'s to make an actual type.
    ///
    /// [`TypeArg`]: crate::types::type_param::TypeArg
    pub params: Vec<TypeParam>,
    /// The unique Resource, if any, owning this TypeDef (of which this TypeDef is a member)
    pub resource: Option<ResourceId>,
    /// Human readable description of the type definition.
    pub description: String,
    /// The definition of the type tag of this definition.
    pub tag: TypeDefTag,
}

impl TypeDef {
    /// Check provided type arguments are valid against parameters.
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        self.check_args_impl(args)
    }

    /// Check [`CustomType`] is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    pub fn check_custom(&self, custom: &CustomType) -> Result<(), SignatureError> {
        self.check_concrete_impl(custom)
    }

    /// Instantiate a concrete [`CustomType`] by providing type arguments.
    ///
    /// # Errors
    ///
    /// This function will return an error if the provided arguments are not
    /// valid instances of the type parameters.
    pub fn instantiate_concrete(
        &self,
        args: impl Into<Vec<TypeArg>>,
    ) -> Result<CustomType, SignatureError> {
        let args = args.into();
        self.check_args_impl(&args)?;
        let tag = self.tag(&args);
        Ok(CustomType::new(
            self.name().clone(),
            args,
            self.resource().expect("Resource not set.").clone(),
            tag,
        ))
    }
}

impl TypeParametrised for TypeDef {
    type Concrete = CustomType;

    fn params(&self) -> &[TypeParam] {
        &self.params
    }

    fn name(&self) -> &SmolStr {
        &self.name
    }

    fn resource(&self) -> Option<&ResourceId> {
        self.resource.as_ref()
    }
}

impl TypeDef {
    /// The [`TypeTag`] of the definition.
    pub fn tag(&self, args: &[TypeArg]) -> TypeTag {
        match &self.tag {
            TypeDefTag::Explicit(tag) => *tag,
            TypeDefTag::FromParams(indices) => {
                let args: Vec<_> = args.iter().collect();
                if indices.is_empty() {
                    // Assume most general case
                    return TypeTag::Simple;
                }
                indices
                    .iter()
                    .map(|i| {
                        args.get(*i)
                            .and_then(|ta| ta.tag_of_type())
                            .expect("TypeParam index invalid or param does not have a TypeTag.")
                    })
                    .fold(TypeTag::Hashable, TypeTag::union)
            }
        }
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
    types: HashMap<SmolStr, TypeDef>,
    /// Operation declarations with serializable definitions.
    // Note: serde will serialize this because we configure with `features=["rc"]`.
    // That will clone anything that has multiple references, but each
    // OpDef should appear exactly once in this map (keyed by its name),
    // and the other references to the OpDef are from ExternalOp's in the Hugr
    // (which are serialized as OpaqueOp's i.e. Strings).
    operations: HashMap<SmolStr, Arc<OpDef>>,
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
    pub fn operations(&self) -> &HashMap<SmolStr, Arc<OpDef>> {
        &self.operations
    }

    /// Allows read-only access to the types in this Resource
    pub fn types(&self) -> &HashMap<SmolStr, TypeDef> {
        &self.types
    }

    /// Returns the name of the resource.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add an exported type to the resource.
    pub fn add_type(&mut self, mut ty: TypeDef) -> Result<(), String> {
        if let Some(resource) = ty.resource {
            return Err(format!(
                "TypeDef {} owned by another resource {}",
                ty.name, resource
            ));
        }
        match self.types.entry(ty.name.clone()) {
            Entry::Occupied(_) => panic!("Resource already has a type called {}", &ty.name),
            Entry::Vacant(ve) => {
                ty.resource = Some(self.name.clone());
                ve.insert(ty);
            }
        }
        Ok(())
    }

    /// Add an operation definition to the resource.
    pub fn add_op(&mut self, mut op: OpDef) -> Result<(), String> {
        if let Some(resource) = op.resource {
            return Err(format!(
                "OpDef {} owned by another resource {}",
                op.name, resource
            ));
        }
        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => panic!("Resource already has an op called {}", &op.name),
            Entry::Vacant(ve) => {
                op.resource = Some(self.name.clone());
                ve.insert(Arc::new(op));
            }
        }
        Ok(())
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
