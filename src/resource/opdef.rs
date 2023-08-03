use crate::Hugr;
use std::collections::hash_map::Entry;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use super::{
    Resource, ResourceBuildError, ResourceId, ResourceSet, SignatureError, TypeParametrised,
};

use crate::types::{SignatureDescription, SimpleRow};

use crate::types::AbstractSignature;

use crate::types::type_param::TypeArg;

use crate::ops::custom::OpaqueOp;

use std::collections::HashMap;

use crate::types::type_param::TypeParam;

use smol_str::SmolStr;

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
pub(super) enum SignatureFunc {
    // Note: I'd prefer to make the YAML version just implement the same CustomSignatureFunc trait,
    // and then just have a Box<dyn CustomSignatureFunc> instead of this enum, but that seems less likely
    // to serialize well.
    /// TODO: these types need to be whatever representation we want of a type scheme encoded in the YAML
    #[serde(rename = "signature")]
    FromDecl { inputs: String, outputs: String },
    #[serde(skip)]
    CustomFunc(Box<dyn CustomSignatureFunc>),
}

impl Debug for SignatureFunc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FromDecl { inputs, outputs } => f
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

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new OpDef's from a serialized definition.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// The unique Resource, if any, owning this OpDef (of which this OpDef is a member)
    resource: ResourceId,
    /// Unique identifier of the operation. Used to look up OpDefs in the registry
    /// when deserializing nodes (which store only the name).
    name: SmolStr,
    /// Human readable description of the operation.
    description: String,
    /// Declared type parameters, values must be provided for each operation node
    params: Vec<TypeParam>,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    misc: HashMap<String, serde_yaml::Value>,

    #[serde(flatten)]
    signature_func: SignatureFunc,
    // Some operations cannot lower themselves and tools that do not understand them
    // can only treat them as opaque/black-box ops.
    #[serde(flatten)]
    lower_funcs: Vec<LowerFunc>,
}

impl TypeParametrised for OpDef {
    type Concrete = OpaqueOp;

    fn params(&self) -> &[TypeParam] {
        self.params()
    }

    fn name(&self) -> &SmolStr {
        self.name()
    }

    fn resource(&self) -> &ResourceId {
        self.resource()
    }
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
            self.resource().clone(),
            self.name().clone(),
            // TODO add description
            "".to_string(),
            args,
            None,
        ))
    }

    /// Computes the signature of a node, i.e. an instantiation of this
    /// OpDef with statically-provided [TypeArg]s.
    pub fn compute_signature(&self, args: &[TypeArg]) -> Result<AbstractSignature, SignatureError> {
        self.check_args(args)?;
        let (ins, outs, res) = match &self.signature_func {
            SignatureFunc::FromDecl { .. } => {
                // Sig should be computed solely from inputs + outputs + args.
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.compute_signature(&self.name, args, &self.misc)?,
        };
        assert!(res.contains(self.resource()));
        Ok(AbstractSignature::new_df(ins, outs).with_resource_delta(&res))
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self, args: &[TypeArg]) -> SignatureDescription {
        match &self.signature_func {
            SignatureFunc::FromDecl { .. } => {
                todo!()
            }
            SignatureFunc::CustomFunc(bf) => bf.describe_signature(&self.name, args, &self.misc),
        }
    }

    pub(crate) fn should_serialize_signature(&self) -> bool {
        match self.signature_func {
            SignatureFunc::CustomFunc(_) => true,
            SignatureFunc::FromDecl { .. } => false,
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

    /// Returns a reference to the name of this [`OpDef`].
    pub fn name(&self) -> &SmolStr {
        &self.name
    }

    /// Returns a reference to the resource of this [`OpDef`].
    pub fn resource(&self) -> &ResourceId {
        &self.resource
    }

    /// Returns a reference to the description of this [`OpDef`].
    pub fn description(&self) -> &str {
        self.description.as_ref()
    }

    /// Returns a reference to the params of this [`OpDef`].
    pub fn params(&self) -> &[TypeParam] {
        self.params.as_ref()
    }
}

impl Resource {
    /// Add an operation definition to the resource.
    fn add_op(
        &mut self,
        name: SmolStr,
        description: String,
        params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        signature_func: SignatureFunc,
    ) -> Result<&OpDef, ResourceBuildError> {
        let op = OpDef {
            resource: self.name.clone(),
            name,
            description,
            params,
            misc,
            signature_func,
            lower_funcs,
        };

        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => Err(ResourceBuildError::OpDefExists(op.name)),
            Entry::Vacant(ve) => Ok(ve.insert(Arc::new(op))),
        }
    }
    /// Create an OpDef with custom binary code to compute the signature
    pub fn add_op_custom_sig(
        &mut self,
        name: SmolStr,
        description: String,
        params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        signature_func: impl CustomSignatureFunc + 'static,
    ) -> Result<&OpDef, ResourceBuildError> {
        self.add_op(
            name,
            description,
            params,
            misc,
            lower_funcs,
            SignatureFunc::CustomFunc(Box::new(signature_func)),
        )
    }

    /// Create an OpDef with a signature (inputs+outputs) read from the
    /// declarative YAML
    pub fn add_op_decl_sig(
        &mut self,
        name: SmolStr,
        description: String,
        params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        (inputs, outputs): (String, String), // separating these makes clippy complain about too many args
    ) -> Result<&OpDef, ResourceBuildError> {
        self.add_op(
            name,
            description,
            params,
            misc,
            lower_funcs,
            SignatureFunc::FromDecl { inputs, outputs },
        )
    }
}
