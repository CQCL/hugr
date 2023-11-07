use std::cmp::min;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use smol_str::SmolStr;

use super::{
    Extension, ExtensionBuildError, ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError,
    TypeParametrised,
};

use crate::ops::custom::OpaqueOp;
use crate::types::type_param::{check_type_args, TypeArg, TypeParam};
use crate::types::{FunctionType, PolyFuncType};
use crate::Hugr;

/// Trait for extensions to provide custom binary code for computing signature.
pub trait CustomSignatureFunc: Send + Sync {
    /// Compute signature of node given the operation name,
    /// values for the type parameters,
    /// and 'misc' data from the extension definition YAML
    fn compute_signature(
        &self,
        name: &SmolStr,
        arg_values: &[TypeArg],
        misc: &HashMap<String, serde_yaml::Value>,
        extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError>;
}

// Note this is very much a utility, rather than definitive;
// one can only do so much without the ExtensionRegistry!
impl<F, R: Into<PolyFuncType>> CustomSignatureFunc for F
where
    F: Fn(&[TypeArg]) -> Result<R, SignatureError> + Send + Sync,
{
    fn compute_signature(
        &self,
        _name: &SmolStr,
        arg_values: &[TypeArg],
        _misc: &HashMap<String, serde_yaml::Value>,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError> {
        Ok(self(arg_values)?.into())
    }
}

/// Trait for Extensions to provide custom binary code that can lower an operation to
/// a Hugr using only a limited set of other extensions. That is, trait
/// implementations can return a Hugr that implements the operation using only
/// those extensions and that can be used to replace the operation node. This may be
/// useful for third-party Extensions or as a fallback for tools that do not support
/// the operation natively.
///
/// This trait allows the Hugr to be varied according to the operation's [TypeArg]s;
/// if this is not necessary then a single Hugr can be provided instead via
/// [LowerFunc::FixedHugr].
pub trait CustomLowerFunc: Send + Sync {
    /// Return a Hugr that implements the node using only the specified available extensions;
    /// may fail.
    /// TODO: some error type to indicate Extensions required?
    fn try_lower(
        &self,
        name: &SmolStr,
        arg_values: &[TypeArg],
        misc: &HashMap<String, serde_yaml::Value>,
        available_extensions: &ExtensionSet,
    ) -> Option<Hugr>;
}

/// The two ways in which an OpDef may compute the Signature of each operation node.
#[derive(serde::Deserialize, serde::Serialize)]
pub(super) enum SignatureFunc {
    // Note: except for serialization, we could have type schemes just implement the same
    // CustomSignatureFunc trait too, and replace this enum with Box<dyn CustomSignatureFunc>.
    // However instead we treat all CustomFunc's as non-serializable.
    #[serde(rename = "signature")]
    TypeScheme(PolyFuncType),
    #[serde(skip)]
    CustomFunc {
        /// Type parameters passed to [func]. (The returned [PolyFuncType]
        /// may require further type parameters, not declared here.)
        static_params: Vec<TypeParam>,
        func: Box<dyn CustomSignatureFunc>,
    },
}

impl Debug for SignatureFunc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeScheme(scheme) => scheme.fmt(f),
            Self::CustomFunc { .. } => f.write_str("<custom sig>"),
        }
    }
}

/// Different ways that an [OpDef] can lower operation nodes i.e. provide a Hugr
/// that implements the operation using a set of other extensions.
#[derive(serde::Deserialize, serde::Serialize)]
pub enum LowerFunc {
    /// Lowering to a fixed Hugr. Since this cannot depend upon the [TypeArg]s,
    /// this will generally only be applicable if the [OpDef] has no [TypeParam]s.
    #[serde(rename = "hugr")]
    FixedHugr(ExtensionSet, Hugr),
    /// Custom binary function that can (fallibly) compute a Hugr
    /// for the particular instance and set of available extensions.
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
    /// The unique Extension owning this OpDef (of which this OpDef is a member)
    extension: ExtensionId,
    /// Unique identifier of the operation. Used to look up OpDefs in the registry
    /// when deserializing nodes (which store only the name).
    name: SmolStr,
    /// Human readable description of the operation.
    description: String,
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

    fn extension(&self) -> &ExtensionId {
        self.extension()
    }
}

impl OpDef {
    /// Check provided type arguments are valid against parameters.
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        self.check_args_impl(args)
    }

    /// Computes the signature of a node, i.e. an instantiation of this
    /// OpDef with statically-provided [TypeArg]s.
    pub fn compute_signature(
        &self,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
    ) -> Result<FunctionType, SignatureError> {
        // Hugr's are monomorphic, so check the args have no free variables
        args.iter().try_for_each(|ta| ta.validate(exts, &[]))?;

        let temp: PolyFuncType; // to keep alive
        let (pf, args) = match &self.signature_func {
            SignatureFunc::TypeScheme(ts) => (ts, args),
            SignatureFunc::CustomFunc {
                static_params,
                func,
            } => {
                let (static_args, other_args) = args.split_at(min(static_params.len(), args.len()));
                check_type_args(static_args, static_params)?;
                temp = func.compute_signature(&self.name, static_args, &self.misc, exts)?;
                (&temp, other_args)
            }
        };

        let res = pf.instantiate(args, exts)?;
        // TODO bring this assert back once resource inference is done?
        // https://github.com/CQCL-DEV/hugr/issues/425
        // assert!(res.contains(self.extension()));
        Ok(res)
    }

    pub(crate) fn should_serialize_signature(&self) -> bool {
        match self.signature_func {
            SignatureFunc::TypeScheme { .. } => true,
            SignatureFunc::CustomFunc { .. } => false,
        }
    }

    /// Fallibly returns a Hugr that may replace an instance of this OpDef
    /// given a set of available extensions that may be used in the Hugr.
    pub fn try_lower(&self, args: &[TypeArg], available_extensions: &ExtensionSet) -> Option<Hugr> {
        self.lower_funcs
            .iter()
            .flat_map(|f| match f {
                LowerFunc::FixedHugr(req_res, h) => {
                    if available_extensions.is_superset(req_res) {
                        Some(h.clone())
                    } else {
                        None
                    }
                }
                LowerFunc::CustomFunc(f) => {
                    f.try_lower(&self.name, args, &self.misc, available_extensions)
                }
            })
            .next()
    }

    /// Returns a reference to the name of this [`OpDef`].
    pub fn name(&self) -> &SmolStr {
        &self.name
    }

    /// Returns a reference to the extension of this [`OpDef`].
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }

    /// Returns a reference to the description of this [`OpDef`].
    pub fn description(&self) -> &str {
        self.description.as_ref()
    }

    /// Returns a reference to the params of this [`OpDef`].
    pub fn params(&self) -> &[TypeParam] {
        match &self.signature_func {
            SignatureFunc::TypeScheme(ts) => ts.params(),
            SignatureFunc::CustomFunc { static_params, .. } => static_params,
        }
    }
}

impl Extension {
    /// Add an operation definition to the extension.
    fn add_op(
        &mut self,
        name: SmolStr,
        description: String,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        signature_func: SignatureFunc,
    ) -> Result<&OpDef, ExtensionBuildError> {
        let op = OpDef {
            extension: self.name.clone(),
            name,
            description,
            misc,
            signature_func,
            lower_funcs,
        };

        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::OpDefExists(op.name)),
            Entry::Vacant(ve) => Ok(ve.insert(Arc::new(op))),
        }
    }

    /// Create an OpDef with custom binary code to compute the signature
    pub fn add_op_custom_sig(
        &mut self,
        name: SmolStr,
        description: String,
        static_params: Vec<TypeParam>,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        signature_func: impl CustomSignatureFunc + 'static,
    ) -> Result<&OpDef, ExtensionBuildError> {
        self.add_op(
            name,
            description,
            misc,
            lower_funcs,
            SignatureFunc::CustomFunc {
                static_params,
                func: Box::new(signature_func),
            },
        )
    }

    /// Create an OpDef with custom binary code to compute the type scheme
    /// (which may be polymorphic); and no "misc" or "lowering functions" defined.
    pub fn add_op_custom_sig_simple(
        &mut self,
        name: SmolStr,
        description: String,
        static_params: Vec<TypeParam>,
        signature_func: impl CustomSignatureFunc + 'static,
    ) -> Result<&OpDef, ExtensionBuildError> {
        self.add_op_custom_sig(
            name,
            description,
            static_params,
            HashMap::default(),
            Vec::new(),
            signature_func,
        )
    }

    /// Create an OpDef with a signature (inputs+outputs) read from e.g.
    /// declarative YAML
    pub fn add_op_type_scheme(
        &mut self,
        name: SmolStr,
        description: String,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        type_scheme: PolyFuncType,
    ) -> Result<&OpDef, ExtensionBuildError> {
        self.add_op(
            name,
            description,
            misc,
            lower_funcs,
            SignatureFunc::TypeScheme(type_scheme),
        )
    }
}

#[cfg(test)]
mod test {
    use smol_str::SmolStr;

    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::USIZE_T;
    use crate::extension::PRELUDE;
    use crate::ops::custom::ExternalOp;
    use crate::ops::LeafOp;
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::Type;
    use crate::types::{type_param::TypeParam, FunctionType, PolyFuncType, TypeArg, TypeBound};
    use crate::{const_extension_ids, Extension};

    const_extension_ids! {
        const EXT_ID: ExtensionId = "MyExt";
    }

    #[test]
    fn op_def_with_type_scheme() -> Result<(), Box<dyn std::error::Error>> {
        let reg1 = [PRELUDE.to_owned(), EXTENSION.to_owned()].into();
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let mut e = Extension::new(EXT_ID);
        const TP: TypeParam = TypeParam::Type(TypeBound::Any);
        let list_of_var =
            Type::new_extension(list_def.instantiate(vec![TypeArg::new_var_use(0, TP)])?);
        const OP_NAME: SmolStr = SmolStr::new_inline("Reverse");
        let type_scheme = PolyFuncType::new_validated(
            vec![TP],
            FunctionType::new_linear(vec![list_of_var]),
            &reg1,
        )?;
        e.add_op_type_scheme(OP_NAME, "".into(), Default::default(), vec![], type_scheme)?;

        let list_usize =
            Type::new_extension(list_def.instantiate(vec![TypeArg::Type { ty: USIZE_T }])?);
        let mut dfg = DFGBuilder::new(FunctionType::new_linear(vec![list_usize]))?;
        let rev = dfg.add_dataflow_op(
            LeafOp::from(ExternalOp::Extension(
                e.instantiate_extension_op(&OP_NAME, vec![TypeArg::Type { ty: USIZE_T }], &reg1)
                    .unwrap(),
            )),
            dfg.input_wires(),
        )?;
        dfg.finish_hugr_with_outputs(
            rev.outputs(),
            &[PRELUDE.to_owned(), EXTENSION.to_owned(), e].into(),
        )?;

        Ok(())
    }
}
