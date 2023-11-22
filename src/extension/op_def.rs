use std::cmp::min;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use smol_str::SmolStr;

use super::{
    Extension, ExtensionBuildError, ExtensionId, ExtensionRegistry, ExtensionSet, SignatureError,
};

use crate::types::type_param::{check_type_args, TypeArg, TypeParam};
use crate::types::{FunctionType, PolyFuncType};
use crate::Hugr;

/// Trait necessary for binary computations of OpDef signature
pub trait CustomSignatureFunc: Send + Sync {
    /// Compute signature of node given
    /// values for the type parameters,
    /// the operation definition and the extension registry.
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        def: &'o OpDef,
        extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError>;
    /// The declared type parameters which require values in order for signature to
    /// be computed.
    fn static_params(&self) -> &[TypeParam];
}

/// Compute signature of `OpDef` given type arguments.
pub trait SignatureFromArgs: Send + Sync {
    /// Compute signature of node given
    /// values for the type parameters.
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
    ) -> Result<PolyFuncType, SignatureError>;
    /// The declared type parameters which require values in order for signature to
    /// be computed.
    fn static_params(&self) -> &[TypeParam];
}

impl<T: SignatureFromArgs> CustomSignatureFunc for T {
    #[inline]
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        _def: &'o OpDef,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<PolyFuncType, SignatureError> {
        SignatureFromArgs::compute_signature(self, arg_values)
    }

    #[inline]
    fn static_params(&self) -> &[TypeParam] {
        SignatureFromArgs::static_params(self)
    }
}

/// Trait for validating type arguments to a PolyFuncType beyond conformation to
/// declared type parameter.
pub trait ValidateTypeArgs: Send + Sync {
    /// Validate the type arguments of node given
    /// values for the type parameters,
    /// the operation definition and the extension registry.
    fn validate<'o, 'a: 'o>(
        &self,
        arg_values: &[TypeArg],
        def: &'o OpDef,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError>;
}

/// Trait for validating type arguments to a PolyFuncType beyond conformation to
/// declared type parameter, given just the arguments.
pub trait ValidateJustArgs: Send + Sync {
    /// Validate the type arguments of node given
    /// values for the type parameters.
    fn validate<'o, 'a: 'o>(&self, arg_values: &[TypeArg]) -> Result<(), SignatureError>;
}

impl<T: ValidateJustArgs> ValidateTypeArgs for T {
    #[inline]
    fn validate<'o, 'a: 'o>(
        &self,
        arg_values: &[TypeArg],
        _def: &'o OpDef,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        ValidateJustArgs::validate(self, arg_values)
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

/// Encode a signature as `PolyFuncType` but optionally allow validating type
/// arguments via a custom binary.
#[derive(serde::Deserialize, serde::Serialize)]
pub struct CustomValidator {
    #[serde(flatten)]
    poly_func: PolyFuncType,
    #[serde(skip)]
    validate: Box<dyn ValidateTypeArgs>,
}

impl CustomValidator {
    /// Encode a signature using a `PolyFuncType`
    pub fn from_polyfunc(poly_func: impl Into<PolyFuncType>) -> Self {
        Self {
            poly_func: poly_func.into(),
            validate: Default::default(),
        }
    }

    /// Encode a signature using a `PolyFuncType`, with a custom function for
    /// validating type arguments before returning the signature.
    pub fn new_with_validator(
        poly_func: impl Into<PolyFuncType>,
        validate: impl ValidateTypeArgs + 'static,
    ) -> Self {
        Self {
            poly_func: poly_func.into(),
            validate: Box::new(validate),
        }
    }
}

/// The two ways in which an OpDef may compute the Signature of each operation node.
#[derive(serde::Deserialize, serde::Serialize)]
pub enum SignatureFunc {
    // Note: except for serialization, we could have type schemes just implement the same
    // CustomSignatureFunc trait too, and replace this enum with Box<dyn CustomSignatureFunc>.
    // However instead we treat all CustomFunc's as non-serializable.
    #[serde(rename = "signature")]
    TypeScheme(CustomValidator),
    #[serde(skip)]
    CustomFunc(Box<dyn CustomSignatureFunc>),
}
struct NoValidate;
impl ValidateTypeArgs for NoValidate {
    fn validate<'o, 'a: 'o>(
        &self,
        _arg_values: &[TypeArg],
        _def: &'o OpDef,
        _extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        Ok(())
    }
}

impl Default for Box<dyn ValidateTypeArgs> {
    fn default() -> Self {
        Box::new(NoValidate)
    }
}

impl<T: CustomSignatureFunc + 'static> From<T> for SignatureFunc {
    fn from(v: T) -> Self {
        Self::CustomFunc(Box::new(v))
    }
}

impl From<PolyFuncType> for SignatureFunc {
    fn from(v: PolyFuncType) -> Self {
        Self::TypeScheme(CustomValidator::from_polyfunc(v))
    }
}

impl From<FunctionType> for SignatureFunc {
    fn from(v: FunctionType) -> Self {
        Self::TypeScheme(CustomValidator::from_polyfunc(v))
    }
}

impl From<CustomValidator> for SignatureFunc {
    fn from(v: CustomValidator) -> Self {
        Self::TypeScheme(v)
    }
}

impl SignatureFunc {
    fn compute_signature<'o, 'a: 'o>(
        &self,
        arg_values: &'a [TypeArg],
        def: &'o OpDef,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(PolyFuncType, &'a [TypeArg]), SignatureError> {
        Ok(match self {
            SignatureFunc::TypeScheme(custom) => {
                custom
                    .validate
                    .validate(arg_values, def, extension_registry)?;
                (custom.poly_func.clone(), arg_values)
            }
            SignatureFunc::CustomFunc(func) => {
                let static_params = self.static_params();
                let (static_args, other_args) =
                    arg_values.split_at(min(static_params.len(), arg_values.len()));

                check_type_args(static_args, static_params)?;
                let pf = func.compute_signature(static_args, def, extension_registry)?;
                (pf, other_args)
            }
        })
    }

    fn static_params(&self) -> &[TypeParam] {
        match self {
            SignatureFunc::TypeScheme(ts) => ts.poly_func.params(),
            SignatureFunc::CustomFunc(func) => func.static_params(),
        }
    }
}

impl Debug for SignatureFunc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeScheme(ts) => ts.poly_func.fmt(f),
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

impl OpDef {
    /// Check provided type arguments are valid against [ExtensionRegistry],
    /// against parameters, and that no type variables are used as static arguments
    /// (to [compute_signature][CustomSignatureFunc::compute_signature])
    pub fn validate_args(
        &self,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        let temp: PolyFuncType; // to keep alive
        let (pf, args) = match &self.signature_func {
            SignatureFunc::TypeScheme(ts) => (&ts.poly_func, args),
            SignatureFunc::CustomFunc(custom) => {
                let (static_args, other_args) =
                    args.split_at(min(custom.static_params().len(), args.len()));
                static_args
                    .iter()
                    .try_for_each(|ta| ta.validate(exts, &[]))?;
                check_type_args(static_args, custom.static_params())?;
                temp = custom.compute_signature(static_args, self, exts)?;
                (&temp, other_args)
            }
        };
        args.iter()
            .try_for_each(|ta| ta.validate(exts, var_decls))?;
        check_type_args(args, pf.params())?;
        Ok(())
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

        let (pf, args) = self.signature_func.compute_signature(args, self, exts)?;

        let res = pf.instantiate(args, exts)?;
        // TODO bring this assert back once resource inference is done?
        // https://github.com/CQCL-DEV/hugr/issues/425
        // assert!(res.contains(self.extension()));
        Ok(res)
    }

    pub(crate) fn should_serialize_signature(&self) -> bool {
        match self.signature_func {
            SignatureFunc::TypeScheme { .. } => false,
            SignatureFunc::CustomFunc { .. } => true,
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
        self.signature_func.static_params()
    }

    pub(super) fn validate(&self, exts: &ExtensionRegistry) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate declared TypeParams
        // for both type scheme and custom binary
        if let SignatureFunc::TypeScheme(ts) = &self.signature_func {
            ts.poly_func.validate(exts, &[])?;
        }
        Ok(())
    }
}

impl Extension {
    /// Add an operation definition to the extension.
    pub fn add_op(
        &mut self,
        name: SmolStr,
        description: String,
        misc: HashMap<String, serde_yaml::Value>,
        lower_funcs: Vec<LowerFunc>,
        signature_func: impl Into<SignatureFunc>,
    ) -> Result<&OpDef, ExtensionBuildError> {
        let op = OpDef {
            extension: self.name.clone(),
            name,
            description,
            misc,
            signature_func: signature_func.into(),
            lower_funcs,
        };

        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::OpDefExists(op.name)),
            Entry::Vacant(ve) => Ok(ve.insert(Arc::new(op))),
        }
    }

    /// Create an OpDef with `PolyFuncType`, `impl CustomSignatureFunc` or `CustomValidator`
    /// ; and no "misc" or "lowering functions" defined.
    pub fn add_op_simple(
        &mut self,
        name: SmolStr,
        description: String,
        signature_func: impl Into<SignatureFunc>,
    ) -> Result<&OpDef, ExtensionBuildError> {
        self.add_op(
            name,
            description,
            HashMap::default(),
            Vec::new(),
            signature_func,
        )
    }
}

#[cfg(test)]
mod test {
    use smol_str::SmolStr;

    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::USIZE_T;
    use crate::extension::{ExtensionRegistry, PRELUDE};
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
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let mut e = Extension::new(EXT_ID);
        const TP: TypeParam = TypeParam::Type(TypeBound::Any);
        let list_of_var =
            Type::new_extension(list_def.instantiate(vec![TypeArg::new_var_use(0, TP)])?);
        const OP_NAME: SmolStr = SmolStr::new_inline("Reverse");
        let type_scheme = PolyFuncType::new(vec![TP], FunctionType::new_endo(vec![list_of_var]));
        e.add_op(OP_NAME, "".into(), Default::default(), vec![], type_scheme)?;
        let reg =
            ExtensionRegistry::try_new([PRELUDE.to_owned(), EXTENSION.to_owned(), e]).unwrap();
        let e = reg.get(&EXT_ID).unwrap();

        let list_usize =
            Type::new_extension(list_def.instantiate(vec![TypeArg::Type { ty: USIZE_T }])?);
        let mut dfg = DFGBuilder::new(FunctionType::new_endo(vec![list_usize]))?;
        let rev = dfg.add_dataflow_op(
            LeafOp::from(ExternalOp::Extension(
                e.instantiate_extension_op(&OP_NAME, vec![TypeArg::Type { ty: USIZE_T }], &reg)
                    .unwrap(),
            )),
            dfg.input_wires(),
        )?;
        dfg.finish_hugr_with_outputs(rev.outputs(), &reg)?;

        Ok(())
    }
}
