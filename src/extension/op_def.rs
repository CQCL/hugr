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
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncType, SignatureError>;
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
/// declared type parameter (which should have been checked beforehand).
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
/// declared type parameter (which should have been checked beforehand), given just the arguments.
pub trait ValidateJustArgs: Send + Sync {
    /// Validate the type arguments of node given
    /// values for the type parameters.
    fn validate(&self, arg_values: &[TypeArg]) -> Result<(), SignatureError>;
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
/// arguments via a custom binary. The binary cannot be serialized so will be
/// lost over a serialization round-trip.
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
    /// A TypeScheme (polymorphic function type), with optional custom
    /// validation for provided type arguments,
    #[serde(rename = "signature")]
    TypeScheme(CustomValidator),
    #[serde(skip)]
    /// A custom binary which computes a polymorphic function type given values
    /// for its static type parameters.
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
    fn static_params(&self) -> &[TypeParam] {
        match self {
            SignatureFunc::TypeScheme(ts) => ts.poly_func.params(),
            SignatureFunc::CustomFunc(func) => func.static_params(),
        }
    }
    pub fn compute_signature(
        &self,
        def: &OpDef,
        args: &[TypeArg],
        exts: &ExtensionRegistry,
    ) -> Result<FunctionType, SignatureError> {
        let temp: PolyFuncType;
        let (pf, args) = match &self {
            SignatureFunc::TypeScheme(custom) => {
                custom.validate.validate(args, def, exts)?;
                (&custom.poly_func, args)
            }
            SignatureFunc::CustomFunc(func) => {
                let static_params = func.static_params();
                let (static_args, other_args) = args.split_at(min(static_params.len(), args.len()));

                check_type_args(static_args, static_params)?;
                temp = func.compute_signature(static_args, def, exts)?;
                (&temp, other_args)
            }
        };

        let res = pf.instantiate(args, exts)?;
        // TODO bring this assert back once resource inference is done?
        // https://github.com/CQCL-DEV/hugr/issues/425
        // assert!(res.contains(self.extension()));
        Ok(res)
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
        self.signature_func.compute_signature(self, args, exts)
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
        // TODO test this
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

    /// Add a lowering function to the [OpDef]
    pub fn add_lower_func(&mut self, lower: LowerFunc) {
        self.lower_funcs.push(lower);
    }

    /// Insert miscellaneous data `v` to the [OpDef], keyed by `k`.
    pub fn add_misc(
        &mut self,
        k: impl ToString,
        v: serde_yaml::Value,
    ) -> Option<serde_yaml::Value> {
        self.misc.insert(k.to_string(), v)
    }
}

impl Extension {
    /// Add an operation definition to the extension. Must be a type scheme
    /// (defined by a [`PolyFuncType`]), a type scheme along with binary
    /// validation for type arguments ([`CustomValidator`]), or a custom binary
    /// function for computing the signature given type arguments (`impl [CustomSignatureFunc]`).
    pub fn add_op(
        &mut self,
        name: SmolStr,
        description: String,
        signature_func: impl Into<SignatureFunc>,
    ) -> Result<&mut OpDef, ExtensionBuildError> {
        let op = OpDef {
            extension: self.name.clone(),
            name,
            description,
            signature_func: signature_func.into(),
            misc: Default::default(),
            lower_funcs: Default::default(),
        };

        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::OpDefExists(op.name)),
            // Just made the arc so should only be one reference to it, can get_mut,
            Entry::Vacant(ve) => Ok(Arc::get_mut(ve.insert(Arc::new(op))).unwrap()),
        }
    }

    pub fn add_op_enum(
        &mut self,
        op: &impl super::simple_op::OpEnum,
    ) -> Result<&mut OpDef, ExtensionBuildError> {
        let def = self.add_op(
            op.name().into(),
            op.description().to_string(),
            op.def_signature(),
        )?;

        op.post_opdef(def);

        Ok(def)
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZeroU64;

    use smol_str::SmolStr;

    use super::SignatureFromArgs;
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::op_def::LowerFunc;
    use crate::extension::prelude::USIZE_T;
    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE};
    use crate::extension::{SignatureError, EMPTY_REG, PRELUDE_REGISTRY};
    use crate::ops::custom::ExternalOp;
    use crate::ops::LeafOp;
    use crate::std_extensions::collections::{EXTENSION, LIST_TYPENAME};
    use crate::types::Type;
    use crate::types::{type_param::TypeParam, FunctionType, PolyFuncType, TypeArg, TypeBound};
    use crate::Hugr;
    use crate::{const_extension_ids, Extension};

    const_extension_ids! {
        const EXT_ID: ExtensionId = "MyExt";
    }

    #[test]
    fn op_def_with_type_scheme() -> Result<(), Box<dyn std::error::Error>> {
        let list_def = EXTENSION.get_type(&LIST_TYPENAME).unwrap();
        let mut e = Extension::new(EXT_ID);
        const TP: TypeParam = TypeParam::Type { b: TypeBound::Any };
        let list_of_var =
            Type::new_extension(list_def.instantiate(vec![TypeArg::new_var_use(0, TP)])?);
        const OP_NAME: SmolStr = SmolStr::new_inline("Reverse");
        let type_scheme = PolyFuncType::new(vec![TP], FunctionType::new_endo(vec![list_of_var]));

        let def = e.add_op(OP_NAME, "desc".into(), type_scheme)?;
        def.add_lower_func(LowerFunc::FixedHugr(ExtensionSet::new(), Hugr::default()));
        def.add_misc("key", Default::default());
        assert_eq!(def.description(), "desc");
        assert_eq!(def.lower_funcs.len(), 1);
        assert_eq!(def.misc.len(), 1);

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

    #[test]
    fn binary_polyfunc() -> Result<(), Box<dyn std::error::Error>> {
        // Test a custom binary `compute_signature` that returns a PolyFuncType
        // where the latter declares more type params itself. In particular,
        // we should be able to substitute (external) type variables into the latter,
        // but not pass them into the former (custom binary function).
        struct SigFun();
        impl SignatureFromArgs for SigFun {
            fn compute_signature(
                &self,
                arg_values: &[TypeArg],
            ) -> Result<PolyFuncType, SignatureError> {
                const TP: TypeParam = TypeParam::Type { b: TypeBound::Any };
                let [TypeArg::BoundedNat { n }] = arg_values else {
                    return Err(SignatureError::InvalidTypeArgs);
                };
                let n = *n as usize;
                let tvs: Vec<Type> = (0..n)
                    .map(|_| Type::new_var_use(0, TypeBound::Any))
                    .collect();
                Ok(PolyFuncType::new(
                    vec![TP.to_owned()],
                    FunctionType::new(tvs.clone(), vec![Type::new_tuple(tvs)]),
                ))
            }

            fn static_params(&self) -> &[TypeParam] {
                const MAX_NAT: &[TypeParam] = &[TypeParam::max_nat()];
                MAX_NAT
            }
        }
        let mut e = Extension::new(EXT_ID);
        let def: &mut crate::extension::OpDef =
            e.add_op("MyOp".into(), "".to_string(), SigFun())?;

        // Base case, no type variables:
        let args = [TypeArg::BoundedNat { n: 3 }, USIZE_T.into()];
        assert_eq!(
            def.compute_signature(&args, &PRELUDE_REGISTRY),
            Ok(FunctionType::new(
                vec![USIZE_T; 3],
                vec![Type::new_tuple(vec![USIZE_T; 3])]
            ))
        );
        assert_eq!(def.validate_args(&args, &PRELUDE_REGISTRY, &[]), Ok(()));

        // Second arg may be a variable (substitutable)
        let tyvar = Type::new_var_use(0, TypeBound::Eq);
        let tyvars: Vec<Type> = vec![tyvar.clone(); 3];
        let args = [TypeArg::BoundedNat { n: 3 }, tyvar.clone().into()];
        assert_eq!(
            def.compute_signature(&args, &PRELUDE_REGISTRY),
            Ok(FunctionType::new(
                tyvars.clone(),
                vec![Type::new_tuple(tyvars)]
            ))
        );
        def.validate_args(&args, &PRELUDE_REGISTRY, &[TypeBound::Eq.into()])
            .unwrap();

        // quick sanity check that we are validating the args - note changed bound:
        assert_eq!(
            def.validate_args(&args, &PRELUDE_REGISTRY, &[TypeBound::Any.into()]),
            Err(SignatureError::TypeVarDoesNotMatchDeclaration {
                actual: TypeBound::Any.into(),
                cached: TypeBound::Eq.into()
            })
        );

        // First arg must be concrete, not a variable
        let kind = TypeParam::bounded_nat(NonZeroU64::new(5).unwrap());
        let args = [TypeArg::new_var_use(0, kind.clone()), USIZE_T.into()];
        // We can't prevent this from getting into our compute_signature implementation:
        assert_eq!(
            def.compute_signature(&args, &PRELUDE_REGISTRY),
            Err(SignatureError::InvalidTypeArgs)
        );
        // But validation rules it out, even when the variable is declared:
        assert_eq!(
            def.validate_args(&args, &PRELUDE_REGISTRY, &[kind]),
            Err(SignatureError::FreeTypeVar {
                idx: 0,
                num_decls: 0
            })
        );

        Ok(())
    }

    #[test]
    fn type_scheme_instantiate_var() -> Result<(), Box<dyn std::error::Error>> {
        // Check that we can instantiate a PolyFuncType-scheme with an (external)
        // type variable
        let mut e = Extension::new(EXT_ID);
        let def = e.add_op(
            "SimpleOp".into(),
            "".into(),
            PolyFuncType::new(
                vec![TypeBound::Any.into()],
                FunctionType::new_endo(vec![Type::new_var_use(0, TypeBound::Any)]),
            ),
        )?;
        let tv = Type::new_var_use(1, TypeBound::Eq);
        let args = [TypeArg::Type { ty: tv.clone() }];
        let decls = [TypeParam::Extensions, TypeBound::Eq.into()];
        def.validate_args(&args, &EMPTY_REG, &decls).unwrap();
        assert_eq!(
            def.compute_signature(&args, &EMPTY_REG),
            Ok(FunctionType::new_endo(vec![tv]))
        );
        Ok(())
    }
}
