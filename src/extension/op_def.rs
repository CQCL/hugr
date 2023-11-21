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
            SignatureFunc::TypeScheme(ts) => (ts, args),
            SignatureFunc::CustomFunc {
                static_params,
                func,
            } => {
                let (static_args, other_args) = args.split_at(min(static_params.len(), args.len()));
                static_args
                    .iter()
                    .try_for_each(|ta| ta.validate(exts, &[]))?;
                check_type_args(static_args, static_params)?;
                temp = func.compute_signature(&self.name, static_args, &self.misc, exts)?;
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
        match &self.signature_func {
            SignatureFunc::TypeScheme(ts) => ts.params(),
            SignatureFunc::CustomFunc { static_params, .. } => static_params,
        }
    }

    pub(super) fn validate(&self, exts: &ExtensionRegistry) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate declared TypeParams
        // for both type scheme and custom binary
        if let SignatureFunc::TypeScheme(ts) = &self.signature_func {
            ts.validate(exts, &[])?;
        }
        Ok(())
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

    /// Create an OpDef with a signature (inputs+outputs) read from e.g.
    /// declarative YAML; and no "misc" or "lowering functions" defined.
    pub fn add_op_type_scheme_simple(
        &mut self,
        name: SmolStr,
        description: String,
        type_scheme: PolyFuncType,
    ) -> Result<&OpDef, ExtensionBuildError> {
        self.add_op(
            name,
            description,
            Default::default(),
            vec![],
            SignatureFunc::TypeScheme(type_scheme),
        )
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZeroU64;

    use smol_str::SmolStr;

    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::USIZE_T;
    use crate::extension::{
        CustomSignatureFunc, ExtensionRegistry, SignatureError, PRELUDE, PRELUDE_REGISTRY,
    };
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
        e.add_op_type_scheme(OP_NAME, "".into(), Default::default(), vec![], type_scheme)?;
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
        struct SigFun();
        impl CustomSignatureFunc for SigFun {
            fn compute_signature(
                &self,
                _name: &SmolStr,
                arg_values: &[TypeArg],
                _misc: &std::collections::HashMap<String, serde_yaml::Value>,
                _exts: &ExtensionRegistry,
            ) -> Result<PolyFuncType, SignatureError> {
                const TP: TypeParam = TypeParam::Type(TypeBound::Any);
                let [TypeArg::BoundedNat {n}] = arg_values else { return Err(SignatureError::InvalidTypeArgs) };
                let n = *n as usize;
                let tvs: Vec<Type> = (0..n)
                    .map(|_| Type::new_var_use(0, TypeBound::Any))
                    .collect();
                Ok(PolyFuncType::new(
                    vec![TP],
                    FunctionType::new(tvs.clone(), vec![Type::new_tuple(tvs)]),
                ))
            }
        }
        let mut e = Extension::new(EXT_ID);
        let def = e.add_op_custom_sig_simple(
            "MyOp".into(),
            "".to_string(),
            vec![TypeParam::max_nat()],
            SigFun(),
        )?;

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
        let tyvars: Vec<Type> = (0..3).map(|_| tyvar.clone()).collect();
        let args = [TypeArg::BoundedNat { n: 3 }, tyvar.clone().into()];
        assert_eq!(
            def.compute_signature(&args, &PRELUDE_REGISTRY),
            Ok(FunctionType::new(
                tyvars.clone(),
                vec![Type::new_tuple(tyvars)]
            ))
        );
        def.validate_args(&args, &PRELUDE_REGISTRY, &[TypeParam::Type(TypeBound::Eq)])
            .unwrap();

        // quick sanity check that we are validating the args - note changed bound:
        assert_eq!(
            def.validate_args(&args, &PRELUDE_REGISTRY, &[TypeParam::Type(TypeBound::Any)]),
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
}
