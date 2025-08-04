use std::cmp::min;
use std::collections::HashMap;
use std::collections::btree_map::Entry;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Weak};

use serde_with::serde_as;

use super::{
    ConstFold, ConstFoldResult, Extension, ExtensionBuildError, ExtensionId, ExtensionSet,
    SignatureError,
};

use crate::Hugr;
use crate::envelope::serde_with::AsBinaryEnvelope;
use crate::ops::{OpName, OpNameRef};
use crate::types::type_param::{TypeArg, TypeParam, check_term_types};
use crate::types::{FuncValueType, PolyFuncType, PolyFuncTypeRV, Signature};
mod serialize_signature_func;

/// Trait necessary for binary computations of `OpDef` signature
pub trait CustomSignatureFunc: Send + Sync {
    /// Compute signature of node given
    /// values for the type parameters,
    /// the operation definition and the extension registry.
    fn compute_signature<'o, 'a: 'o>(
        &'a self,
        arg_values: &[TypeArg],
        def: &'o OpDef,
    ) -> Result<PolyFuncTypeRV, SignatureError>;
    /// The declared type parameters which require values in order for signature to
    /// be computed.
    fn static_params(&self) -> &[TypeParam];
}

/// Compute signature of `OpDef` given type arguments.
pub trait SignatureFromArgs: Send + Sync {
    /// Compute signature of node given
    /// values for the type parameters.
    fn compute_signature(&self, arg_values: &[TypeArg]) -> Result<PolyFuncTypeRV, SignatureError>;
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
    ) -> Result<PolyFuncTypeRV, SignatureError> {
        SignatureFromArgs::compute_signature(self, arg_values)
    }

    #[inline]
    fn static_params(&self) -> &[TypeParam] {
        SignatureFromArgs::static_params(self)
    }
}

/// Trait for validating type arguments to a `PolyFuncTypeRV` beyond conformation to
/// declared type parameter (which should have been checked beforehand).
pub trait ValidateTypeArgs: Send + Sync {
    /// Validate the type arguments of node given
    /// values for the type parameters,
    /// the operation definition and the extension registry.
    fn validate<'o, 'a: 'o>(
        &self,
        arg_values: &[TypeArg],
        def: &'o OpDef,
    ) -> Result<(), SignatureError>;
}

/// Trait for validating type arguments to a `PolyFuncTypeRV` beyond conformation to
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
/// This trait allows the Hugr to be varied according to the operation's [`TypeArg`]s;
/// if this is not necessary then a single Hugr can be provided instead via
/// [`LowerFunc::FixedHugr`].
pub trait CustomLowerFunc: Send + Sync {
    /// Return a Hugr that implements the node using only the specified available extensions;
    /// may fail.
    /// TODO: some error type to indicate Extensions required?
    fn try_lower(
        &self,
        name: &OpNameRef,
        arg_values: &[TypeArg],
        misc: &HashMap<String, serde_json::Value>,
        available_extensions: &ExtensionSet,
    ) -> Option<Hugr>;
}

/// Encode a signature as [`PolyFuncTypeRV`] but with additional validation of type
/// arguments via a custom binary. The binary cannot be serialized so will be
/// lost over a serialization round-trip.
pub struct CustomValidator {
    poly_func: PolyFuncTypeRV,
    /// Custom function for validating type arguments before returning the signature.
    pub(crate) validate: Box<dyn ValidateTypeArgs>,
}

impl CustomValidator {
    /// Encode a signature using a `PolyFuncTypeRV`, with a custom function for
    /// validating type arguments before returning the signature.
    pub fn new(
        poly_func: impl Into<PolyFuncTypeRV>,
        validate: impl ValidateTypeArgs + 'static,
    ) -> Self {
        Self {
            poly_func: poly_func.into(),
            validate: Box::new(validate),
        }
    }

    /// Return a mutable reference to the `PolyFuncType`.
    pub(super) fn poly_func_mut(&mut self) -> &mut PolyFuncTypeRV {
        &mut self.poly_func
    }
}

/// The ways in which an `OpDef` may compute the Signature of each operation node.
pub enum SignatureFunc {
    /// An explicit polymorphic function type.
    PolyFuncType(PolyFuncTypeRV),
    /// A polymorphic function type (like [`Self::PolyFuncType`] but also with a custom binary for validating type arguments.
    CustomValidator(CustomValidator),
    /// Serialized declaration specified a custom validate binary but it was not provided.
    MissingValidateFunc(PolyFuncTypeRV),
    /// A custom binary which computes a polymorphic function type given values
    /// for its static type parameters.
    CustomFunc(Box<dyn CustomSignatureFunc>),
    /// Serialized declaration specified a custom compute binary but it was not provided.
    MissingComputeFunc,
}

impl<T: CustomSignatureFunc + 'static> From<T> for SignatureFunc {
    fn from(v: T) -> Self {
        Self::CustomFunc(Box::new(v))
    }
}

impl From<PolyFuncType> for SignatureFunc {
    fn from(value: PolyFuncType) -> Self {
        Self::PolyFuncType(value.into())
    }
}

impl From<PolyFuncTypeRV> for SignatureFunc {
    fn from(v: PolyFuncTypeRV) -> Self {
        Self::PolyFuncType(v)
    }
}

impl From<FuncValueType> for SignatureFunc {
    fn from(v: FuncValueType) -> Self {
        Self::PolyFuncType(v.into())
    }
}

impl From<Signature> for SignatureFunc {
    fn from(v: Signature) -> Self {
        Self::PolyFuncType(FuncValueType::from(v).into())
    }
}

impl From<CustomValidator> for SignatureFunc {
    fn from(v: CustomValidator) -> Self {
        Self::CustomValidator(v)
    }
}

impl SignatureFunc {
    fn static_params(&self) -> Result<&[TypeParam], SignatureError> {
        Ok(match self {
            SignatureFunc::PolyFuncType(ts)
            | SignatureFunc::CustomValidator(CustomValidator { poly_func: ts, .. })
            | SignatureFunc::MissingValidateFunc(ts) => ts.params(),
            SignatureFunc::CustomFunc(func) => func.static_params(),
            SignatureFunc::MissingComputeFunc => return Err(SignatureError::MissingComputeFunc),
        })
    }

    /// If the signature is missing a custom validation function, ignore and treat as
    /// self-contained type scheme (with no custom validation).
    pub fn ignore_missing_validation(&mut self) {
        if let SignatureFunc::MissingValidateFunc(ts) = self {
            *self = SignatureFunc::PolyFuncType(ts.clone());
        }
    }

    /// Compute the concrete signature ([`FuncValueType`]).
    ///
    /// # Panics
    ///
    /// Panics if `self` is a [`SignatureFunc::CustomFunc`] and there are not enough type
    /// arguments provided to match the number of static parameters.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type arguments are invalid or
    /// there is some error in type computation.
    pub fn compute_signature(
        &self,
        def: &OpDef,
        args: &[TypeArg],
    ) -> Result<Signature, SignatureError> {
        let temp: PolyFuncTypeRV; // to keep alive
        let (pf, args) = match &self {
            SignatureFunc::CustomValidator(custom) => {
                custom.validate.validate(args, def)?;
                (&custom.poly_func, args)
            }
            SignatureFunc::PolyFuncType(ts) => (ts, args),
            SignatureFunc::CustomFunc(func) => {
                let static_params = func.static_params();
                let (static_args, other_args) = args.split_at(min(static_params.len(), args.len()));

                check_term_types(static_args, static_params)?;
                temp = func.compute_signature(static_args, def)?;
                (&temp, other_args)
            }
            SignatureFunc::MissingComputeFunc => return Err(SignatureError::MissingComputeFunc),
            // TODO raise warning: https://github.com/CQCL/hugr/issues/1432
            SignatureFunc::MissingValidateFunc(ts) => (ts, args),
        };
        let res = pf.instantiate(args)?;

        // If there are any row variables left, this will fail with an error:
        res.try_into()
    }
}

impl Debug for SignatureFunc {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CustomValidator(ts) => ts.poly_func.fmt(f),
            Self::PolyFuncType(ts) => ts.fmt(f),
            Self::CustomFunc { .. } => f.write_str("<custom sig>"),
            Self::MissingComputeFunc => f.write_str("<missing custom sig>"),
            Self::MissingValidateFunc(_) => f.write_str("<missing custom validation>"),
        }
    }
}

/// Different ways that an [OpDef] can lower operation nodes i.e. provide a Hugr
/// that implements the operation using a set of other extensions.
///
/// Does not implement [`serde::Deserialize`] directly since the serde error for
/// untagged enums is unhelpful. Use [`deserialize_lower_funcs`] with
/// [`serde(deserialize_with = "deserialize_lower_funcs")] instead.
#[serde_as]
#[derive(serde::Serialize)]
#[serde(untagged)]
pub enum LowerFunc {
    /// Lowering to a fixed Hugr. Since this cannot depend upon the [TypeArg]s,
    /// this will generally only be applicable if the [OpDef] has no [TypeParam]s.
    FixedHugr {
        /// The extensions required by the [`Hugr`]
        extensions: ExtensionSet,
        /// The [`Hugr`] to be used to replace [ExtensionOp]s matching the parent
        /// [OpDef]
        ///
        /// [ExtensionOp]: crate::ops::ExtensionOp
        #[serde_as(as = "Box<AsBinaryEnvelope>")]
        hugr: Box<Hugr>,
    },
    /// Custom binary function that can (fallibly) compute a Hugr
    /// for the particular instance and set of available extensions.
    #[serde(skip)]
    CustomFunc(Box<dyn CustomLowerFunc>),
}

/// A function for deserializing sequences of [`LowerFunc::FixedHugr`].
///
/// We could let serde deserialize [`LowerFunc`] as-is, but if the LowerFunc
/// deserialization fails it just returns an opaque "data did not match any
/// variant of untagged enum LowerFunc" error. This function will return the
/// internal errors instead.
pub fn deserialize_lower_funcs<'de, D>(deserializer: D) -> Result<Vec<LowerFunc>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[serde_as]
    #[derive(serde::Deserialize)]
    struct FixedHugrDeserializer {
        pub extensions: ExtensionSet,
        #[serde_as(as = "Box<AsBinaryEnvelope>")]
        pub hugr: Box<Hugr>,
    }

    let funcs: Vec<FixedHugrDeserializer> = serde::Deserialize::deserialize(deserializer)?;
    Ok(funcs
        .into_iter()
        .map(|f| LowerFunc::FixedHugr {
            extensions: f.extensions,
            hugr: f.hugr,
        })
        .collect())
}

impl Debug for LowerFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixedHugr { .. } => write!(f, "FixedHugr"),
            Self::CustomFunc(_) => write!(f, "<custom lower>"),
        }
    }
}

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new `OpDef`'s from a serialized definition.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// The unique Extension owning this `OpDef` (of which this `OpDef` is a member)
    extension: ExtensionId,
    /// A weak reference to the extension defining this operation.
    #[serde(skip)]
    extension_ref: Weak<Extension>,
    /// Unique identifier of the operation. Used to look up `OpDefs` in the registry
    /// when deserializing nodes (which store only the name).
    name: OpName,
    /// Human readable description of the operation.
    description: String,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    misc: HashMap<String, serde_json::Value>,

    #[serde(with = "serialize_signature_func", flatten)]
    signature_func: SignatureFunc,
    // Some operations cannot lower themselves and tools that do not understand them
    // can only treat them as opaque/black-box ops.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_lower_funcs"
    )]
    pub(crate) lower_funcs: Vec<LowerFunc>,

    /// Operations can optionally implement [`ConstFold`] to implement constant folding.
    #[serde(skip)]
    constant_folder: Option<Box<dyn ConstFold>>,
}

impl OpDef {
    /// Check provided type arguments are valid against their extensions,
    /// against parameters, and that no type variables are used as static arguments
    /// (to [`compute_signature`][CustomSignatureFunc::compute_signature])
    pub fn validate_args(
        &self,
        args: &[TypeArg],
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        let temp: PolyFuncTypeRV; // to keep alive
        let (pf, args) = match &self.signature_func {
            SignatureFunc::CustomValidator(ts) => (&ts.poly_func, args),
            SignatureFunc::PolyFuncType(ts) => (ts, args),
            SignatureFunc::CustomFunc(custom) => {
                let (static_args, other_args) =
                    args.split_at(min(custom.static_params().len(), args.len()));
                static_args.iter().try_for_each(|ta| ta.validate(&[]))?;
                check_term_types(static_args, custom.static_params())?;
                temp = custom.compute_signature(static_args, self)?;
                (&temp, other_args)
            }
            SignatureFunc::MissingComputeFunc => return Err(SignatureError::MissingComputeFunc),
            SignatureFunc::MissingValidateFunc(_) => {
                return Err(SignatureError::MissingValidateFunc);
            }
        };
        args.iter().try_for_each(|ta| ta.validate(var_decls))?;
        check_term_types(args, pf.params())?;
        Ok(())
    }

    /// Computes the signature of a node, i.e. an instantiation of this
    /// `OpDef` with statically-provided [`TypeArg`]s.
    pub fn compute_signature(&self, args: &[TypeArg]) -> Result<Signature, SignatureError> {
        self.signature_func.compute_signature(self, args)
    }

    /// Fallibly returns a Hugr that may replace an instance of this `OpDef`
    /// given a set of available extensions that may be used in the Hugr.
    #[must_use]
    pub fn try_lower(&self, args: &[TypeArg], available_extensions: &ExtensionSet) -> Option<Hugr> {
        // TODO test this
        self.lower_funcs
            .iter()
            .filter_map(|f| match f {
                LowerFunc::FixedHugr { extensions, hugr } => {
                    if available_extensions.is_superset(extensions) {
                        Some(hugr.as_ref().clone())
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
    #[must_use]
    pub fn name(&self) -> &OpName {
        &self.name
    }

    /// Returns a reference to the extension id of this [`OpDef`].
    #[must_use]
    pub fn extension_id(&self) -> &ExtensionId {
        &self.extension
    }

    /// Returns a weak reference to the extension defining this operation.
    #[must_use]
    pub fn extension(&self) -> Weak<Extension> {
        self.extension_ref.clone()
    }

    /// Returns a mutable reference to the weak extension pointer in the operation definition.
    pub(super) fn extension_mut(&mut self) -> &mut Weak<Extension> {
        &mut self.extension_ref
    }

    /// Returns a reference to the description of this [`OpDef`].
    #[must_use]
    pub fn description(&self) -> &str {
        self.description.as_ref()
    }

    /// Returns a reference to the params of this [`OpDef`].
    pub fn params(&self) -> Result<&[TypeParam], SignatureError> {
        self.signature_func.static_params()
    }

    pub(super) fn validate(&self) -> Result<(), SignatureError> {
        // TODO https://github.com/CQCL/hugr/issues/624 validate declared TypeParams
        // for both type scheme and custom binary
        if let SignatureFunc::CustomValidator(ts) = &self.signature_func {
            // The type scheme may contain row variables so be of variable length;
            // these will have to be substituted to fixed-length concrete types when
            // the OpDef is instantiated into an actual OpType.
            ts.poly_func.validate()?;
        }
        Ok(())
    }

    /// Add a lowering function to the [`OpDef`]
    pub fn add_lower_func(&mut self, lower: LowerFunc) {
        self.lower_funcs.push(lower);
    }

    /// Insert miscellaneous data `v` to the [`OpDef`], keyed by `k`.
    pub fn add_misc(
        &mut self,
        k: impl ToString,
        v: serde_json::Value,
    ) -> Option<serde_json::Value> {
        self.misc.insert(k.to_string(), v)
    }

    /// Iterate over all miscellaneous data in the [`OpDef`].
    #[allow(unused)] // Unused when no features are enabled
    pub(crate) fn iter_misc(&self) -> impl ExactSizeIterator<Item = (&str, &serde_json::Value)> {
        self.misc.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Set the constant folding function for this Op, which can evaluate it
    /// given constant inputs.
    pub fn set_constant_folder(&mut self, fold: impl ConstFold + 'static) {
        self.constant_folder = Some(Box::new(fold));
    }

    /// Evaluate an instance of this [`OpDef`] defined by the `type_args`, given
    /// [`crate::ops::Const`] values for inputs at [`crate::IncomingPort`]s.
    #[must_use]
    pub fn constant_fold(
        &self,
        type_args: &[TypeArg],
        consts: &[(crate::IncomingPort, crate::ops::Value)],
    ) -> ConstFoldResult {
        (self.constant_folder.as_ref())?.fold(type_args, consts)
    }

    /// Returns a reference to the signature function of this [`OpDef`].
    #[must_use]
    pub fn signature_func(&self) -> &SignatureFunc {
        &self.signature_func
    }

    /// Returns a mutable reference to the signature function of this [`OpDef`].
    pub(super) fn signature_func_mut(&mut self) -> &mut SignatureFunc {
        &mut self.signature_func
    }
}

impl Extension {
    /// Add an operation definition to the extension. Must be a type scheme
    /// (defined by a [`PolyFuncTypeRV`]), a type scheme along with binary
    /// validation for type arguments ([`CustomValidator`]), or a custom binary
    /// function for computing the signature given type arguments (implementing
    /// `[CustomSignatureFunc]`).
    ///
    /// This method requires a [`Weak`] reference to the [`Arc`] containing the
    /// extension being defined. The intended way to call this method is inside
    /// the closure passed to [`Extension::new_arc`] when defining the extension.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr_core::types::Signature;
    /// # use hugr_core::extension::{Extension, ExtensionId, Version};
    /// Extension::new_arc(
    ///     ExtensionId::new_unchecked("my.extension"),
    ///     Version::new(0, 1, 0),
    ///     |ext, extension_ref| {
    ///         ext.add_op(
    ///             "MyOp".into(),
    ///             "Some operation".into(),
    ///             Signature::new_endo(vec![]),
    ///             extension_ref,
    ///         );
    ///     },
    /// );
    /// ```
    pub fn add_op(
        &mut self,
        name: OpName,
        description: String,
        signature_func: impl Into<SignatureFunc>,
        extension_ref: &Weak<Extension>,
    ) -> Result<&mut OpDef, ExtensionBuildError> {
        let op = OpDef {
            extension: self.name.clone(),
            extension_ref: extension_ref.clone(),
            name,
            description,
            signature_func: signature_func.into(),
            misc: Default::default(),
            lower_funcs: Default::default(),
            constant_folder: Default::default(),
        };

        match self.operations.entry(op.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::OpDefExists(op.name)),
            // Just made the arc so should only be one reference to it, can get_mut,
            Entry::Vacant(ve) => Ok(Arc::get_mut(ve.insert(Arc::new(op))).unwrap()),
        }
    }
}

#[cfg(test)]
pub(super) mod test {
    use std::num::NonZeroU64;

    use itertools::Itertools;

    use super::SignatureFromArgs;
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig};
    use crate::extension::SignatureError;
    use crate::extension::op_def::{CustomValidator, LowerFunc, OpDef, SignatureFunc};
    use crate::extension::prelude::usize_t;
    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE};
    use crate::ops::OpName;
    use crate::std_extensions::collections::list;
    use crate::types::type_param::{TermTypeError, TypeParam};
    use crate::types::{PolyFuncTypeRV, Signature, Type, TypeArg, TypeBound, TypeRV};
    use crate::{Extension, const_extension_ids};

    const_extension_ids! {
        const EXT_ID: ExtensionId = "MyExt";
    }

    /// A dummy wrapper over an operation definition.
    #[derive(serde::Serialize, serde::Deserialize, Debug)]
    pub struct SimpleOpDef(OpDef);

    impl SimpleOpDef {
        /// Create a new dummy opdef.
        #[must_use]
        pub fn new(op_def: OpDef) -> Self {
            assert!(op_def.constant_folder.is_none());
            assert!(matches!(
                op_def.signature_func,
                SignatureFunc::PolyFuncType(_)
            ));
            assert!(
                op_def
                    .lower_funcs
                    .iter()
                    .all(|lf| matches!(lf, LowerFunc::FixedHugr { .. }))
            );
            Self(op_def)
        }
    }

    impl From<SimpleOpDef> for OpDef {
        fn from(value: SimpleOpDef) -> Self {
            value.0
        }
    }

    impl PartialEq for SimpleOpDef {
        fn eq(&self, other: &Self) -> bool {
            let OpDef {
                extension,
                extension_ref: _,
                name,
                description,
                misc,
                signature_func,
                lower_funcs,
                constant_folder: _,
            } = &self.0;
            let OpDef {
                extension: other_extension,
                extension_ref: _,
                name: other_name,
                description: other_description,
                misc: other_misc,
                signature_func: other_signature_func,
                lower_funcs: other_lower_funcs,
                constant_folder: _,
            } = &other.0;

            let get_sig = |sf: &_| match sf {
                // if SignatureFunc or CustomValidator are changed we should get
                // a compile error here. To fix: modify the fields matched on here,
                // maintaining the lack of `..` and, for each part that is
                // serializable, ensure we are checking it for equality below.
                SignatureFunc::CustomValidator(CustomValidator {
                    poly_func,
                    validate: _,
                })
                | SignatureFunc::PolyFuncType(poly_func)
                | SignatureFunc::MissingValidateFunc(poly_func) => Some(poly_func.clone()),
                SignatureFunc::CustomFunc(_) | SignatureFunc::MissingComputeFunc => None,
            };

            let get_lower_funcs = |lfs: &Vec<LowerFunc>| {
                lfs.iter()
                    .map(|lf| match lf {
                        // as with get_sig above, this should break if the hierarchy
                        // is changed, update similarly.
                        LowerFunc::FixedHugr { extensions, hugr } => {
                            Some((extensions.clone(), hugr.clone()))
                        }
                        // This is ruled out by `new()` but leave it here for later.
                        LowerFunc::CustomFunc(_) => None,
                    })
                    .collect_vec()
            };

            extension == other_extension
                && name == other_name
                && description == other_description
                && misc == other_misc
                && get_sig(signature_func) == get_sig(other_signature_func)
                && get_lower_funcs(lower_funcs) == get_lower_funcs(other_lower_funcs)
        }
    }

    #[test]
    fn op_def_with_type_scheme() -> Result<(), Box<dyn std::error::Error>> {
        let list_def = list::EXTENSION.get_type(&list::LIST_TYPENAME).unwrap();
        const OP_NAME: OpName = OpName::new_inline("Reverse");

        let ext = Extension::try_new_test_arc(EXT_ID, |ext, extension_ref| {
            const TP: TypeParam = TypeParam::RuntimeType(TypeBound::Linear);
            let list_of_var =
                Type::new_extension(list_def.instantiate(vec![TypeArg::new_var_use(0, TP)])?);
            let type_scheme = PolyFuncTypeRV::new(vec![TP], Signature::new_endo(vec![list_of_var]));

            let def = ext.add_op(OP_NAME, "desc".into(), type_scheme, extension_ref)?;
            def.add_lower_func(LowerFunc::FixedHugr {
                extensions: ExtensionSet::new(),
                hugr: Box::new(crate::builder::test::simple_dfg_hugr()), // this is nonsense, but we are not testing the actual lowering here
            });
            def.add_misc("key", Default::default());
            assert_eq!(def.description(), "desc");
            assert_eq!(def.lower_funcs.len(), 1);
            assert_eq!(def.misc.len(), 1);

            Ok(())
        })?;

        let reg = ExtensionRegistry::new([PRELUDE.clone(), list::EXTENSION.clone(), ext]);
        reg.validate()?;
        let e = reg.get(&EXT_ID).unwrap();

        let list_usize = Type::new_extension(list_def.instantiate(vec![usize_t().into()])?);
        let mut dfg = DFGBuilder::new(endo_sig(vec![list_usize]))?;
        let rev = dfg.add_dataflow_op(
            e.instantiate_extension_op(&OP_NAME, vec![usize_t().into()])
                .unwrap(),
            dfg.input_wires(),
        )?;
        dfg.finish_hugr_with_outputs(rev.outputs())?;

        Ok(())
    }

    #[test]
    fn binary_polyfunc() -> Result<(), Box<dyn std::error::Error>> {
        // Test a custom binary `compute_signature` that returns a PolyFuncTypeRV
        // where the latter declares more type params itself. In particular,
        // we should be able to substitute (external) type variables into the latter,
        // but not pass them into the former (custom binary function).
        struct SigFun();
        impl SignatureFromArgs for SigFun {
            fn compute_signature(
                &self,
                arg_values: &[TypeArg],
            ) -> Result<PolyFuncTypeRV, SignatureError> {
                const TP: TypeParam = TypeParam::RuntimeType(TypeBound::Linear);
                let [TypeArg::BoundedNat(n)] = arg_values else {
                    return Err(SignatureError::InvalidTypeArgs);
                };
                let n = *n as usize;
                let tvs: Vec<Type> = (0..n)
                    .map(|_| Type::new_var_use(0, TypeBound::Linear))
                    .collect();
                Ok(PolyFuncTypeRV::new(
                    vec![TP.clone()],
                    Signature::new(tvs.clone(), vec![Type::new_tuple(tvs)]),
                ))
            }

            fn static_params(&self) -> &[TypeParam] {
                const MAX_NAT: &[TypeParam] = &[TypeParam::max_nat_type()];
                MAX_NAT
            }
        }
        let _ext = Extension::try_new_test_arc(EXT_ID, |ext, extension_ref| {
            let def: &mut crate::extension::OpDef =
                ext.add_op("MyOp".into(), String::new(), SigFun(), extension_ref)?;

            // Base case, no type variables:
            let args = [TypeArg::BoundedNat(3), usize_t().into()];
            assert_eq!(
                def.compute_signature(&args),
                Ok(Signature::new(
                    vec![usize_t(); 3],
                    vec![Type::new_tuple(vec![usize_t(); 3])]
                ))
            );
            assert_eq!(def.validate_args(&args, &[]), Ok(()));

            // Second arg may be a variable (substitutable)
            let tyvar = Type::new_var_use(0, TypeBound::Copyable);
            let tyvars: Vec<Type> = vec![tyvar.clone(); 3];
            let args = [TypeArg::BoundedNat(3), tyvar.clone().into()];
            assert_eq!(
                def.compute_signature(&args),
                Ok(Signature::new(
                    tyvars.clone(),
                    vec![Type::new_tuple(tyvars)]
                ))
            );
            def.validate_args(&args, &[TypeBound::Copyable.into()])
                .unwrap();

            // quick sanity check that we are validating the args - note changed bound:
            assert_eq!(
                def.validate_args(&args, &[TypeBound::Linear.into()]),
                Err(SignatureError::TypeVarDoesNotMatchDeclaration {
                    actual: Box::new(TypeBound::Linear.into()),
                    cached: Box::new(TypeBound::Copyable.into())
                })
            );

            // First arg must be concrete, not a variable
            let kind = TypeParam::bounded_nat_type(NonZeroU64::new(5).unwrap());
            let args = [TypeArg::new_var_use(0, kind.clone()), usize_t().into()];
            // We can't prevent this from getting into our compute_signature implementation:
            assert_eq!(
                def.compute_signature(&args),
                Err(SignatureError::InvalidTypeArgs)
            );
            // But validation rules it out, even when the variable is declared:
            assert_eq!(
                def.validate_args(&args, &[kind]),
                Err(SignatureError::FreeTypeVar {
                    idx: 0,
                    num_decls: 0
                })
            );

            Ok(())
        })?;

        Ok(())
    }

    #[test]
    fn type_scheme_instantiate_var() -> Result<(), Box<dyn std::error::Error>> {
        // Check that we can instantiate a PolyFuncTypeRV-scheme with an (external)
        // type variable
        let _ext = Extension::try_new_test_arc(EXT_ID, |ext, extension_ref| {
            let def = ext.add_op(
                "SimpleOp".into(),
                String::new(),
                PolyFuncTypeRV::new(
                    vec![TypeBound::Linear.into()],
                    Signature::new_endo(vec![Type::new_var_use(0, TypeBound::Linear)]),
                ),
                extension_ref,
            )?;
            let tv = Type::new_var_use(0, TypeBound::Copyable);
            let args = [tv.clone().into()];
            let decls = [TypeBound::Copyable.into()];
            def.validate_args(&args, &decls).unwrap();
            assert_eq!(def.compute_signature(&args), Ok(Signature::new_endo(tv)));
            // But not with an external row variable
            let arg: TypeArg = TypeRV::new_row_var_use(0, TypeBound::Copyable).into();
            assert_eq!(
                def.compute_signature(std::slice::from_ref(&arg)),
                Err(SignatureError::TypeArgMismatch(
                    TermTypeError::TypeMismatch {
                        type_: Box::new(TypeBound::Linear.into()),
                        term: Box::new(arg),
                    }
                ))
            );
            Ok(())
        })?;
        Ok(())
    }

    mod proptest {
        use std::sync::Weak;

        use super::SimpleOpDef;
        use ::proptest::prelude::*;

        use crate::{
            builder::test::simple_dfg_hugr,
            extension::{ExtensionId, ExtensionSet, OpDef, SignatureFunc, op_def::LowerFunc},
            types::PolyFuncTypeRV,
        };

        impl Arbitrary for SignatureFunc {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                // TODO there is also  SignatureFunc::CustomFunc, but for now
                // this is not serialized. When it is, we should generate
                // examples here .
                any::<PolyFuncTypeRV>()
                    .prop_map(SignatureFunc::PolyFuncType)
                    .boxed()
            }
        }

        impl Arbitrary for LowerFunc {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                // TODO There is also LowerFunc::CustomFunc, but for now this is
                // not serialized. When it is, we should generate examples here.
                any::<ExtensionSet>()
                    .prop_map(|extensions| LowerFunc::FixedHugr {
                        extensions,
                        hugr: Box::new(simple_dfg_hugr()),
                    })
                    .boxed()
            }
        }

        impl Arbitrary for SimpleOpDef {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                use crate::proptest::{any_serde_json_value, any_smolstr, any_string};
                use proptest::collection::{hash_map, vec};
                let misc = hash_map(any_string(), any_serde_json_value(), 0..3);
                (
                    any::<ExtensionId>(),
                    any_smolstr(),
                    any_string(),
                    misc,
                    any::<SignatureFunc>(),
                    vec(any::<LowerFunc>(), 0..2),
                )
                    .prop_map(
                        |(extension, name, description, misc, signature_func, lower_funcs)| {
                            Self::new(OpDef {
                                extension,
                                // Use a dead weak reference. Trying to access the extension will always return None.
                                extension_ref: Weak::default(),
                                name,
                                description,
                                misc,
                                signature_func,
                                lower_funcs,
                                // TODO ``constant_folder` is not serialized, we should
                                // generate examples once it is.
                                constant_folder: None,
                            })
                        },
                    )
                    .boxed()
            }
        }
    }
}
