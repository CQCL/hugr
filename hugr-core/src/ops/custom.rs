//! Extensible operations.

use itertools::Itertools;
use std::borrow::Cow;
use std::sync::Arc;
use thiserror::Error;
#[cfg(test)]
use {
    crate::extension::test::SimpleOpDef, crate::proptest::any_nonempty_smolstr,
    crate::types::proptest_utils::any_serde_type_arg_vec, ::proptest::prelude::*,
    ::proptest_derive::Arbitrary,
};

use crate::core::HugrNode;
use crate::extension::simple_op::MakeExtensionOp;
use crate::extension::{ConstFoldResult, ExtensionId, OpDef, SignatureError};
use crate::types::{Signature, type_param::TypeArg};
use crate::{IncomingPort, ops};

use super::dataflow::DataflowOpTrait;
use super::tag::OpTag;
use super::{NamedOp, OpName, OpNameRef};

/// An operation defined by an [`OpDef`] from a loaded [Extension].
///
/// Extension ops are not serializable. They must be downgraded into an [`OpaqueOp`] instead.
/// See [`ExtensionOp::make_opaque`].
///
/// [Extension]: crate::Extension
#[derive(Clone, Debug, serde::Serialize)]
#[serde(into = "OpaqueOp")]
#[cfg_attr(test, derive(Arbitrary))]
pub struct ExtensionOp {
    #[cfg_attr(
        test,
        proptest(strategy = "any::<SimpleOpDef>().prop_map(|x| Arc::new(x.into()))")
    )]
    def: Arc<OpDef>,
    #[cfg_attr(test, proptest(strategy = "any_serde_type_arg_vec()"))]
    args: Vec<TypeArg>,
    signature: Signature, // Cache
}

impl ExtensionOp {
    /// Create a new `ExtensionOp` given the type arguments and specified input extensions
    pub fn new(def: Arc<OpDef>, args: impl Into<Vec<TypeArg>>) -> Result<Self, SignatureError> {
        let args: Vec<TypeArg> = args.into();
        let signature = def.compute_signature(&args)?;
        Ok(Self {
            def,
            args,
            signature,
        })
    }

    /// If `OpDef` is missing binary computation, trust the cached signature.
    pub(crate) fn new_with_cached(
        def: Arc<OpDef>,
        args: impl IntoIterator<Item = TypeArg>,
        opaque: &OpaqueOp,
    ) -> Result<Self, SignatureError> {
        let args: Vec<TypeArg> = args.into_iter().collect();
        // TODO skip computation depending on config
        // see https://github.com/CQCL/hugr/issues/1363
        let signature = match def.compute_signature(&args) {
            Ok(sig) => sig,
            Err(SignatureError::MissingComputeFunc) => {
                // TODO raise warning: https://github.com/CQCL/hugr/issues/1432
                opaque.signature().into_owned()
            }
            Err(e) => return Err(e),
        };
        Ok(Self {
            def,
            args,
            signature,
        })
    }

    /// Return the argument values for this operation.
    #[must_use]
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns a reference to the [`OpDef`] of this [`ExtensionOp`].
    #[must_use]
    pub fn def(&self) -> &OpDef {
        self.def.as_ref()
    }

    /// Gets an Arc to the [`OpDef`] of this instance, i.e. usable to create
    /// new instances.
    #[must_use]
    pub fn def_arc(&self) -> &Arc<OpDef> {
        &self.def
    }

    /// Attempt to evaluate this operation. See [`OpDef::constant_fold`].
    #[must_use]
    pub fn constant_fold(&self, consts: &[(IncomingPort, ops::Value)]) -> ConstFoldResult {
        self.def().constant_fold(self.args(), consts)
    }

    /// Creates a new [`OpaqueOp`] as a downgraded version of this
    /// [`ExtensionOp`].
    ///
    /// Regenerating the [`ExtensionOp`] back from the [`OpaqueOp`] requires a
    /// registry with the appropriate extension.
    ///
    /// For a non-cloning version of this operation, use [`OpaqueOp::from`].
    #[must_use]
    pub fn make_opaque(&self) -> OpaqueOp {
        OpaqueOp {
            extension: self.def.extension_id().clone(),
            name: self.def.name().clone(),
            args: self.args.clone(),
            signature: self.signature.clone(),
        }
    }

    /// Returns a mutable reference to the cached signature of the operation.
    pub fn signature_mut(&mut self) -> &mut Signature {
        &mut self.signature
    }

    /// Returns a mutable reference to the type arguments of the operation.
    pub(crate) fn args_mut(&mut self) -> &mut [TypeArg] {
        self.args.as_mut_slice()
    }

    /// Cast the operation to an specific extension op.
    ///
    /// Returns `None` if the operation is not of the requested type.
    #[must_use]
    pub fn cast<T: MakeExtensionOp>(&self) -> Option<T> {
        T::from_extension_op(self).ok()
    }

    /// Returns the extension id of the operation.
    #[must_use]
    pub fn extension_id(&self) -> &ExtensionId {
        self.def.extension_id()
    }

    /// Returns the unqualified id of the operation. e.g. 'iadd'
    ///
    #[must_use]
    pub fn unqualified_id(&self) -> &OpNameRef {
        self.def.name()
    }

    /// Returns the qualified id of the operation. e.g. 'arithmetic.iadd'
    #[must_use]
    pub fn qualified_id(&self) -> OpName {
        qualify_name(self.extension_id(), self.unqualified_id())
    }
}

impl From<ExtensionOp> for OpaqueOp {
    fn from(op: ExtensionOp) -> Self {
        let ExtensionOp {
            def,
            args,
            signature,
        } = op;
        OpaqueOp {
            extension: def.extension_id().clone(),
            name: def.name().clone(),
            args,
            signature,
        }
    }
}

impl PartialEq for ExtensionOp {
    fn eq(&self, other: &Self) -> bool {
        if Arc::<OpDef>::ptr_eq(&self.def, &other.def) {
            // If the OpDef is exactly the same, we can skip some checks.
            self.args() == other.args()
        } else {
            self.args() == other.args()
                && self.signature() == other.signature()
                && self.def.name() == other.def.name()
                && self.def.extension_id() == other.def.extension_id()
        }
    }
}

impl Eq for ExtensionOp {}

impl NamedOp for ExtensionOp {
    /// The name of the operation.
    fn name(&self) -> OpName {
        self.qualified_id()
    }
}

impl DataflowOpTrait for ExtensionOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        self.def().description()
    }

    fn signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.signature)
    }

    fn substitute(&self, subst: &crate::types::Substitution) -> Self {
        let args = self
            .args
            .iter()
            .map(|ta| ta.substitute(subst))
            .collect::<Vec<_>>();
        let signature = self.signature.substitute(subst);
        Self {
            def: self.def.clone(),
            args,
            signature,
        }
    }
}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`].
///
/// [`ExtensionOp`]s are serialised as `OpaqueOp`s.
///
/// The signature of a [`ExtensionOp`] always includes that op's extension. We do not
/// require that the `signature` field of [`OpaqueOp`] contains `extension`,
/// instead we are careful to add it whenever we look at the `signature` of an
/// `OpaqueOp`. This is a small efficiency in serialisation and allows us to
/// be more liberal in deserialisation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct OpaqueOp {
    extension: ExtensionId,
    #[cfg_attr(test, proptest(strategy = "any_nonempty_smolstr()"))]
    name: OpName,
    #[cfg_attr(test, proptest(strategy = "any_serde_type_arg_vec()"))]
    args: Vec<TypeArg>,
    // note that the `signature` field might not include `extension`. Thus this must
    // remain private, and should be accessed through
    // `DataflowOpTrait::signature`.
    signature: Signature,
}

fn qualify_name(res_id: &ExtensionId, name: &OpNameRef) -> OpName {
    format!("{res_id}.{name}").into()
}

impl OpaqueOp {
    /// Creates a new `OpaqueOp` from all the fields we'd expect to serialize.
    pub fn new(
        extension: ExtensionId,
        name: impl Into<OpName>,
        args: impl Into<Vec<TypeArg>>,
        signature: Signature,
    ) -> Self {
        Self {
            extension,
            name: name.into(),
            args: args.into(),
            signature,
        }
    }

    /// Returns a mutable reference to the signature of the operation.
    pub fn signature_mut(&mut self) -> &mut Signature {
        &mut self.signature
    }
}

impl NamedOp for OpaqueOp {
    fn name(&self) -> OpName {
        format!("OpaqueOp:{}", self.qualified_id()).into()
    }
}

impl OpaqueOp {
    /// Unique name of the operation.
    #[must_use]
    pub fn unqualified_id(&self) -> &OpName {
        &self.name
    }

    /// Unique name of the operation.
    #[must_use]
    pub fn qualified_id(&self) -> OpName {
        qualify_name(self.extension(), self.unqualified_id())
    }

    /// Type arguments.
    #[must_use]
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Parent extension.
    #[must_use]
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }

    /// Returns a mutable reference to the type arguments of the operation.
    pub(crate) fn args_mut(&mut self) -> &mut [TypeArg] {
        self.args.as_mut_slice()
    }
}

impl DataflowOpTrait for OpaqueOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        "Opaque operation"
    }

    fn signature(&self) -> Cow<'_, Signature> {
        Cow::Borrowed(&self.signature)
    }

    fn substitute(&self, subst: &crate::types::Substitution) -> Self {
        Self {
            args: self.args.iter().map(|ta| ta.substitute(subst)).collect(),
            signature: self.signature.substitute(subst),
            ..self.clone()
        }
    }
}

/// Errors that arise after loading a Hugr containing opaque ops (serialized just as their names)
/// when trying to resolve the serialized names against a registry of known Extensions.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum OpaqueOpError<N: HugrNode> {
    /// The Extension was found but did not contain the expected `OpDef`
    #[error("Operation '{op}' in {node} not found in Extension {extension}. Available operations: {}",
            available_ops.iter().join(", ")
    )]
    OpNotFoundInExtension {
        /// The node where the error occurred.
        node: N,
        /// The missing operation.
        op: OpName,
        /// The extension where the operation was expected.
        extension: ExtensionId,
        /// The available operations in the extension.
        available_ops: Vec<OpName>,
    },
    /// Extension and `OpDef` found, but computed signature did not match stored
    #[error(
        "Conflicting signature: resolved {op} in extension {extension} to a concrete implementation which computed {computed} but stored signature was {stored}"
    )]
    #[allow(missing_docs)]
    SignatureMismatch {
        node: N,
        extension: ExtensionId,
        op: OpName,
        stored: Box<Signature>,
        computed: Box<Signature>,
    },
    /// An error in computing the signature of the `ExtensionOp`
    #[error("Error in signature of operation '{name}' in {node}: {cause}")]
    #[allow(missing_docs)]
    SignatureError {
        node: N,
        name: OpName,
        #[source]
        cause: SignatureError,
    },
    /// Unresolved operation encountered during validation.
    #[error("Unexpected unresolved opaque operation '{1}' in {0}, from Extension {2}.")]
    UnresolvedOp(N, OpName, ExtensionId),
    /// Error updating the extension registry in the Hugr while resolving opaque ops.
    #[error("Error updating extension registry: {0}")]
    ExtensionRegistryError(#[from] crate::extension::ExtensionRegistryError),
}

#[cfg(test)]
mod test {

    use ops::OpType;

    use crate::Node;
    use crate::extension::ExtensionRegistry;
    use crate::extension::resolution::resolve_op_extensions;
    use crate::std_extensions::STD_REG;
    use crate::std_extensions::arithmetic::conversions::{self};
    use crate::{
        Extension,
        extension::{
            SignatureFunc,
            prelude::{bool_t, qb_t, usize_t},
        },
        std_extensions::arithmetic::int_types::INT_TYPES,
        types::FuncValueType,
    };

    use super::*;

    /// Unwrap the replacement type's `OpDef` from the return type of `resolve_op_definition`.
    fn resolve_res_definition(res: &OpType) -> &OpDef {
        res.as_extension_op().unwrap().def()
    }

    #[test]
    fn new_opaque_op() {
        let sig = Signature::new_endo(vec![qb_t()]);
        let op = OpaqueOp::new(
            "res".try_into().unwrap(),
            "op",
            vec![usize_t().into()],
            sig.clone(),
        );
        assert_eq!(op.name(), "OpaqueOp:res.op");
        assert_eq!(op.args(), &[usize_t().into()]);
        assert_eq!(op.signature().as_ref(), &sig);
    }

    #[test]
    fn resolve_opaque_op() {
        let registry = &STD_REG;
        let i0 = &INT_TYPES[0];
        let opaque = OpaqueOp::new(
            conversions::EXTENSION_ID,
            "itobool",
            vec![],
            Signature::new(i0.clone(), bool_t()),
        );
        let mut resolved = opaque.into();
        resolve_op_extensions(
            Node::from(portgraph::NodeIndex::new(1)),
            &mut resolved,
            registry,
        )
        .unwrap();
        assert_eq!(resolve_res_definition(&resolved).name(), "itobool");
    }

    #[test]
    fn resolve_missing() {
        let val_name = "missing_val";
        let comp_name = "missing_comp";
        let endo_sig = Signature::new_endo(bool_t());

        let ext = Extension::new_test_arc("ext".try_into().unwrap(), |ext, extension_ref| {
            ext.add_op(
                val_name.into(),
                String::new(),
                SignatureFunc::MissingValidateFunc(FuncValueType::from(endo_sig.clone()).into()),
                extension_ref,
            )
            .unwrap();

            ext.add_op(
                comp_name.into(),
                String::new(),
                SignatureFunc::MissingComputeFunc,
                extension_ref,
            )
            .unwrap();
        });
        let ext_id = ext.name().clone();

        let registry = ExtensionRegistry::new([ext]);
        registry.validate().unwrap();
        let opaque_val = OpaqueOp::new(ext_id.clone(), val_name, vec![], endo_sig.clone());
        let opaque_comp = OpaqueOp::new(ext_id.clone(), comp_name, vec![], endo_sig);
        let mut resolved_val = opaque_val.into();
        resolve_op_extensions(
            Node::from(portgraph::NodeIndex::new(1)),
            &mut resolved_val,
            &registry,
        )
        .unwrap();
        assert_eq!(resolve_res_definition(&resolved_val).name(), val_name);

        let mut resolved_comp = opaque_comp.into();
        resolve_op_extensions(
            Node::from(portgraph::NodeIndex::new(2)),
            &mut resolved_comp,
            &registry,
        )
        .unwrap();
        assert_eq!(resolve_res_definition(&resolved_comp).name(), comp_name);
    }
}
