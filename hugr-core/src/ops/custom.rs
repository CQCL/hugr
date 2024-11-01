//! Extensible operations.

use std::sync::Arc;
use thiserror::Error;
#[cfg(test)]
use {
    crate::extension::test::SimpleOpDef,
    crate::proptest::{any_nonempty_smolstr, any_nonempty_string},
    ::proptest::prelude::*,
    ::proptest_derive::Arbitrary,
};

use crate::extension::{ConstFoldResult, ExtensionId, ExtensionRegistry, OpDef, SignatureError};
use crate::hugr::HugrView;
use crate::types::{type_param::TypeArg, Signature};
use crate::{extension::ExtOpSignature, hugr::internal::HugrMutInternals, types::EdgeKind};
use crate::{ops, Hugr, IncomingPort, Node};

use super::dataflow::DataflowOpTrait;
use super::tag::OpTag;
use super::{NamedOp, OpName, OpNameRef, OpTrait, OpType};

/// An operation defined by an [OpDef] from a loaded [Extension].
///
/// Extension ops are not serializable. They must be downgraded into an [OpaqueOp] instead.
/// See [ExtensionOp::make_opaque].
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
    args: Vec<TypeArg>,
    signature: ExtOpSignature, // Cache
}

impl ExtensionOp {
    /// Create a new ExtensionOp given the type arguments and specified input extensions
    pub fn new(
        def: Arc<OpDef>,
        args: impl Into<Vec<TypeArg>>,
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let args = args.into();
        let signature = def.compute_signature(&args, exts)?;
        Ok(Self {
            def,
            args,
            signature,
        })
    }

    /// If OpDef is missing binary computation, trust the cached signature.
    fn new_with_cached(
        def: Arc<OpDef>,
        args: impl Into<Vec<TypeArg>>,
        opaque: &OpaqueOp,
        exts: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let args = args.into();
        // TODO skip computation depending on config
        // see https://github.com/CQCL/hugr/issues/1363
        let signature = match def.compute_signature(&args, exts) {
            Ok(sig) => sig,
            Err(SignatureError::MissingComputeFunc) => {
                // TODO raise warning: https://github.com/CQCL/hugr/issues/1432
                opaque.ext_op_signature()
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
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns a reference to the [`OpDef`] of this [`ExtensionOp`].
    pub fn def(&self) -> &OpDef {
        self.def.as_ref()
    }

    /// Attempt to evaluate this operation. See [`OpDef::constant_fold`].
    pub fn constant_fold(&self, consts: &[(IncomingPort, ops::Value)]) -> ConstFoldResult {
        self.def().constant_fold(self.args(), consts)
    }

    /// Creates a new [`OpaqueOp`] as a downgraded version of this
    /// [`ExtensionOp`].
    ///
    /// Regenerating the [`ExtensionOp`] back from the [`OpaqueOp`] requires a
    /// registry with the appropriate extension. See [`resolve_opaque_op`].
    ///
    /// For a non-cloning version of this operation, use [`OpaqueOp::from`].
    pub fn make_opaque(&self) -> OpaqueOp {
        OpaqueOp {
            extension: self.def.extension().clone(),
            name: self.def.name().clone(),
            description: self.def.description().into(),
            args: self.args.clone(),
            signature: self.signature.clone(),
        }
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
            extension: def.extension().clone(),
            name: def.name().clone(),
            description: def.description().into(),
            args,
            signature,
        }
    }
}

impl PartialEq for ExtensionOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::<OpDef>::ptr_eq(&self.def, &other.def) && self.args == other.args
    }
}

impl Eq for ExtensionOp {}

impl NamedOp for ExtensionOp {
    /// The name of the operation.
    fn name(&self) -> OpName {
        qualify_name(self.def.extension(), self.def.name())
    }
}

impl DataflowOpTrait for ExtensionOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        self.def().description()
    }

    fn signature(&self) -> Signature {
        self.signature.func_type().clone()
    }

    fn static_inputs(&self) -> Vec<EdgeKind> {
        self.signature
            .static_inputs()
            .iter()
            .cloned()
            .map(EdgeKind::Const)
            .collect()
    }
}

/// An opaquely-serialized op that refers to an as-yet-unresolved [`OpDef`].
///
/// [ExtensionOp]s are serialised as `OpaqueOp`s.
///
/// The signature of a [ExtensionOp] always includes that op's extension. We do not
/// require that the `signature` field of [OpaqueOp] contains `extension`,
/// instead we are careful to add it whenever we look at the `signature` of an
/// `OpaqueOp`. This is a small efficiency in serialisation and allows us to
/// be more liberal in deserialisation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct OpaqueOp {
    extension: ExtensionId,
    #[cfg_attr(test, proptest(strategy = "any_nonempty_smolstr()"))]
    name: OpName,
    #[cfg_attr(test, proptest(strategy = "any_nonempty_string()"))]
    description: String, // cache in advance so description() can return &str
    args: Vec<TypeArg>,
    // note that the `signature` field might not include `extension`. Thus this must
    // remain private, and should be accessed through
    // `DataflowOpTrait::signature`.
    signature: ExtOpSignature,
}

fn qualify_name(res_id: &ExtensionId, name: &OpNameRef) -> OpName {
    format!("{}.{}", res_id, name).into()
}

impl OpaqueOp {
    /// Creates a new OpaqueOp from all the fields we'd expect to serialize.
    pub fn new(
        extension: ExtensionId,
        name: impl Into<OpName>,
        description: String,
        args: impl Into<Vec<TypeArg>>,
        signature: impl Into<ExtOpSignature>,
    ) -> Self {
        Self {
            extension,
            name: name.into(),
            description,
            args: args.into(),
            signature: signature.into(),
        }
    }
}

impl NamedOp for OpaqueOp {
    /// The name of the operation.
    fn name(&self) -> OpName {
        qualify_name(&self.extension, &self.name)
    }
}
impl OpaqueOp {
    /// Unique name of the operation.
    pub fn op_name(&self) -> &OpName {
        &self.name
    }

    /// Type arguments.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Parent extension.
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }

    /// Instantiated signature of the operation.
    pub fn ext_op_signature(&self) -> ExtOpSignature {
        let mut sig = self.signature.clone();
        sig.func_type = sig.func_type.with_extension_delta(self.extension.clone());
        sig
    }
}

impl DataflowOpTrait for OpaqueOp {
    const TAG: OpTag = OpTag::Leaf;

    fn description(&self) -> &str {
        &self.description
    }

    fn signature(&self) -> Signature {
        self.ext_op_signature().func_type
    }
}

/// Resolve serialized names of operations into concrete implementation (OpDefs) where possible
pub fn resolve_extension_ops(
    h: &mut Hugr,
    extension_registry: &ExtensionRegistry,
) -> Result<(), OpaqueOpError> {
    let mut replacements = Vec::new();
    for n in h.nodes() {
        if let OpType::OpaqueOp(opaque) = h.get_optype(n) {
            let resolved = resolve_opaque_op(n, opaque, extension_registry)?;
            replacements.push((n, resolved));
        }
    }
    // Only now can we perform the replacements as the 'for' loop was borrowing 'h' preventing use from using it mutably
    for (n, op) in replacements {
        debug_assert_eq!(h.get_optype(n).tag(), OpTag::Leaf);
        debug_assert_eq!(op.tag(), OpTag::Leaf);
        h.replace_op(n, op).unwrap();
    }
    Ok(())
}

/// Try to resolve a [`OpaqueOp`] to a [`ExtensionOp`] by looking the op up in
/// the registry.
///
/// # Return
/// Some if the serialized opaque resolves to an extension-defined op and all is
/// ok; None if the serialized opaque doesn't identify an extension
///
/// # Errors
/// If the serialized opaque resolves to a definition that conflicts with what
/// was serialized
pub fn resolve_opaque_op(
    node: Node,
    opaque: &OpaqueOp,
    extension_registry: &ExtensionRegistry,
) -> Result<ExtensionOp, OpaqueOpError> {
    if let Some(r) = extension_registry.get(&opaque.extension) {
        // Fail if the Extension was found but did not have the expected operation
        let Some(def) = r.get_op(&opaque.name) else {
            return Err(OpaqueOpError::OpNotFoundInExtension(
                node,
                opaque.name.clone(),
                r.name().clone(),
            ));
        };
        dbg!(opaque.signature().extension_reqs);
        let ext_op = ExtensionOp::new_with_cached(
            def.clone(),
            opaque.args.clone(),
            opaque,
            extension_registry,
        )
        .map_err(|e| OpaqueOpError::SignatureError {
            node,
            name: opaque.name.clone(),
            cause: e,
        })?;
        if opaque.signature() != ext_op.signature() {
            dbg!(opaque.signature().extension_reqs);
            dbg!(ext_op.signature().extension_reqs);
            return Err(OpaqueOpError::SignatureMismatch {
                node,
                extension: opaque.extension.clone(),
                op: def.name().clone(),
                computed: ext_op.signature(),
                stored: opaque.signature(),
            });
        };
        Ok(ext_op)
    } else {
        Err(OpaqueOpError::UnresolvedOp(
            node,
            opaque.name.clone(),
            opaque.extension.clone(),
        ))
    }
}

/// Errors that arise after loading a Hugr containing opaque ops (serialized just as their names)
/// when trying to resolve the serialized names against a registry of known Extensions.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum OpaqueOpError {
    /// The Extension was found but did not contain the expected OpDef
    #[error("Operation '{1}' in {0} not found in Extension {2}")]
    OpNotFoundInExtension(Node, OpName, ExtensionId),
    /// Extension and OpDef found, but computed signature did not match stored
    #[error("Conflicting signature: resolved {op} in extension {extension} to a concrete implementation which computed {computed} but stored signature was {stored}")]
    #[allow(missing_docs)]
    SignatureMismatch {
        node: Node,
        extension: ExtensionId,
        op: OpName,
        stored: Signature,
        computed: Signature,
    },
    /// An error in computing the signature of the ExtensionOp
    #[error("Error in signature of operation '{name}' in {node}: {cause}")]
    #[allow(missing_docs)]
    SignatureError {
        node: Node,
        name: OpName,
        #[source]
        cause: SignatureError,
    },
    /// Unresolved operation encountered during validation.
    #[error("Unexpected unresolved opaque operation '{1}' in {0}, from Extension {2}.")]
    UnresolvedOp(Node, OpName, ExtensionId),
}

#[cfg(test)]
mod test {

    use crate::std_extensions::arithmetic::conversions::{self, CONVERT_OPS_REGISTRY};
    use crate::{
        extension::{
            prelude::{BOOL_T, QB_T, USIZE_T},
            SignatureFunc,
        },
        std_extensions::arithmetic::int_types::INT_TYPES,
        types::FuncValueType,
        Extension,
    };

    use super::*;

    #[test]
    fn new_opaque_op() {
        let sig = Signature::new_endo(vec![QB_T]);
        let op = OpaqueOp::new(
            "res".try_into().unwrap(),
            "op",
            "desc".into(),
            vec![TypeArg::Type { ty: USIZE_T }],
            sig.clone(),
        );
        assert_eq!(op.name(), "res.op");
        assert_eq!(DataflowOpTrait::description(&op), "desc");
        assert_eq!(op.args(), &[TypeArg::Type { ty: USIZE_T }]);
        assert_eq!(
            op.signature(),
            sig.with_extension_delta(op.extension().clone())
        );
    }

    #[test]
    fn resolve_opaque_op() {
        let registry = &CONVERT_OPS_REGISTRY;
        let i0 = &INT_TYPES[0];
        let opaque = OpaqueOp::new(
            conversions::EXTENSION_ID,
            "itobool",
            "description".into(),
            vec![],
            Signature::new(i0.clone(), BOOL_T),
        );
        let resolved =
            super::resolve_opaque_op(Node::from(portgraph::NodeIndex::new(1)), &opaque, registry)
                .unwrap();
        assert_eq!(resolved.def().name(), "itobool");
    }

    #[test]
    fn resolve_missing() {
        let mut ext = Extension::new_test("ext".try_into().unwrap());
        let ext_id = ext.name().clone();
        let val_name = "missing_val";
        let comp_name = "missing_comp";

        let endo_sig = Signature::new_endo(BOOL_T);
        ext.add_op(
            val_name.into(),
            "".to_string(),
            SignatureFunc::MissingValidateFunc(FuncValueType::from(endo_sig.clone()).into()),
        )
        .unwrap();

        ext.add_op(
            comp_name.into(),
            "".to_string(),
            SignatureFunc::MissingComputeFunc,
        )
        .unwrap();
        let registry = ExtensionRegistry::try_new([ext]).unwrap();
        let opaque_val = OpaqueOp::new(
            ext_id.clone(),
            val_name,
            "".into(),
            vec![],
            endo_sig.clone(),
        );
        let opaque_comp = OpaqueOp::new(ext_id.clone(), comp_name, "".into(), vec![], endo_sig);
        let resolved_val = super::resolve_opaque_op(
            Node::from(portgraph::NodeIndex::new(1)),
            &opaque_val,
            &registry,
        )
        .unwrap();
        assert_eq!(resolved_val.def().name(), val_name);

        let resolved_comp = super::resolve_opaque_op(
            Node::from(portgraph::NodeIndex::new(2)),
            &opaque_comp,
            &registry,
        )
        .unwrap_or_else(|e| panic!("{}", e));
        assert_eq!(resolved_comp.def().name(), comp_name);
    }
}
