//! Extensible operations.

use downcast_rs::{impl_downcast, Downcast};
use once_cell::sync::OnceCell;
use smol_str::SmolStr;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Deref;

use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::resource::ResourceSet;
use crate::types::SimpleType;
use crate::types::{Signature, SignatureDescription};

/// A wrapped [`CustomOp`] with fast equality checks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    /// Operation name, cached for fast equality checks.
    id: SmolStr,

    /// The custom operation.
    op: Box<dyn CustomOp>,
}

impl OpaqueOp {
    /// The name of the operation, cached for fast equality checks.
    pub fn name(&self) -> SmolStr {
        self.id.clone()
    }
}

impl PartialEq for OpaqueOp {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for OpaqueOp {}

impl Deref for OpaqueOp {
    type Target = dyn CustomOp;

    fn deref(&self) -> &Self::Target {
        self.op.as_ref()
    }
}

impl<T: CustomOp> From<T> for OpaqueOp {
    fn from(op: T) -> Self {
        Self {
            id: op.name(),
            op: Box::new(op),
        }
    }
}

/// Custom definition for an operation.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
#[typetag::serde]
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Any + Downcast {
    /// Try to convert the custom op to a graph definition.
    ///
    /// TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn try_into_hugr(&self, resources: &ResourceSet) -> Option<Hugr> {
        let _ = resources;
        None
    }

    /// List the resources required to execute this operation.
    fn resources(&self) -> &ResourceSet;

    /// The name of the operation.
    fn name(&self) -> SmolStr;

    /// Optional description of the operation.
    fn description(&self) -> &str {
        ""
    }

    /// The signature of the operation.
    fn signature(&self) -> Signature;

    /// Optional descriptions of the ports in the signature.
    fn signature_desc(&self) -> SignatureDescription {
        Default::default()
    }
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);

/// Serializable definition for dynamically loaded operations.
///
/// TODO: Define a way to construct new CustomOps from a serialized definition.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// Unique identifier of the operation.
    ///
    /// This is used to compare two custom ops for equality.
    pub name: SmolStr,
    /// Human readable description of the operation.
    pub description: String,
    inputs: Vec<(Option<SmolStr>, SimpleType)>,
    outputs: Vec<(Option<SmolStr>, SimpleType)>,
    /// Miscellaneous data associated with the operation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub misc: HashMap<String, serde_yaml::Value>,
    /// (YAML?)-encoded definition of the operation.
    ///
    /// TODO: Define the format of this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub def: Option<String>,
    /// Resources required to execute this operation.
    pub resource_reqs: ResourceSet,

    /// Signature of the operation.
    ///
    /// Computed from the serialized `inputs` and `outputs`.
    #[serde(skip)]
    signature: OnceCell<Signature>,
    /// Optional port descriptions.
    ///
    /// Computed from the serialized `inputs` and `outputs`.
    #[serde(skip)]
    port_names: OnceCell<SignatureDescription>,
}

impl OpDef {
    /// Initialize a new operation definition with a fixed signature.
    pub fn new(name: SmolStr, signature: Signature) -> Self {
        Self::new_with_description(
            name,
            String::new(),
            signature,
            SignatureDescription::default(),
        )
    }

    /// Initialize a new operation definition with a fixed signature.
    pub fn new_with_description(
        name: SmolStr,
        description: String,
        signature: Signature,
        port_names: SignatureDescription,
    ) -> Self {
        let inputs: Vec<_> = port_names
            .input_zip(&signature)
            .chain(port_names.const_input_zip(&signature))
            .map(|(n, t)| (Some(n.clone()), t.clone()))
            .collect();

        let outputs = port_names
            .output_zip(&signature)
            .map(|(n, t)| (Some(n.clone()), t.clone()));
        Self {
            name,
            description,
            inputs,
            outputs: outputs.collect(),
            misc: HashMap::new(),
            def: None,
            resource_reqs: ResourceSet::new(),
            signature: OnceCell::with_value(signature),
            port_names: OnceCell::with_value(port_names),
        }
    }

    /// The signature of the operation.
    pub fn signature(&self) -> Signature {
        self.signature
            .get_or_init(|| {
                let inputs = self
                    .inputs
                    .iter()
                    .map(|(_, t)| t.clone())
                    .collect::<Vec<_>>();
                let outputs = self
                    .outputs
                    .iter()
                    .map(|(_, t)| t.clone())
                    .collect::<Vec<_>>();
                Signature::new_df(inputs, outputs)
            })
            .clone()
    }

    /// Optional description of the ports in the signature.
    pub fn signature_desc(&self) -> Option<SignatureDescription> {
        Some(
            self.port_names
                .get_or_init(|| {
                    let inputs = self
                        .inputs
                        .iter()
                        .map(|(n, _)| n.clone().unwrap_or_default())
                        .collect::<Vec<_>>();
                    let outputs = self
                        .outputs
                        .iter()
                        .map(|(n, _)| n.clone().unwrap_or_default())
                        .collect::<Vec<_>>();
                    SignatureDescription::new_df(inputs, outputs)
                })
                .clone(),
        )
    }
}
