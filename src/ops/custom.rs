//! Extensible operations.

use downcast_rs::{impl_downcast, Downcast};
use once_cell::sync::OnceCell;
use smol_str::SmolStr;
use std::any::Any;
use std::collections::HashMap;

use super::Op;
use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::resource::ResourceSet;
use crate::types::SimpleType;
use crate::types::{Signature, SignatureDescription};

/// A wrapped custom operation with fast equality checks.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OpaqueOp {
    pub id: SmolStr,
    pub custom_op: Box<dyn CustomOp>,
}

impl PartialEq for OpaqueOp {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for OpaqueOp {}

impl Op for OpaqueOp {
    fn name(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        self.custom_op.description()
    }

    fn signature(&self) -> Signature {
        self.custom_op.signature()
    }

    fn signature_desc(&self) -> Option<&SignatureDescription> {
        self.custom_op.signature_desc()
    }
}

impl<T: CustomOp> From<T> for OpaqueOp {
    fn from(op: T) -> Self {
        Self {
            id: op.name().into(),
            custom_op: Box::new(op),
        }
    }
}

/// Custom definition for an operation.
///
/// Note that any implementation of this trait must include the `#[typetag::serde]` attribute.
#[typetag::serde]
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Op + Any + Downcast {
    /// Try to convert the custom op to a graph definition.
    ///
    /// TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn try_into_hugr(&self, resources: &ResourceSet) -> Option<Hugr> {
        let _ = resources;
        None
    }

    /// List the resources required to execute this operation.
    fn resources(&self) -> &ResourceSet;
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);

/// Dynamically loaded operation definition.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpDef {
    /// Unique identifier of the operation.
    ///
    /// This is used to compare two custom ops for equality.
    pub name: SmolStr,
    /// Human readable description of the operation.
    pub description: String,
    inputs: Vec<(Option<String>, SimpleType)>,
    outputs: Vec<(Option<String>, SimpleType)>,
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
        let inputs = port_names
            .input_zip(&signature)
            .chain(port_names.const_input_zip(&signature))
            .map(|(n, t)| (Some(n.clone()), t.clone()));
        let outputs = port_names
            .output_zip(&signature)
            .chain(port_names.const_output_zip(&signature))
            .map(|(n, t)| (Some(n.clone()), t.clone()));
        Self {
            name,
            description,
            inputs: inputs.collect(),
            outputs: outputs.collect(),
            misc: HashMap::new(),
            def: None,
            resource_reqs: ResourceSet::new(),
            signature: OnceCell::with_value(signature),
            port_names: OnceCell::with_value(port_names),
        }
    }
}

impl Op for OpDef {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn signature(&self) -> Signature {
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

    fn signature_desc(&self) -> Option<&SignatureDescription> {
        Some(self.port_names.get_or_init(|| {
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
        }))
    }
}

#[typetag::serde]
impl CustomOp for OpDef {
    fn try_into_hugr(&self, _resources: &ResourceSet) -> Option<Hugr> {
        todo!("Parse definition, check the available resources, and create a HUGR.")
    }

    fn resources(&self) -> &ResourceSet {
        &self.resource_reqs
    }
}
