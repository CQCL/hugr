//! Extensible operations.

use downcast_rs::{impl_downcast, Downcast};
use once_cell::sync::OnceCell;
use smol_str::SmolStr;
use std::any::Any;
use std::collections::HashMap;

use super::Op;
use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::types::SimpleType;
use crate::types::{Signature, SignatureDescription};

/// Custom definition for an operation.
///
/// Note that any implementation of this trait must include the `#[typetag::serde]` attribute.
#[typetag::serde]
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Op + Any + Downcast {
    /// Get the an unique identifier of the custom op.
    ///
    /// This is used to compare two custom ops for equality.
    fn name(&self) -> &SmolStr;

    /// Try to convert the custom op to a graph definition.
    ///
    /// TODO: Create a separate HUGR, or create a children subgraph in the HUGR?
    fn try_to_hugr(&self) -> Option<Hugr> {
        None
    }

    /// Check if two custom ops are equal, by downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomOp) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomOp);
impl_box_clone!(CustomOp, CustomOpBoxClone);

/// Dynamically loaded operation definition.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
            signature: OnceCell::with_value(signature),
            port_names: OnceCell::with_value(port_names),
        }
    }

    /// Miscellaneous data associated with the operation.
    pub fn misc(&self) -> &HashMap<String, serde_yaml::Value> {
        &self.misc
    }

    /// Definition of the operation.
    pub fn def(&self) -> Option<&str> {
        self.def.as_deref()
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
    fn name(&self) -> &SmolStr {
        &self.name
    }

    fn try_to_hugr(&self) -> Option<Hugr> {
        todo!()
    }

    fn eq(&self, other: &dyn CustomOp) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.name == other.name && self.signature == other.signature
        } else {
            false
        }
    }
}
