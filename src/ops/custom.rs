//! Extensible operations.

use downcast_rs::{impl_downcast, Downcast};
use once_cell::sync::OnceCell;
use std::any::Any;
use std::collections::HashMap;

use super::Op;
use crate::hugr::Hugr;
use crate::macros::impl_box_clone;
use crate::types::DataType;
use crate::types::{Signature, SignatureDescription};

/// Custom definition for an operation.
///
/// Note that any implementation of this trait must include the `#[typetag::serde]` attribute.
#[typetag::serde]
pub trait CustomOp: Send + Sync + std::fmt::Debug + CustomOpBoxClone + Op + Any + Downcast {
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
    name: String,
    description: String,
    inputs: Vec<(Option<String>, DataType)>,
    outputs: Vec<(Option<String>, DataType)>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    misc: HashMap<String, serde_yaml::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    def: Option<String>,

    /// Signature of the operation. Computed from `inputs` and `outputs`.
    #[serde(skip)]
    signature: OnceCell<Signature>,
    #[serde(skip)]
    port_names: OnceCell<SignatureDescription>,
}

impl OpDef {
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
    fn eq(&self, other: &dyn CustomOp) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.name == other.name && self.signature == other.signature
        } else {
            false
        }
    }

    fn try_to_hugr(&self) -> Option<Hugr> {
        todo!()
    }
}
