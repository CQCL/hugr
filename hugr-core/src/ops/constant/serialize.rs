//! Helper definitions used to serialize constant values and ops.

use itertools::Itertools;

use crate::ops::Value;
use crate::types::SumType;
use crate::types::serialize::SerSimpleType;

use super::Sum;

/// Helper struct to serialize constant [`Sum`] values with a custom layout.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct SerialSum {
    #[serde(default)]
    tag: usize,
    #[serde(rename = "vs")]
    values: Vec<Value>,
    /// Uses the `SerSimpleType` wrapper here instead of a direct `SumType`,
    /// to ensure it gets correctly tagged with the `t` discriminant field.
    #[serde(default, rename = "typ")]
    sum_type: Option<SerSimpleType>,
}

impl From<Sum> for SerialSum {
    fn from(value: Sum) -> Self {
        Self {
            tag: value.tag,
            values: value.values,
            sum_type: Some(SerSimpleType::Sum(value.sum_type)),
        }
    }
}

impl TryFrom<SerialSum> for Sum {
    type Error = &'static str;

    fn try_from(value: SerialSum) -> Result<Self, Self::Error> {
        let SerialSum {
            tag,
            values,
            sum_type,
        } = value;

        let sum_type = if let Some(SerSimpleType::Sum(sum_type)) = sum_type {
            sum_type
        } else {
            if tag != 0 {
                return Err("Sum type must be provided if tag is not 0");
            }
            SumType::new_tuple(values.iter().map(Value::get_type).collect_vec())
        };

        Ok(Self {
            tag,
            values,
            sum_type,
        })
    }
}
