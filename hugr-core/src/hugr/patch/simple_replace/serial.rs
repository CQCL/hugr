//! Serialisation of [`SimpleReplacement`]

use super::*;

/// Serialized format for [`SimpleReplacement`]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerialSimpleReplacement<H, N> {
    /// The subgraph to be replaced
    pub subgraph: SiblingSubgraph<N>,
    /// The replacement Hugr
    pub replacement: H,
}

impl<N> SimpleReplacement<N> {
    /// Create a new [`SimpleReplacement`] from its serialized format
    pub fn from_serial<H: Into<Hugr>>(value: SerialSimpleReplacement<H, N>) -> Self {
        let SerialSimpleReplacement {
            subgraph,
            replacement,
        } = value;
        SimpleReplacement {
            subgraph,
            replacement: replacement.into(),
        }
    }

    /// Convert a [`SimpleReplacement`] into its serialized format
    pub fn into_serial<H: From<Hugr>>(self) -> SerialSimpleReplacement<H, N> {
        let SimpleReplacement {
            subgraph,
            replacement,
        } = self;
        SerialSimpleReplacement {
            subgraph,
            replacement: replacement.into(),
        }
    }

    /// Create its serialized format from a reference to [`SimpleReplacement`]
    pub fn to_serial<'a, H>(&'a self) -> SerialSimpleReplacement<H, N>
    where
        N: Clone,
        H: From<&'a Hugr>,
    {
        let SimpleReplacement {
            subgraph,
            replacement,
        } = self;
        SerialSimpleReplacement {
            subgraph: subgraph.clone(),
            replacement: replacement.into(),
        }
    }
}

impl<N, H: From<Hugr>> From<SimpleReplacement<N>> for SerialSimpleReplacement<H, N> {
    fn from(value: SimpleReplacement<N>) -> Self {
        value.into_serial()
    }
}

impl<H: Into<Hugr>, N> From<SerialSimpleReplacement<H, N>> for SimpleReplacement<N> {
    fn from(value: SerialSimpleReplacement<H, N>) -> Self {
        SimpleReplacement::from_serial(value)
    }
}
