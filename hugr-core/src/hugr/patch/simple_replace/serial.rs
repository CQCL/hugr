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

#[cfg(test)]
mod test {
    use super::super::test::*;
    use super::*;
    use crate::{envelope::serde_with::AsStringEnvelope, utils::test_quantum_extension::cx_gate};

    use derive_more::derive::{From, Into};
    use rstest::rstest;
    use serde_with::serde_as;

    #[serde_as]
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, From, Into)]
    struct WrappedHugr {
        #[serde_as(as = "AsStringEnvelope")]
        pub hugr: Hugr,
    }

    impl<'h> From<&'h Hugr> for WrappedHugr {
        fn from(value: &'h Hugr) -> Self {
            WrappedHugr {
                hugr: value.clone(),
            }
        }
    }

    #[rstest]
    fn test_serial(simple_hugr: Hugr, dfg_hugr: Hugr) {
        let h: Hugr = simple_hugr;
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .entry_descendants()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let replacement = dfg_hugr;
        // 4. Define the replacement
        let r = SimpleReplacement {
            subgraph: SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            replacement,
        };

        let other_repl_serial = r.to_serial::<WrappedHugr>();
        let repl_serial = r.into_serial::<WrappedHugr>();

        assert_eq!(repl_serial, other_repl_serial);
    }
}
