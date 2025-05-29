use serde::de::DeserializeSeed;

use crate::{Hugr, core::HugrNode, extension::ExtensionRegistry, hugr::serialize::ExtensionsSeed};

use super::SimpleReplacement;

impl<'de, N: HugrNode + serde::Deserialize<'de>> DeserializeSeed<'de>
    for ExtensionsSeed<'_, SimpleReplacement<N>>
{
    type Value = SimpleReplacement<N>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Fields {
            Subgraph,
            Replacement,
        }
        struct Visitor<'a, N> {
            extensions: &'a ExtensionRegistry,
            _marker: std::marker::PhantomData<N>,
        }
        impl<'de, N: HugrNode + serde::Deserialize<'de>> serde::de::Visitor<'de> for Visitor<'_, N> {
            type Value = SimpleReplacement<N>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a SimpleReplacement")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut subgraph = None;
                let mut replacement = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Fields::Subgraph => {
                            subgraph = Some(map.next_value()?);
                        }
                        Fields::Replacement => {
                            replacement = Some(
                                map.next_value_seed(ExtensionsSeed::<Hugr>::new(self.extensions))?,
                            );
                        }
                    }
                }

                let subgraph = subgraph.ok_or(serde::de::Error::missing_field("subgraph"))?;
                let replacement =
                    replacement.ok_or(serde::de::Error::missing_field("replacement"))?;

                Ok(SimpleReplacement::new_unchecked(subgraph, replacement))
            }
        }

        serde::de::Deserializer::deserialize_struct(
            deserializer,
            "SimpleReplacement",
            &["subgraph", "replacement"],
            Visitor {
                _marker: std::marker::PhantomData,
                extensions: self.extensions,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};
    use crate::extension::prelude::bool_t;
    use crate::hugr::serialize::ExtensionsSeed;
    use crate::hugr::views::SiblingSubgraph;
    use crate::types::Signature;
    use crate::{Hugr, HugrView, IncomingPort, OutgoingPort, SimpleReplacement};

    use super::super::test::copy_not_not_copy_hugr;

    use itertools::Itertools;
    use rstest::rstest;
    use serde::de::DeserializeSeed;

    #[rstest]
    fn test_serialize_deserialize(copy_not_not_copy_hugr: Hugr) {
        let hugr = copy_not_not_copy_hugr;
        let [inp, _out] = hugr.get_io(hugr.entrypoint()).unwrap();
        let [not1, not2] = hugr.output_neighbours(inp).collect_array().unwrap();
        let subg_incoming = vec![vec![
            (not1, IncomingPort::from(0)),
            (not2, IncomingPort::from(0)),
        ]];
        let subg_outgoing = [not1, not2].map(|n| (n, OutgoingPort::from(0))).to_vec();

        let subgraph = SiblingSubgraph::try_new(subg_incoming, subg_outgoing, &hugr).unwrap();

        // Create an empty replacement (just copies)
        let repl = {
            let b = DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(); 2])).unwrap();
            let [w] = b.input_wires_arr();
            let repl_hugr = b.finish_hugr_with_outputs([w, w]).unwrap();
            SimpleReplacement::try_new(subgraph, &hugr, repl_hugr).unwrap()
        };

        let ser = serde_json::to_string_pretty(&repl).unwrap();
        let mut deserializer = serde_json::Deserializer::from_str(&ser);
        let deser = ExtensionsSeed::<SimpleReplacement>::empty()
            .deserialize(&mut deserializer)
            .unwrap();
        assert_eq!(repl.subgraph(), deser.subgraph());
        assert_eq!(
            repl.replacement().num_nodes(),
            deser.replacement().num_nodes()
        );
        insta::assert_snapshot!(ser);
    }
}
