//! Serialization and deserialization.

use crate::{
    Hugr,
    extension::ExtensionRegistry,
    hugr::{
        persistent::{CommitId, PersistentReplacement},
        serialize::ExtensionsSeed,
    },
};
use relrc::{
    HistoryGraph,
    serialization::{SerializedHistoryGraph, SerializedInnerData},
};
use serde::{Deserialize, Serialize, de::VariantAccess};
use std::collections::BTreeMap;

use super::{CommitStateSpace, PointerEqResolver, state_space::CommitData};

impl<'de> serde::de::DeserializeSeed<'de> for ExtensionsSeed<'_, CommitData> {
    type Value = CommitData;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier)]
        enum Fields {
            Base,
            Replacement,
        }
        struct Visitor<'a> {
            extensions: &'a ExtensionRegistry,
        }
        impl<'de> serde::de::Visitor<'de> for Visitor<'_> {
            type Value = CommitData;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a CommitData")
            }

            fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::EnumAccess<'de>,
            {
                match data.variant()? {
                    (Fields::Base, variant) => {
                        let hugr = variant
                            .newtype_variant_seed(ExtensionsSeed::<Hugr>::new(self.extensions))?;
                        Ok(CommitData::Base(hugr))
                    }
                    (Fields::Replacement, variant) => {
                        let replacement =
                            variant.newtype_variant_seed(
                                ExtensionsSeed::<PersistentReplacement>::new(self.extensions),
                            )?;
                        Ok(CommitData::Replacement(replacement))
                    }
                }
            }
        }
        deserializer.deserialize_enum(
            "CommitData",
            &["Base", "Replacement"],
            Visitor {
                extensions: self.extensions,
            },
        )
    }
}

pub(super) fn serialize_history_graph<S: serde::Serializer>(
    graph: &relrc::HistoryGraph<CommitData, (), PointerEqResolver>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let ser_graph = graph.to_serialized();
    ser_graph.serialize(serializer)
}

impl<'de> serde::de::DeserializeSeed<'de> for ExtensionsSeed<'_, CommitStateSpace> {
    type Value = CommitStateSpace;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Fields {
            Graph,
            BaseCommit,
        }
        struct Visitor<'a> {
            extensions: &'a ExtensionRegistry,
        }
        impl<'de> serde::de::Visitor<'de> for Visitor<'_> {
            type Value = CommitStateSpace;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a CommitStateSpace")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut graph = None;
                let mut base_commit = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Fields::Graph => {
                            let ser_graph = map.next_value_seed(ExtensionsSeed::<
                                SerializedHistoryGraph<CommitData, (), PointerEqResolver>,
                            >::new(
                                self.extensions
                            ))?;
                            graph = Some(
                                HistoryGraph::try_from_serialized(ser_graph, PointerEqResolver)
                                    .expect("support SerdeHashResolver"),
                            );
                        }
                        Fields::BaseCommit => {
                            base_commit = Some(map.next_value()?);
                        }
                    }
                }

                let graph = graph.ok_or(serde::de::Error::missing_field("graph"))?;
                let base_commit =
                    base_commit.ok_or(serde::de::Error::missing_field("base_commit"))?;

                Ok(CommitStateSpace { graph, base_commit })
            }
        }

        serde::de::Deserializer::deserialize_struct(
            deserializer,
            "CommitStateSpace",
            &["graph", "base_commit"],
            Visitor {
                extensions: self.extensions,
            },
        )
    }
}

impl<'de, E: Deserialize<'de>> serde::de::DeserializeSeed<'de>
    for ExtensionsSeed<'_, SerializedInnerData<CommitData, E>>
{
    type Value = SerializedInnerData<CommitData, E>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Fields {
            Value,
            Incoming,
        }
        struct Visitor<'a, N, E> {
            extensions: &'a ExtensionRegistry,
            _marker: std::marker::PhantomData<(N, E)>,
        }
        impl<'de, 'a, N, E: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<'a, N, E>
        where
            ExtensionsSeed<'a, N>: serde::de::DeserializeSeed<'de, Value = N>,
        {
            type Value = SerializedInnerData<N, E>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a SerializedInnerData")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut value = None;
                let mut incoming = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Fields::Value => {
                            value = Some(
                                map.next_value_seed(ExtensionsSeed::<N>::new(self.extensions))?,
                            );
                        }
                        Fields::Incoming => {
                            incoming = Some(map.next_value()?);
                        }
                    }
                }

                let value = value.ok_or(serde::de::Error::missing_field("value"))?;
                let incoming = incoming.ok_or(serde::de::Error::missing_field("incoming"))?;

                Ok(SerializedInnerData { value, incoming })
            }
        }

        serde::de::Deserializer::deserialize_struct(
            deserializer,
            "SerializedInnerData",
            &["value", "incoming"],
            Visitor {
                _marker: std::marker::PhantomData,
                extensions: self.extensions,
            },
        )
    }
}

impl<'de, E: Deserialize<'de>, R: Deserialize<'de>> serde::de::DeserializeSeed<'de>
    for ExtensionsSeed<'_, SerializedHistoryGraph<CommitData, E, R>>
{
    type Value = SerializedHistoryGraph<CommitData, E, R>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Fields {
            Nodes,
            ResolverId,
        }
        struct Visitor<'a, E, R> {
            extensions: &'a ExtensionRegistry,
            _marker: std::marker::PhantomData<(E, R)>,
        }
        impl<'de, E: Deserialize<'de>, R: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<'_, E, R> {
            type Value = SerializedHistoryGraph<CommitData, E, R>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a SerializedHistoryGraph")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut nodes = None;
                let mut resolver_id = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Fields::Nodes => {
                            nodes = Some(map.next_value_seed(ExtensionsSeed::<
                                BTreeMap<CommitId, SerializedInnerData<CommitData, E>>,
                            >::new(
                                self.extensions
                            ))?);
                        }
                        Fields::ResolverId => {
                            resolver_id = Some(map.next_value()?);
                        }
                    }
                }

                let nodes = nodes.ok_or(serde::de::Error::missing_field("nodes"))?;
                let resolver_id =
                    resolver_id.ok_or(serde::de::Error::missing_field("resolver_id"))?;

                Ok(SerializedHistoryGraph { nodes, resolver_id })
            }
        }

        serde::de::Deserializer::deserialize_struct(
            deserializer,
            "SerializedHistoryGraph",
            &["nodes", "resolver_id"],
            Visitor {
                _marker: std::marker::PhantomData,
                extensions: self.extensions,
            },
        )
    }
}

impl<'de, K: Ord + Deserialize<'de>, E: Deserialize<'de>> serde::de::DeserializeSeed<'de>
    for ExtensionsSeed<'_, BTreeMap<K, SerializedInnerData<CommitData, E>>>
{
    type Value = BTreeMap<K, SerializedInnerData<CommitData, E>>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MapVisitor<'a, K, V> {
            extensions: &'a ExtensionRegistry,
            _marker: std::marker::PhantomData<(K, V)>,
        }

        impl<'de, 'a, K: Ord + Deserialize<'de>, V> serde::de::Visitor<'de> for MapVisitor<'a, K, V>
        where
            ExtensionsSeed<'a, V>: serde::de::DeserializeSeed<'de, Value = V>,
        {
            type Value = BTreeMap<K, V>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a map")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut result = BTreeMap::new();
                while let Some(key) = map.next_key()? {
                    let value = map.next_value_seed(ExtensionsSeed::<V>::new(self.extensions))?;
                    result.insert(key, value);
                }
                Ok(result)
            }
        }

        deserializer.deserialize_map(MapVisitor {
            extensions: self.extensions,
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::hugr::persistent::tests::test_state_space;

    use super::*;

    use rstest::rstest;
    use serde::de::DeserializeSeed;

    #[rstest]
    fn test_serialize_deserialize(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let state_space = test_state_space.0;

        let ser = serde_json::to_string_pretty(&state_space).unwrap();
        let mut deserializer = serde_json::Deserializer::from_str(&ser);
        let deser = ExtensionsSeed::<CommitStateSpace>::empty()
            .deserialize(&mut deserializer)
            .unwrap();
        assert_eq!(
            state_space.all_commit_ids().count(),
            deser.all_commit_ids().count()
        );
    }
}
