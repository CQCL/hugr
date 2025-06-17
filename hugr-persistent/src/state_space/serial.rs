//! Serialized format for [`CommitStateSpace`]

use relrc::serialization::SerializedHistoryGraph;

use super::*;
use hugr_core::hugr::patch::simple_replace::serial::SerialSimpleReplacement;

/// Serialized format for [`PersistentReplacement`]
pub type SerialPersistentReplacement<H> = SerialSimpleReplacement<H, PatchNode>;

/// Serialized format for CommitData
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SerialCommitData<H> {
    /// Base commit containing a Hugr
    Base(H),
    /// Replacement commit containing a serialized replacement
    Replacement(SerialPersistentReplacement<H>),
}

impl CommitData {
    /// Create a new [`CommitData`]` from its serialized format
    pub fn from_serial<H: Into<Hugr>>(value: SerialCommitData<H>) -> Self {
        match value {
            SerialCommitData::Base(h) => CommitData::Base(h.into()),
            SerialCommitData::Replacement(replacement) => {
                CommitData::Replacement(replacement.into())
            }
        }
    }

    /// Convert this [`CommitData`] into its serialized format
    pub fn into_serial<H: From<Hugr>>(self) -> SerialCommitData<H> {
        match self {
            CommitData::Base(h) => SerialCommitData::Base(h.into()),
            CommitData::Replacement(replacement) => {
                SerialCommitData::Replacement(replacement.into_serial())
            }
        }
    }
}

impl<H: From<Hugr>> From<CommitData> for SerialCommitData<H> {
    fn from(value: CommitData) -> Self {
        value.into_serial()
    }
}

impl<H: Into<Hugr>> From<SerialCommitData<H>> for CommitData {
    fn from(value: SerialCommitData<H>) -> Self {
        CommitData::from_serial(value)
    }
}

/// Serialized format for commit state space
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerialCommitStateSpace<H> {
    /// The serialized history graph containing commit data
    pub graph: SerializedHistoryGraph<SerialCommitData<H>, (), SerdeHashResolver<H>>,
    /// The base commit ID
    pub base_commit: CommitId,
}

impl<H: Into<Hugr> + From<Hugr> + serde::Serialize> CommitStateSpace<SerdeHashResolver<H>> {
    /// Create a new [`CommitStateSpace`] from its serialized format
    pub fn from_serial(value: SerialCommitStateSpace<H>) -> Self {
        let SerialCommitStateSpace { graph, base_commit } = value;

        // Deserialize the SerializedHistoryGraph into a HistoryGraph with CommitData
        let graph = graph.map_nodes(|n| CommitData::from_serial(n));
        let graph = HistoryGraph::try_from_serialized(graph, SerdeHashResolver::default())
            .expect("failed to deserialize history graph");

        Self { graph, base_commit }
    }

    /// Convert a [`CommitStateSpace`] into its serialized format
    pub fn into_serial(self) -> SerialCommitStateSpace<H> {
        let Self { graph, base_commit } = self;
        let graph = graph.to_serialized();
        let graph = graph.map_nodes(|n| n.into_serial());
        SerialCommitStateSpace { graph, base_commit }
    }

    /// Create a serialized format from a reference to [`CommitStateSpace`]
    pub fn to_serial(&self) -> SerialCommitStateSpace<H>
    where
        H: From<Hugr>,
    {
        let Self { graph, base_commit } = self;
        let graph = graph.to_serialized();
        let graph = graph.map_nodes(|n| n.into_serial());
        SerialCommitStateSpace {
            graph,
            base_commit: *base_commit,
        }
    }
}

impl<H: Into<Hugr> + From<Hugr> + serde::Serialize> From<CommitStateSpace<SerdeHashResolver<H>>>
    for SerialCommitStateSpace<H>
{
    fn from(value: CommitStateSpace<SerdeHashResolver<H>>) -> Self {
        value.into_serial()
    }
}

impl<H: Into<Hugr> + From<Hugr> + serde::Serialize> From<SerialCommitStateSpace<H>>
    for CommitStateSpace<SerdeHashResolver<H>>
{
    fn from(value: SerialCommitStateSpace<H>) -> Self {
        CommitStateSpace::from_serial(value)
    }
}

#[cfg(test)]
mod tests {
    use derive_more::derive::Into;
    use hugr_core::envelope::serde_with::AsStringEnvelope;
    use rstest::rstest;
    use serde_with::serde_as;

    use super::*;
    use crate::tests::test_state_space;

    #[serde_as]
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, From, Into)]
    struct WrappedHugr {
        #[serde_as(as = "AsStringEnvelope")]
        pub hugr: Hugr,
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[rstest]
    fn test_serialize_state_space(
        test_state_space: (
            CommitStateSpace<SerdeHashResolver<WrappedHugr>>,
            [CommitId; 4],
        ),
    ) {
        let (state_space, _) = test_state_space;
        let serialized = state_space.to_serial();

        let deser = CommitStateSpace::from_serial(serialized.clone());
        let serialized_2 = deser.to_serial();

        insta::assert_snapshot!(serde_json::to_string_pretty(&serialized).unwrap());
        assert_eq!(
            serde_json::to_string(&serialized).unwrap(),
            serde_json::to_string(&serialized_2).unwrap()
        );
    }
}
