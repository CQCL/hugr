use relrc::serialization::SerializedHistoryGraph;

use super::*;
use crate::hugr::patch::simple_replace::serial::SerialSimpleReplacement;

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
    pub graph: SerializedHistoryGraph<SerialCommitData<H>, (), PointerEqResolver>,
    /// The base commit ID
    pub base_commit: CommitId,
}

impl CommitStateSpace {
    /// Create a new [`CommitStateSpace`] from its serialized format
    pub fn from_serial<H: Into<Hugr> + Clone>(value: SerialCommitStateSpace<H>) -> Self {
        let SerialCommitStateSpace { graph, base_commit } = value;

        // Deserialize the SerializedHistoryGraph into a HistoryGraph with CommitData
        let graph = graph.map_nodes(|n| CommitData::from_serial(n));
        let graph = HistoryGraph::try_from_serialized(graph, PointerEqResolver)
            .expect("failed to deserialize history graph");

        Self { graph, base_commit }
    }

    /// Convert a [`CommitStateSpace`] into its serialized format
    pub fn into_serial<H: From<Hugr>>(self) -> SerialCommitStateSpace<H> {
        let Self { graph, base_commit } = self;
        let graph = graph.to_serialized();
        let graph = graph.map_nodes(|n| n.into_serial());
        SerialCommitStateSpace { graph, base_commit }
    }

    /// Create a serialized format from a reference to [`CommitStateSpace`]
    pub fn to_serial<H>(&self) -> SerialCommitStateSpace<H>
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

impl<H: From<Hugr>> From<CommitStateSpace> for SerialCommitStateSpace<H> {
    fn from(value: CommitStateSpace) -> Self {
        value.into_serial()
    }
}

impl<H: Clone + Into<Hugr>> From<SerialCommitStateSpace<H>> for CommitStateSpace {
    fn from(value: SerialCommitStateSpace<H>) -> Self {
        CommitStateSpace::from_serial(value)
    }
}

#[cfg(test)]
mod tests {
    use derive_more::derive::Into;
    use rstest::rstest;
    use serde_with::serde_as;

    use super::*;
    use crate::{
        envelope::serde_with::AsStringEnvelope, hugr::persistent::tests::test_state_space,
    };

    #[serde_as]
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, From, Into)]
    pub struct WrappedHugr {
        #[serde_as(as = "AsStringEnvelope")]
        pub hugr: Hugr,
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[rstest]
    fn test_serialize_state_space(test_state_space: (CommitStateSpace, [CommitId; 4])) {
        let (state_space, _) = test_state_space;
        let serialized = state_space.to_serial::<WrappedHugr>();

        let deser = CommitStateSpace::from_serial(serialized);
        let _serialized_2 = deser.to_serial::<WrappedHugr>();

        // TODO: add this once PointerEqResolver is replaced by a deterministic resolver
        // insta::assert_snapshot!(serde_json::to_string_pretty(&serialized).unwrap());
        // see https://github.com/CQCL/hugr/issues/2299
        // assert_eq!(
        //     serde_json::to_string(&serialized),
        //     serde_json::to_string(&serialized_2)
        // );
    }
}
