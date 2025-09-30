//! Serialized format for [`CommitStateSpace`]

use relrc::serialization::SerializedRegistry;

use crate::serial::SerialPersistentHugr;

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

/// Serialize a [`CommitStateSpace`] alongside a set of [`PersistentHugr`]s in
/// that space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerialCommitStateSpace<H> {
    pub registry: SerializedRegistry<SerialCommitData<H>, ()>,
    pub hugrs: Vec<SerialPersistentHugr>,
}

impl<H> SerialCommitStateSpace<H> {
    pub fn new(state_space: &CommitStateSpace) -> Self
    where
        H: From<Hugr>,
    {
        let registry = state_space.as_registry().borrow().to_serialized();
        let registry = registry.map_nodes(|n| n.into_serial());
        Self {
            registry,
            hugrs: Vec::new(),
        }
    }

    pub fn add_hugr(&mut self, hugr: PersistentHugr) {
        self.hugrs.push(hugr.to_serial());
    }

    pub fn deserialize_into_hugrs(self) -> Vec<PersistentHugr>
    where
        H: Clone + Into<Hugr>,
    {
        let registry = self.registry.map_nodes(|n| CommitData::from_serial(n));
        let (registry, all_relrcs) = Registry::from_serialized(registry);
        let state_space = CommitStateSpace::from(registry);
        for (exp_node, rc) in &all_relrcs {
            let node = rc
                .try_register_in(state_space.as_registry())
                .expect("deserialised rc is not registered");
            debug_assert_eq!(
                exp_node, node,
                "a new node ID was assigned to a node already in registry"
            );
        }
        self.hugrs
            .into_iter()
            .map(|h| PersistentHugr::from_serial(h, &state_space))
            .collect()
    }
}

impl PersistentHugr {
    /// Create a new [`CommitStateSpace`] from its serialized format
    pub fn from_serial_state_space<H: Clone + Into<Hugr>>(
        value: SerialCommitStateSpace<H>,
    ) -> Vec<Self> {
        value.deserialize_into_hugrs()
    }
}

impl CommitStateSpace {
    /// Convert a [`CommitStateSpace`] into its serialized format
    pub fn to_serial<H: From<Hugr>>(&self) -> SerialCommitStateSpace<H> {
        SerialCommitStateSpace::new(self)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rstest::rstest;

    use super::*;
    use crate::tests::{TestStateSpace, WrappedHugr, test_state_space};

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[rstest]
    fn test_serialize_state_space(test_state_space: TestStateSpace) {
        let commits: &[_; 4] = test_state_space.commits();
        let state_space = commits[0].state_space();
        let base_id = state_space.base_commit().unwrap().id();
        let mut ser_state_space = state_space.to_serial::<WrappedHugr>();

        let deser = PersistentHugr::from_serial_state_space(ser_state_space.clone());
        assert!(deser.is_empty());

        let cm_set1 = commits[..2].to_owned();
        let cm_set2 = commits[..2]
            .iter()
            .chain([&commits[3]])
            .cloned()
            .collect_vec();

        ser_state_space.add_hugr(PersistentHugr::try_new(cm_set1.clone()).unwrap());
        ser_state_space.add_hugr(PersistentHugr::try_new(cm_set2.clone()).unwrap());

        let deser = PersistentHugr::from_serial_state_space(ser_state_space.clone());

        let [first, second] = deser.as_slice() else {
            panic!("there should be two deserialized hugrs")
        };
        assert_eq!(
            first.all_commit_ids().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(cm_set1.iter().map(|c| c.id()).chain([base_id]))
        );

        assert_eq!(
            second.all_commit_ids().collect::<BTreeSet<_>>(),
            BTreeSet::from_iter(cm_set2.iter().map(|c| c.id()).chain([base_id]))
        );

        insta::assert_yaml_snapshot!(ser_state_space);
    }
}
