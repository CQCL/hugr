//! Serialized format for [`PersistentHugr`]

use crate::{CommitId, CommitStateSpace};

use super::PersistentHugr;

/// Serialized format for [`PersistentHugr`]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerialPersistentHugr {
    /// The state space of all commits.
    pub commits: Vec<CommitId>,
}

impl PersistentHugr {
    /// Create a new [`CommitStateSpace`] from its serialized format
    pub fn from_serial(value: SerialPersistentHugr, state_space: &CommitStateSpace) -> Self {
        state_space.try_create(value.commits).unwrap()
    }

    /// Convert a [`CommitStateSpace`] into its serialized format
    pub fn to_serial(&self) -> SerialPersistentHugr {
        SerialPersistentHugr {
            commits: self.all_commit_ids().collect(),
        }
    }
}

impl From<PersistentHugr> for SerialPersistentHugr {
    fn from(value: PersistentHugr) -> Self {
        value.to_serial()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{TestStateSpace, test_state_space};

    use rstest::rstest;

    #[rstest]
    fn test_serde_persistent_hugr(test_state_space: TestStateSpace) {
        let [cm1, cm2, _, cm4] = test_state_space.commits();

        let per_hugr = PersistentHugr::try_new([cm1.clone(), cm2.clone(), cm4.clone()]).unwrap();
        let ser_per_hugr = per_hugr.to_serial();

        insta::assert_snapshot!(serde_json::to_string_pretty(&ser_per_hugr).unwrap());
    }
}
