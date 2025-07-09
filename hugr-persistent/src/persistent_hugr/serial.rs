//! Serialized format for [`PersistentHugr`]

use hugr_core::Hugr;

use crate::{CommitStateSpace, Resolver, state_space::serial::SerialCommitStateSpace};

use super::PersistentHugr;

/// Serialized format for [`PersistentHugr`]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerialPersistentHugr<H, R> {
    /// The state space of all commits.
    state_space: SerialCommitStateSpace<H, R>,
}

impl<R: Resolver> PersistentHugr<R> {
    /// Create a new [`CommitStateSpace`] from its serialized format
    pub fn from_serial<H: Into<Hugr>>(value: SerialPersistentHugr<H, R>) -> Self {
        let SerialPersistentHugr { state_space } = value;
        let state_space = CommitStateSpace::from_serial(state_space);
        Self { state_space }
    }

    /// Convert a [`CommitStateSpace`] into its serialized format
    pub fn into_serial<H: From<Hugr>>(self) -> SerialPersistentHugr<H, R> {
        let Self { state_space } = self;
        let state_space = state_space.into_serial();
        SerialPersistentHugr { state_space }
    }

    /// Create a serialized format from a reference to [`CommitStateSpace`]
    pub fn to_serial<H: From<Hugr>>(&self) -> SerialPersistentHugr<H, R> {
        let Self { state_space } = self;
        let state_space = state_space.to_serial();
        SerialPersistentHugr { state_space }
    }
}

impl<H: From<Hugr>, R: Resolver> From<PersistentHugr<R>> for SerialPersistentHugr<H, R> {
    fn from(value: PersistentHugr<R>) -> Self {
        value.into_serial()
    }
}

impl<H: Into<Hugr>, R: Resolver> From<SerialPersistentHugr<H, R>> for PersistentHugr<R> {
    fn from(value: SerialPersistentHugr<H, R>) -> Self {
        PersistentHugr::from_serial(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommitId, SerdeHashResolver,
        tests::{WrappedHugr, test_state_space},
    };

    use rstest::rstest;

    #[rstest]
    fn test_serde_persistent_hugr(
        test_state_space: (
            CommitStateSpace<SerdeHashResolver<WrappedHugr>>,
            [CommitId; 4],
        ),
    ) {
        let (state_space, [cm1, cm2, _, cm4]) = test_state_space;

        let per_hugr = state_space.try_extract_hugr([cm1, cm2, cm4]).unwrap();
        let ser_per_hugr = per_hugr.to_serial::<WrappedHugr>();

        insta::assert_snapshot!(serde_json::to_string_pretty(&ser_per_hugr).unwrap());
    }
}
