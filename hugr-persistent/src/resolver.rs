use std::marker::PhantomData;

use hugr_core::Hugr;
use relrc::EquivalenceResolver;
use wyhash::wyhash; // a fast platform-independent hash function

use crate::state_space::CommitData;

/// A trait for resolvers that can be used in [`CommitStateSpace`](super::CommitStateSpace).
pub trait Resolver: Clone + Default + EquivalenceResolver<CommitData, ()> {}
impl<T: Clone + Default + EquivalenceResolver<CommitData, ()>> Resolver for T {}

/// A resolver that considers two nodes equivalent if they are the same pointer.
///
/// Resolvers determine when two patches are equivalent and should be merged
/// in the patch history.
///
/// This is a trivial resolver (to be expanded on later), that considers two
/// patches equivalent if they point to the same data in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PointerEqResolver;

impl<N, E: Clone> EquivalenceResolver<N, E> for PointerEqResolver {
    type MergeMapping = ();

    type DedupKey = *const N;

    fn id(&self) -> String {
        "PointerEqResolver".to_string()
    }

    fn dedup_key(&self, value: &N, _incoming_edges: &[&E]) -> Self::DedupKey {
        value as *const N
    }

    fn try_merge_mapping(
        &self,
        a_value: &N,
        _a_incoming_edges: &[&E],
        b_value: &N,
        _b_incoming_edges: &[&E],
    ) -> Result<Self::MergeMapping, relrc::resolver::NotEquivalent> {
        if std::ptr::eq(a_value, b_value) {
            Ok(())
        } else {
            Err(relrc::resolver::NotEquivalent)
        }
    }

    fn move_edge_source(&self, _mapping: &Self::MergeMapping, edge: &E) -> E {
        edge.clone()
    }
}

/// A resolver that considers two nodes equivalent if the hashes of their
/// serialisation is the same.
///
/// ### Generic type parameter
///
/// This is parametrised over a serializable type `H`, which must implement
/// [`From<Hugr>`]. This type is used to serialise the commit data before
/// hashing it.
///
/// Resolvers determine when two patches are equivalent and should be merged
/// in the patch history.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerdeHashResolver<H>(#[serde(skip)] PhantomData<H>);

impl<H> Default for SerdeHashResolver<H> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<H> SerdeHashResolver<H> {
    fn hash(value: &impl serde::Serialize) -> u64 {
        let bytes = serde_json::to_vec(value).unwrap();
        const SEED: u64 = 0;
        wyhash(&bytes, SEED)
    }
}

impl<H: serde::Serialize + From<Hugr>> EquivalenceResolver<CommitData, ()>
    for SerdeHashResolver<H>
{
    type MergeMapping = ();

    type DedupKey = u64;

    fn id(&self) -> String {
        "SerdeHashResolver".to_string()
    }

    fn dedup_key(&self, value: &CommitData, _incoming_edges: &[&()]) -> Self::DedupKey {
        let ser_value = value.clone().into_serial::<H>();
        Self::hash(&ser_value)
    }

    fn try_merge_mapping(
        &self,
        a_value: &CommitData,
        _a_incoming_edges: &[&()],
        b_value: &CommitData,
        _b_incoming_edges: &[&()],
    ) -> Result<Self::MergeMapping, relrc::resolver::NotEquivalent> {
        let a_ser_value = a_value.clone().into_serial::<H>();
        let b_ser_value = b_value.clone().into_serial::<H>();
        if Self::hash(&a_ser_value) == Self::hash(&b_ser_value) {
            Ok(())
        } else {
            Err(relrc::resolver::NotEquivalent)
        }
    }

    fn move_edge_source(&self, _mapping: &Self::MergeMapping, _edge: &()) {}
}

#[cfg(test)]
mod tests {
    use hugr_core::{builder::endo_sig, ops::FuncDefn};

    use super::*;
    use crate::{CommitData, tests::WrappedHugr};

    #[test]
    fn test_serde_hash_resolver_equality() {
        let resolver = SerdeHashResolver::<WrappedHugr>::default();

        // Create a base CommitData
        let base_data = CommitData::Base(Hugr::new());

        // Clone the data to create an equivalent copy
        let cloned_data = base_data.clone();

        // Check that original and cloned data are considered equivalent
        let result = resolver.try_merge_mapping(&base_data, &[], &cloned_data, &[]);
        // Verify that the merge succeeds since the data is equivalent
        assert!(result.is_ok());

        // Check that the original and replacement data are considered different
        let repl_data = CommitData::Base(
            Hugr::new_with_entrypoint(FuncDefn::new("dummy", endo_sig(vec![]))).unwrap(),
        );
        let result = resolver.try_merge_mapping(&base_data, &[], &repl_data, &[]);
        assert!(result.is_err());
    }
}
