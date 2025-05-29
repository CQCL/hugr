use relrc::EquivalenceResolver;

/// A resolver that considers two nodes equivalent if they are the same pointer.
///
/// Resolvers determine when two patches are equivalent and should be merged
/// in the patch history.
///
/// This is a trivial resolver (to be expanded on later), that considers two
/// patches equivalent if they point to the same data in memory.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
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
