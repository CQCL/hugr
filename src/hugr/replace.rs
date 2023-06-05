//! Rewrite operations on the HUGR.

#[allow(clippy::module_inception)] // TODO: Rename?
pub mod rewrite;
use crate::Hugr;
pub use rewrite::{OpenHugr, Rewrite, RewriteError};

/// An operation that can be applied to mutate a Hugr
pub enum RewriteOp {
    /// Replace a set of dataflow sibling nodes with new ones
    Rewrite(Rewrite),
    /// Move a set of controlflow sibling nodes into a nested CFG
    OutlineCFG,
}

impl RewriteOp {
    /// Applies a ReplacementOp to the graph.
    pub fn apply(self, h: &mut Hugr) -> Result<(), RewriteError> {
        match self {
            Self::Rewrite(rewrite) => {
                // Get the open graph for the rewrites, and a HUGR with the additional components.
                let (rewrite, mut replacement, parents) = rewrite.into_parts();

                // TODO: Use `parents` to update the hierarchy, and keep the internal hierarchy from `replacement`.
                let _ = parents;

                let node_inserted = |old, new| {
                    std::mem::swap(&mut h.op_types[new], &mut replacement.op_types[old]);
                    // TODO: metadata (Fn parameter ?)
                };
                rewrite.apply_with_callbacks(
                    &mut h.graph,
                    |_| {},
                    |_| {},
                    node_inserted,
                    |_, _| {},
                    |_, _| {},
                )?;

                // TODO: Check types
            }
            Self::OutlineCFG => {
                todo!();
            }
        };
        Ok(())
    }
}
