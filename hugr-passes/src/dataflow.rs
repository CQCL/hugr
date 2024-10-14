#![warn(missing_docs)]
//! Dataflow analysis of Hugrs.

mod datalog;

mod machine;
pub use machine::{Machine, TailLoopTermination};

mod partial_value;
pub use partial_value::{AbstractValue, PartialSum, PartialValue, Sum};

use hugr_core::{Hugr, Node};
use std::hash::Hash;

/// Clients of the dataflow framework (particular analyses, such as constant folding)
/// must implement this trait (including providing an appropriate domain type `V`).
pub trait DFContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    /// Given lattice values for each input, produce lattice values for (what we know of)
    /// the outputs. Returning `None` indicates nothing can be deduced.
    fn interpret_leaf_op(
        &self,
        node: Node,
        ins: &[PartialValue<V>],
    ) -> Option<Vec<PartialValue<V>>>;
}

#[cfg(test)]
mod test;
