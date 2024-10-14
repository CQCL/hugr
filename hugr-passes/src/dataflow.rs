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
    /// Given lattice values for each input, update lattice values for the (dataflow) outputs.
    /// `_outs` is an array with one element per dataflow output, each initialized to [PartialValue::Top]
    /// which is the correct value to leave if nothing can be deduced about that output.
    /// (The default does nothing, i.e. leaves `Top` for all outputs.)
    fn interpret_leaf_op(
        &self,
        _node: Node,
        _ins: &[PartialValue<V>],
        _outs: &mut [PartialValue<V>],
    ) {
    }
}

#[cfg(test)]
mod test;
