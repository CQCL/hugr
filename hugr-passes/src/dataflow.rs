//! Dataflow analysis of Hugrs.

mod datalog;

mod machine;
pub use machine::Machine;

mod partial_value;
pub use partial_value::{AbstractValue, PVEnum, PartialSum, PartialValue, ValueOrSum};

mod total_context;
pub use total_context::TotalContext;

use hugr_core::{Hugr, Node};
use std::hash::Hash;

pub trait DFContext<V>: Clone + Eq + Hash + std::ops::Deref<Target = Hugr> {
    fn interpret_leaf_op(
        &self,
        node: Node,
        ins: &[PartialValue<V>],
    ) -> Option<Vec<PartialValue<V>>>;
}

#[cfg(test)]
mod test;
