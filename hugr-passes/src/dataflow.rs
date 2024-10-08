#![warn(missing_docs)]
//! Dataflow analysis of Hugrs.

mod datalog;
pub use datalog::{AbstractValue, DFContext};

mod machine;
pub use machine::{Machine, TailLoopTermination};

mod partial_value;
pub use partial_value::{BaseValue, PVEnum, PartialSum, PartialValue, Sum};

mod total_context;
pub use total_context::TotalContext;

#[cfg(test)]
mod test;
