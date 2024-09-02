//! Dataflow analysis of Hugrs.

mod datalog;
pub use datalog::Machine;

mod partial_value;
pub use partial_value::{PartialValue, AbstractValue};

mod value_row;
pub use value_row::ValueRow;

mod total_context;
pub use total_context::TotalContext;
