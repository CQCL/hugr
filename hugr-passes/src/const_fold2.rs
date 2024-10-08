#![warn(missing_docs)]
//! An (example) use of the [super::dataflow](dataflow-analysis framework)
//! to perform constant-folding.

// These are pub because this "example" is used for testing the framework.
mod context;
pub mod value_handle;
pub use context::HugrValueContext;
