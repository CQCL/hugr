//! An (example) use of the [super::dataflow](dataflow-analysis framework)
//! to perform constant-folding.

// These are pub because this "example" is used for testing the framework.
pub mod value_handle;
mod context;
pub use context::HugrValueContext;
