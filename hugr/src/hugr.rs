//! The Hugr data structure, and its basic component handles.

// Exports everything except the `internal` module.
pub use hugr_core::hugr::{
    hugrmut, patch, serialize, validate, views, Patch, Hugr, HugrError, HugrView, IdentList,
    InvalidIdentifier, LoadHugrError, NodeMetadata, NodeMetadataMap, OpType, RootTagged,
    SimpleReplacement, SimpleReplacementError, ValidationError, DEFAULT_OPTYPE,
};
