//! The Hugr data structure, and its basic component handles.

// Exports everything except the `internal` module.
pub use hugr_core::hugr::{
    hugrmut, patch, serialize, validate, views, Hugr, HugrError, HugrView, IdentList,
    InvalidIdentifier, NodeMetadata, NodeMetadataMap, OpType, Patch, SimpleReplacement,
    SimpleReplacementError, ValidationError, DEFAULT_OPTYPE,
};
