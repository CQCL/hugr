//! The Hugr data structure, and its basic component handles.

// Exports everything except the `internal` module.
pub use hugr_core::hugr::{
    DEFAULT_OPTYPE, Hugr, HugrError, HugrView, IdentList, InvalidIdentifier, NodeMetadata,
    NodeMetadataMap, OpType, Patch, SimpleReplacement, SimpleReplacementError, ValidationError,
    hugrmut, patch, serialize, validate, views,
};
