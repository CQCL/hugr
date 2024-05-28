//! The Hugr data structure, and its basic component handles.

// Exports everything except the `internal` module.
pub use hugr_core::hugr::{
    hugrmut, rewrite, serialize, validate, views, Hugr, HugrError, NodeMetadata, NodeMetadataMap,
    NodeType, DEFAULT_NODETYPE,
};
