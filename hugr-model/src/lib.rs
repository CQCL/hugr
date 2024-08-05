//! This crate describes the hugr data format.
//!
//! All of the types in this crate are plain data.
//! Tools that process this data further should convert these types
//! into data structures that allow more efficient access,
//! and validate that invariants are met.
//!
//! We intend to support serialization from and into the following file formats:
//!
//! - S-expressions via `parens` as a human readable and writable format.
pub mod v2_syntax;
