//! Binary format based on capnproto.
//!
//! The binary format is optimised for fast serialization and deserialization of
//! hugr modules in the [table] representation. It is the preferred format to
//! communicate hugr graphs between machines. When a hugr module is to be
//! written or read by humans, the [text] format can be used instead.
//!
//! [table]: crate::v0::table
//! [text]: crate::v0::ast
mod read;
mod write;

pub use read::{ReadError, read_from_reader, read_from_slice};
pub use write::{WriteError, write_to_vec, write_to_writer};
