//! The HUGR binary representation.
mod read;
mod write;

pub use read::{read_from_slice, ReadError};
pub use write::write_to_vec;
