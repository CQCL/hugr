//! The HUGR binary representation.
mod read;
mod write;

pub use read::{read_module_list_from_reader, read_from_reader, read_from_slice, ReadError};
pub use write::{write_module_list_to_writer, write_to_vec, write_to_writer, WriteError};
