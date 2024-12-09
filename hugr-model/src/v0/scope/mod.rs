//! Utilities for working with scoped symbols, variables and links.
mod link;
mod symbol;
mod vars;

pub use link::LinkTable;
pub use symbol::{DuplicateSymbolError, SymbolTable, UnknownSymbolError};
pub use vars::{DuplicateVarError, UnknownVarError, VarTable};
