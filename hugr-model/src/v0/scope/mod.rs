//! Utilities for working with scoped symbols, variables and links.
mod link;
mod symbol;
mod vars;

pub use link::LinkTable;
pub use symbol::{SymbolIntroError, SymbolResolveError, SymbolTable};
pub use vars::{DuplicateVarError, UnknownVarError, VarTable};
