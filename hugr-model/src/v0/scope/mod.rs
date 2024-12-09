//! Utilities for working with scoped symbols, variables and links.
mod link;
mod symbol;

pub use link::LinkTable;
pub use symbol::{SymbolIntroError, SymbolResolveError, SymbolTable};
