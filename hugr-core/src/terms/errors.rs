use hugr_model::v0::{SymbolName, VarName};
use std::error::Error;
use thiserror::Error;

use super::Term;

/// Constructor is applied to wrong number of arguments.
#[derive(Debug, Error)]
#[error(
    "`{constructor}` expects `{expected}` arguments but got `{actual}` in term:\n```\n{term}\n```"
)]
pub struct ArityError {
    /// The number of arguments that the constructor expects.
    pub expected: usize,
    /// The number of arguments that were passed to the constructor.
    pub actual: usize,
    /// The term that caused the error.
    pub term: Term,
    /// The constructor that is applied to the wrong number of arguments.
    pub constructor: SymbolName,
}

/// There is an error within a field of a constructor.
#[derive(Debug, Error)]
#[error(
    "invalid field `{name}` (index {index}) of constructor `{constructor}` in term:\n```\n{term}\n```"
)]
pub struct FieldError {
    /// The original error in the field.
    #[source]
    pub error: AnyError,
    /// The index of the field within the constructor's parameter list.
    pub index: usize,
    /// The name of the field.
    pub name: VarName,
    /// The name of the constructor.
    pub constructor: SymbolName,
    /// The constructor application that caused the error.
    pub term: Term,
}

/// A particular term constructor was expected.
#[derive(Debug, Error)]
#[error("expected constructor `{constructor}` but got term:\n```\n{term}\n```")]
pub struct ConstructorError {
    /// The constructor that was expected.
    pub constructor: SymbolName,
    /// The term that caused the error.
    pub term: Term,
}

type AnyError = Box<dyn Error + Send + Sync>;
