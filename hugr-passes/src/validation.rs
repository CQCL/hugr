//! Provides [ValidationLevel] with tools to run passes with configurable
//! validation.

use thiserror::Error;

use hugr_core::hugr::{hugrmut::HugrMut, ValidationError};
use hugr_core::HugrView;

#[derive(Debug, Clone, Copy, Ord, Eq, PartialOrd, PartialEq)]
/// A type for running [HugrMut] algorithms with verification.
///
/// Provides [ValidationLevel::run_validated_pass] to invoke a closure with pre and post
/// validation.
///
/// The default level is `None` because validation can be expensive.
pub enum ValidationLevel {
    /// Do no verification.
    None,
    /// Validate using [HugrView::validate].
    Validate,
}

#[derive(Error, Debug, PartialEq)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum ValidatePassError {
    #[error("Failed to validate input HUGR: {err}\n{pretty_hugr}")]
    InputError {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
    #[error("Failed to validate output HUGR: {err}\n{pretty_hugr}")]
    OutputError {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
}

impl Default for ValidationLevel {
    fn default() -> Self {
        if cfg!(test) {
            Self::Validate
        } else {
            Self::None
        }
    }
}

impl ValidationLevel {
    /// Run an operation on a [HugrMut]. `hugr` will be verified according to
    /// [self](ValidationLevel), then `pass` will be invoked. If `pass` succeeds
    /// then `hugr` will be verified again.
    pub fn run_validated_pass<H: HugrMut, E, T>(
        &self,
        hugr: &mut H,
        pass: impl FnOnce(&mut H, &Self) -> Result<T, E>,
    ) -> Result<T, E>
    where
        ValidatePassError: Into<E>,
    {
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::InputError {
            err,
            pretty_hugr,
        })?;
        let result = pass(hugr, self)?;
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::OutputError {
            err,
            pretty_hugr,
        })?;
        Ok(result)
    }

    fn validation_impl<E>(
        &self,
        hugr: &impl HugrView,
        mk_err: impl FnOnce(ValidationError, String) -> ValidatePassError,
    ) -> Result<(), E>
    where
        ValidatePassError: Into<E>,
    {
        match self {
            ValidationLevel::None => Ok(()),
            ValidationLevel::Validate => hugr.validate(),
        }
        .map_err(|err| mk_err(err, hugr.mermaid_string()).into())
    }
}
