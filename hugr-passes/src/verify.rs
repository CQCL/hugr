//! Provides [VerifyLevel] with tools to run passes with configurable
//! verification.

use thiserror::Error;

use hugr_core::{
    extension::ExtensionRegistry,
    hugr::{hugrmut::HugrMut, ValidationError},
    HugrView,
};

#[derive(Debug, Clone, Copy, Ord, Eq, PartialOrd, PartialEq)]
/// A type for running [HugrMut] algorithms with verification.
///
/// Provides [VerifyLevel::run_verified_pass] to invoke a closure with pre and post
/// verification.
///
/// The default level is `None` because verification can be expensive.
pub enum VerifyLevel {
    /// Do no verification.
    None,
    /// Verify using [HugrView::validate_no_extensions]. This is useful when you
    /// do not expect valid Extension annotations on Nodes.
    WithoutExtensions,
    /// Verify using [HugrView::validate].
    WithExtensions,
}

#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum VerifyError {
    #[error("Failed to verify input HUGR: {err}\n{pretty_hugr}")]
    InputError {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
    #[error("Failed to verify output HUGR: {err}\n{pretty_hugr}")]
    OutputError {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
}

impl Default for VerifyLevel {
    fn default() -> Self {
        if cfg!(test) {
            // Many tests fail when run with Self::WithExtensions
            Self::WithoutExtensions
        } else {
            Self::None
        }
    }
}

impl VerifyLevel {
    /// Run an operation on a [HugrMut]. `hugr` will be verified according to
    /// [self](VerifyLevel), then `pass` will be invoked. If `pass` succeeds
    /// then `hugr` will be verified again.
    pub fn run_verified_pass<H: HugrMut, E, T>(
        &self,
        hugr: &mut H,
        reg: &ExtensionRegistry,
        pass: impl FnOnce(&mut H, &Self) -> Result<T, E>,
    ) -> Result<T, E>
    where
        VerifyError: Into<E>,
    {
        self.verify_impl(hugr, reg, |err, pretty_hugr| VerifyError::InputError {
            err,
            pretty_hugr,
        })?;
        let result = pass(hugr, self)?;
        self.verify_impl(hugr, reg, |err, pretty_hugr| VerifyError::OutputError {
            err,
            pretty_hugr,
        })?;
        Ok(result)
    }

    fn verify_impl<E>(
        &self,
        hugr: &impl HugrView,
        reg: &ExtensionRegistry,
        mk_err: impl FnOnce(ValidationError, String) -> VerifyError,
    ) -> Result<(), E>
    where
        VerifyError: Into<E>,
    {
        match self {
            VerifyLevel::None => Ok(()),
            VerifyLevel::WithoutExtensions => hugr.validate_no_extensions(reg),
            VerifyLevel::WithExtensions => hugr.validate(reg),
        }
        .map_err(|err| mk_err(err, hugr.mermaid_string()).into())
    }
}
