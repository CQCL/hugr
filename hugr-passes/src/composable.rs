//! Compiler passes and utilities for composing them

use std::{error::Error, marker::PhantomData};

use hugr_core::hugr::{hugrmut::HugrMut, ValidationError};
use hugr_core::HugrView;
use itertools::Either;

pub trait ComposablePass: Sized {
    type Err: Error;
    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::Err>;
    fn map_err<E2: Error>(self, f: impl Fn(Self::Err) -> E2) -> impl ComposablePass<Err = E2> {
        ErrMapper::new(self, f)
    }
    fn sequence(
        self,
        other: impl ComposablePass<Err = Self::Err>,
    ) -> impl ComposablePass<Err = Self::Err> {
        (self, other) // SequencePass::new(self, other) ?
    }
    fn sequence_either<P: ComposablePass>(
        self,
        other: P,
    ) -> impl ComposablePass<Err = Either<Self::Err, P::Err>> {
        self.map_err(Either::Left)
            .sequence(other.map_err(Either::Right))
    }
}

struct ErrMapper<P, E, F>(P, F, PhantomData<E>);

impl<P: ComposablePass, E: Error, F: Fn(P::Err) -> E> ErrMapper<P, E, F> {
    fn new(pass: P, err_fn: F) -> Self {
        Self(pass, err_fn, PhantomData)
    }
}

impl<P: ComposablePass, E: Error, F: Fn(P::Err) -> E> ComposablePass for ErrMapper<P, E, F> {
    type Err = E;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::Err> {
        self.0.run(hugr).map_err(&self.1)
    }
}

impl<E: Error, P1: ComposablePass<Err = E>, P2: ComposablePass<Err = E>> ComposablePass
    for (P1, P2)
{
    type Err = E;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::Err> {
        self.0.run(hugr)?;
        self.1.run(hugr)
    }
}

#[derive(thiserror::Error, Debug)]
#[allow(missing_docs)]
pub enum ValidatePassError<E> {
    #[error("Failed to validate input HUGR: {err}\n{pretty_hugr}")]
    Input {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
    #[error("Failed to validate output HUGR: {err}\n{pretty_hugr}")]
    Output {
        #[source]
        err: ValidationError,
        pretty_hugr: String,
    },
    #[error(transparent)]
    Underlying(E),
}

/// Runs another, underlying, pass, with validation of the Hugr
/// both before and afterwards.
pub struct ValidatingPass<P>(P, bool);

impl<P: ComposablePass> ValidatingPass<P> {
    pub fn new_default(underlying: P) -> Self {
        // Self(underlying, cfg!(feature = "extension_inference"))
        // Sadly, many tests fail with extension inference, hence:
        Self(underlying, false)
    }

    pub fn new_validating_extensions(underlying: P) -> Self {
        Self(underlying, true)
    }

    pub fn new(underlying: P, validate_extensions: bool) -> Self {
        Self(underlying, validate_extensions)
    }

    fn validation_impl<E>(
        &self,
        hugr: &impl HugrView,
        mk_err: impl FnOnce(ValidationError, String) -> ValidatePassError<E>,
    ) -> Result<(), ValidatePassError<E>> {
        match self.1 {
            false => hugr.validate_no_extensions(),
            true => hugr.validate(),
        }
        .map_err(|err| mk_err(err, hugr.mermaid_string()))
    }
}

impl<P: ComposablePass> ComposablePass for ValidatingPass<P> {
    type Err = ValidatePassError<P::Err>;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::Err> {
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Input {
            err,
            pretty_hugr,
        })?;
        self.0.run(hugr).map_err(ValidatePassError::Underlying)?;
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Output {
            err,
            pretty_hugr,
        })
    }
}

pub fn validate_if_test<P: ComposablePass>(
    pass: P,
    hugr: &mut impl HugrMut,
) -> Result<(), ValidatePassError<P::Err>> {
    if cfg!(test) {
        ValidatingPass::new_default(pass).run(hugr)
    } else {
        pass.run(hugr).map_err(ValidatePassError::Underlying)
    }
}
