//! Compiler passes and utilities for composing them

use std::{error::Error, marker::PhantomData};

use hugr_core::hugr::{hugrmut::HugrMut, ValidationError};
use hugr_core::{HugrView, Node};
use itertools::Either;

pub trait ComposablePass: Sized {
    type E: Error;
    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::E>;
    fn add_entry_point(&mut self, func_node: Node);
    fn map_err<E2: Error>(self, f: impl Fn(Self::E) -> E2) -> impl ComposablePass<E = E2> {
        ErrMapper::new(self, f)
    }
    fn sequence(self, other: impl ComposablePass<E = Self::E>) -> impl ComposablePass<E = Self::E> {
        (self, other) // SequencePass::new(self, other) ?
    }
    fn sequence_either<P: ComposablePass>(
        self,
        other: P,
    ) -> impl ComposablePass<E = Either<Self::E, P::E>> {
        self.map_err(Either::Left)
            .sequence(other.map_err(Either::Right))
    }
}

struct ErrMapper<P, E, F>(P, F, PhantomData<E>);

impl<P: ComposablePass, E: Error, F: Fn(P::E) -> E> ErrMapper<P, E, F> {
    fn new(pass: P, err_fn: F) -> Self {
        Self(pass, err_fn, PhantomData)
    }
}

impl<P: ComposablePass, Err: Error, F: Fn(P::E) -> Err> ComposablePass for ErrMapper<P, Err, F> {
    type E = Err;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::E> {
        self.0.run(hugr).map_err(&self.1)
    }

    fn add_entry_point(&mut self, func_node: Node) {
        self.0.add_entry_point(func_node)
    }
}

impl<Err: Error, P1: ComposablePass<E = Err>, P2: ComposablePass<E = Err>> ComposablePass
    for (P1, P2)
{
    type E = Err;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::E> {
        self.0.run(hugr)?;
        self.1.run(hugr)
    }

    fn add_entry_point(&mut self, func_node: Node) {
        self.0.add_entry_point(func_node);
        self.1.add_entry_point(func_node);
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

pub struct ValidatingPass<P>(P, bool);

impl<P: ComposablePass> ValidatingPass<P> {
    pub fn new_default(underlying: P) -> Self {
        Self(underlying, false)
    }

    pub fn new_validating_extensions(underlying: P) -> Self {
        Self(underlying, true)
    }

    pub fn new(underlying: P, extensions: bool) -> Self {
        Self(underlying, extensions)
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
    type E = ValidatePassError<P::E>;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<(), Self::E> {
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

    fn add_entry_point(&mut self, func_node: Node) {
        self.0.add_entry_point(func_node);
    }
}
