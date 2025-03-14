//! Compiler passes and utilities for composing them

use std::{error::Error, marker::PhantomData};

use hugr_core::hugr::{hugrmut::HugrMut, ValidationError};
use hugr_core::HugrView;
use itertools::Either;

/// An optimization pass that can be sequenced with another and/or wrapped
/// e.g. by [ValidatingPass]
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

/// Error from a [ValidatingPass]
#[derive(thiserror::Error, Debug)]
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
    Underlying(#[from] E),
}

/// Runs an underlying pass, but with validation of the Hugr
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

pub(crate) fn validate_if_test<P: ComposablePass>(
    pass: P,
    hugr: &mut impl HugrMut,
) -> Result<(), ValidatePassError<P::Err>> {
    if cfg!(test) {
        ValidatingPass::new_default(pass).run(hugr)
    } else {
        pass.run(hugr).map_err(ValidatePassError::Underlying)
    }
}

#[cfg(test)]
mod test {
    use std::convert::Infallible;

    use hugr_core::builder::{
        Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use hugr_core::extension::prelude::{bool_t, usize_t, ConstUsize};
    use hugr_core::hugr::hugrmut::HugrMut;
    use hugr_core::ops::{handle::NodeHandle, Input, Output, DEFAULT_OPTYPE, DFG};
    use hugr_core::{types::Signature, Hugr, HugrView, IncomingPort};
    use itertools::Either;

    use crate::composable::{ValidatePassError, ValidatingPass};
    use crate::const_fold::{ConstFoldError, ConstantFoldPass};
    use crate::DeadCodeElimPass;

    use super::ComposablePass;

    #[test]
    fn test_sequence() {
        let mut mb = ModuleBuilder::new();
        let id1 = mb
            .define_function("id1", Signature::new_endo(usize_t()))
            .unwrap();
        let inps = id1.input_wires();
        let id1 = id1.finish_with_outputs(inps).unwrap();
        let id2 = mb
            .define_function("id2", Signature::new_endo(usize_t()))
            .unwrap();
        let inps = id2.input_wires();
        let id2 = id2.finish_with_outputs(inps).unwrap();
        let hugr = mb.finish_hugr().unwrap();

        let dce = DeadCodeElimPass::default().with_entry_points([id1.node()]);
        let cfold =
            ConstantFoldPass::default().with_inputs(id2.node(), [(0, ConstUsize::new(2).into())]);

        cfold.run(&mut hugr.clone()).unwrap();

        let exp_err = ConstFoldError::InvalidEntryPoint(id2.node(), DEFAULT_OPTYPE);
        let r: Result<(), Either<Infallible, ConstFoldError>> = dce
            .clone()
            .sequence_either(cfold.clone())
            .run(&mut hugr.clone());
        assert_eq!(r, Err(Either::Right(exp_err.clone())));

        let r: Result<(), ConstFoldError> = dce
            .map_err(|inf| match inf {})
            .sequence(cfold)
            .run(&mut hugr.clone());
        assert_eq!(r, Err(exp_err));
    }

    #[test]
    fn test_validation() {
        let mut h = Hugr::new(DFG {
            signature: Signature::new(usize_t(), bool_t()),
        });
        let inp = h.add_node_with_parent(
            h.root(),
            Input {
                types: usize_t().into(),
            },
        );
        let outp = h.add_node_with_parent(
            h.root(),
            Output {
                types: bool_t().into(),
            },
        );
        h.connect(inp, 0, outp, 0);
        let backup = h.clone();
        let err = backup.validate().unwrap_err();

        let no_inputs: [(IncomingPort, _); 0] = [];
        let cfold = ConstantFoldPass::default().with_inputs(backup.root(), no_inputs);
        cfold.run(&mut h).unwrap();
        assert_eq!(h, backup); // Did nothing

        let r = ValidatingPass(cfold, false).run(&mut h);
        assert!(matches!(r, Err(ValidatePassError::Input { err: e, .. }) if e == err));
    }
}
