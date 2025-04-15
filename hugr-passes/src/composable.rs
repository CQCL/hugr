//! Compiler passes and utilities for composing them

use std::{error::Error, marker::PhantomData};

use hugr_core::hugr::{hugrmut::HugrMut, ValidationError};
use hugr_core::HugrView;
use itertools::Either;

/// An optimization pass that can be sequenced with another and/or wrapped
/// e.g. by [ValidatingPass]
pub trait ComposablePass: Sized {
    type Error: Error;
    type Result; // Would like to default to () but currently unstable

    fn run(&self, hugr: &mut impl HugrMut) -> Result<Self::Result, Self::Error>;

    fn map_err<E2: Error>(
        self,
        f: impl Fn(Self::Error) -> E2,
    ) -> impl ComposablePass<Result = Self::Result, Error = E2> {
        ErrMapper::new(self, f)
    }

    /// Returns a [ComposablePass] that does "`self` then `other`", so long as
    /// `other::Err` maps into ours.
    fn then<P: ComposablePass, E: ErrorCombiner<Self::Error, P::Error>>(
        self,
        other: P,
    ) -> impl ComposablePass<Result = (Self::Result, P::Result), Error = E> {
        struct Sequence<E, P1, P2>(P1, P2, PhantomData<E>);
        impl<E, P1, P2> ComposablePass for Sequence<E, P1, P2>
        where
            P1: ComposablePass,
            P2: ComposablePass,
            E: ErrorCombiner<P1::Error, P2::Error>,
        {
            type Error = E;

            type Result = (P1::Result, P2::Result);

            fn run(&self, hugr: &mut impl HugrMut) -> Result<Self::Result, Self::Error> {
                let res1 = self.0.run(hugr).map_err(E::from_first)?;
                let res2 = self.1.run(hugr).map_err(E::from_second)?;
                Ok((res1, res2))
            }
        }

        Sequence(self, other, PhantomData)
    }
}

/// Trait for combining the error types from two different passes
/// into a single error.
pub trait ErrorCombiner<A, B>: Error {
    fn from_first(a: A) -> Self;
    fn from_second(b: B) -> Self;
}

impl<A: Error, B: Into<A>> ErrorCombiner<A, B> for A {
    fn from_first(a: A) -> Self {
        a
    }

    fn from_second(b: B) -> Self {
        b.into()
    }
}

impl<A: Error, B: Error> ErrorCombiner<A, B> for Either<A, B> {
    fn from_first(a: A) -> Self {
        Either::Left(a)
    }

    fn from_second(b: B) -> Self {
        Either::Right(b)
    }
}

// Note: in the short term two we could wish for more impls:
//   impl<E:Error> ErrorCombiner<Infallible, E> for E
//   impl<E:Error> ErrorCombiner<E, Infallible> for E
// however, these aren't possible as they conflict with
//   impl<A, B:Into<A>> ErrorCombiner<A,B> for A
// when A=E=Infallible, boo :-(.
// However this will become possible, indeed automatic, when Infallible is replaced
// by ! (never_type) as (unlike Infallible) ! converts Into anything

// ErrMapper ------------------------------
struct ErrMapper<P, E, F>(P, F, PhantomData<E>);

impl<P: ComposablePass, E: Error, F: Fn(P::Error) -> E> ErrMapper<P, E, F> {
    fn new(pass: P, err_fn: F) -> Self {
        Self(pass, err_fn, PhantomData)
    }
}

impl<P: ComposablePass, E: Error, F: Fn(P::Error) -> E> ComposablePass for ErrMapper<P, E, F> {
    type Error = E;
    type Result = P::Result;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<P::Result, Self::Error> {
        self.0.run(hugr).map_err(&self.1)
    }
}

// ValidatingPass ------------------------------

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
    type Error = ValidatePassError<P::Error>;
    type Result = P::Result;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<P::Result, Self::Error> {
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Input {
            err,
            pretty_hugr,
        })?;
        let res = self.0.run(hugr).map_err(ValidatePassError::Underlying)?;
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Output {
            err,
            pretty_hugr,
        })?;
        Ok(res)
    }
}

// IfTrueThen ------------------------------
/// [ComposablePass] that executes a first pass that returns a `bool`
/// result; and then, if-and-only-if that first result was true,
/// executes a second pass
pub struct IfTrueThen<E, A, B>(A, B, PhantomData<E>);
impl<A: ComposablePass<Result = bool>, B: ComposablePass, E: ErrorCombiner<A::Error, B::Error>>
    ComposablePass for IfTrueThen<E, A, B>
{
    type Error = E;

    type Result = Option<B::Result>;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<Self::Result, Self::Error> {
        let res: bool = self.0.run(hugr).map_err(ErrorCombiner::from_first)?;
        res.then(|| self.1.run(hugr).map_err(ErrorCombiner::from_second))
            .transpose()
    }
}

pub(crate) fn validate_if_test<P: ComposablePass>(
    pass: P,
    hugr: &mut impl HugrMut,
) -> Result<P::Result, ValidatePassError<P::Error>> {
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
    fn test_then() {
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
        let r: Result<_, Either<Infallible, ConstFoldError>> =
            dce.clone().then(cfold.clone()).run(&mut hugr.clone());
        assert_eq!(r, Err(Either::Right(exp_err.clone())));

        let r = dce
            .clone()
            .map_err(|inf| match inf {})
            .then(cfold.clone())
            .run(&mut hugr.clone());
        assert_eq!(r, Err(exp_err));

        let r2: Result<_, Either<_, _>> = cfold.then(dce).run(&mut hugr.clone());
        r2.unwrap();
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
