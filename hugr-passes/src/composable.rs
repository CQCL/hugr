//! Compiler passes and utilities for composing them

use std::{error::Error, marker::PhantomData};

use hugr_core::core::HugrNode;
use hugr_core::hugr::{ValidationError, hugrmut::HugrMut};
use itertools::Either;

/// An optimization pass that can be sequenced with another and/or wrapped
/// e.g. by [`ValidatingPass`]
pub trait ComposablePass<H: HugrMut>: Sized {
    /// Error thrown by this pass.
    type Error: Error;
    /// Result returned by this pass.
    type Result; // Would like to default to () but currently unstable

    /// Run the pass on the given HUGR.
    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error>;

    /// Apply a function to the error type of this pass, returning a new
    /// [`ComposablePass`] that has the same result type.
    fn map_err<E2: Error>(
        self,
        f: impl Fn(Self::Error) -> E2,
    ) -> impl ComposablePass<H, Result = Self::Result, Error = E2> {
        ErrMapper::new(self, f)
    }

    /// Returns a [`ComposablePass`] that does "`self` then `other`", so long as
    /// `other::Err` can be combined with ours.
    fn then<P: ComposablePass<H>, E: ErrorCombiner<Self::Error, P::Error>>(
        self,
        other: P,
    ) -> impl ComposablePass<H, Result = (Self::Result, P::Result), Error = E> {
        struct Sequence<E, P1, P2>(P1, P2, PhantomData<E>);
        impl<H, E, P1, P2> ComposablePass<H> for Sequence<E, P1, P2>
        where
            H: HugrMut,
            P1: ComposablePass<H>,
            P2: ComposablePass<H>,
            E: ErrorCombiner<P1::Error, P2::Error>,
        {
            type Error = E;
            type Result = (P1::Result, P2::Result);

            fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
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
    /// Create a combined error from the first pass's error.
    fn from_first(a: A) -> Self;
    /// Create a combined error from the second pass's error.
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

// Note: in the short term we could wish for two more impls:
//   impl<E:Error> ErrorCombiner<Infallible, E> for E
//   impl<E:Error> ErrorCombiner<E, Infallible> for E
// however, these aren't possible as they conflict with
//   impl<A, B:Into<A>> ErrorCombiner<A,B> for A
// when A=E=Infallible, boo :-(.
// However this will become possible, indeed automatic, when Infallible is replaced
// by ! (never_type) as (unlike Infallible) ! converts Into anything

// ErrMapper ------------------------------
struct ErrMapper<P, H, E, F>(P, F, PhantomData<(E, H)>);

impl<H: HugrMut, P: ComposablePass<H>, E: Error, F: Fn(P::Error) -> E> ErrMapper<P, H, E, F> {
    fn new(pass: P, err_fn: F) -> Self {
        Self(pass, err_fn, PhantomData)
    }
}

impl<P: ComposablePass<H>, H: HugrMut, E: Error, F: Fn(P::Error) -> E> ComposablePass<H>
    for ErrMapper<P, H, E, F>
{
    type Error = E;
    type Result = P::Result;

    fn run(&self, hugr: &mut H) -> Result<P::Result, Self::Error> {
        self.0.run(hugr).map_err(&self.1)
    }
}

// ValidatingPass ------------------------------

/// Error from a [`ValidatingPass`]
#[derive(thiserror::Error, Debug)]
pub enum ValidatePassError<N, E>
where
    N: HugrNode + 'static,
{
    /// Validation failed on the initial HUGR.
    #[error("Failed to validate input HUGR: {err}\n{pretty_hugr}")]
    Input {
        /// The validation error that occurred.
        #[source]
        err: Box<ValidationError<N>>,
        /// A pretty-printed representation of the HUGR that failed validation.
        pretty_hugr: String,
    },
    /// Validation failed on the final HUGR.
    #[error("Failed to validate output HUGR: {err}\n{pretty_hugr}")]
    Output {
        /// The validation error that occurred.
        #[source]
        err: Box<ValidationError<N>>,
        /// A pretty-printed representation of the HUGR that failed validation.
        pretty_hugr: String,
    },
    /// An error from the underlying pass.
    #[error(transparent)]
    Underlying(Box<E>),
}

impl<N: HugrNode, E> From<E> for ValidatePassError<N, E> {
    fn from(err: E) -> Self {
        Self::Underlying(Box::new(err))
    }
}

/// Runs an underlying pass, but with validation of the Hugr
/// both before and afterwards.
pub struct ValidatingPass<P, H>(P, PhantomData<H>);

impl<P: ComposablePass<H>, H: HugrMut> ValidatingPass<P, H> {
    /// Return a new [`ValidatingPass`] that wraps the given underlying pass.
    pub fn new(underlying: P) -> Self {
        Self(underlying, PhantomData)
    }

    fn validation_impl<E>(
        &self,
        hugr: &H,
        mk_err: impl FnOnce(ValidationError<H::Node>, String) -> ValidatePassError<H::Node, E>,
    ) -> Result<(), ValidatePassError<H::Node, E>> {
        hugr.validate()
            .map_err(|err| mk_err(err, hugr.mermaid_string()))
    }
}

impl<P: ComposablePass<H>, H: HugrMut> ComposablePass<H> for ValidatingPass<P, H>
where
    H::Node: 'static,
{
    type Error = ValidatePassError<H::Node, P::Error>;
    type Result = P::Result;

    fn run(&self, hugr: &mut H) -> Result<P::Result, Self::Error> {
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Input {
            err: Box::new(err),
            pretty_hugr,
        })?;
        let res = self.0.run(hugr)?;
        self.validation_impl(hugr, |err, pretty_hugr| ValidatePassError::Output {
            err: Box::new(err),
            pretty_hugr,
        })?;
        Ok(res)
    }
}

// IfThen ------------------------------
/// [`ComposablePass`] that executes a first pass that returns a `bool`
/// result; and then, if-and-only-if that first result was true,
/// executes a second pass
pub struct IfThen<E, H, A, B>(A, B, PhantomData<(E, H)>);

impl<
    A: ComposablePass<H, Result = bool>,
    B: ComposablePass<H>,
    H: HugrMut,
    E: ErrorCombiner<A::Error, B::Error>,
> IfThen<E, H, A, B>
{
    /// Make a new instance given the [`ComposablePass`] to run first
    /// and (maybe) second
    pub fn new(fst: A, opt_snd: B) -> Self {
        Self(fst, opt_snd, PhantomData)
    }
}

impl<
    A: ComposablePass<H, Result = bool>,
    B: ComposablePass<H>,
    H: HugrMut,
    E: ErrorCombiner<A::Error, B::Error>,
> ComposablePass<H> for IfThen<E, H, A, B>
{
    type Error = E;
    type Result = Option<B::Result>;

    fn run(&self, hugr: &mut H) -> Result<Self::Result, Self::Error> {
        let res: bool = self.0.run(hugr).map_err(ErrorCombiner::from_first)?;
        res.then(|| self.1.run(hugr).map_err(ErrorCombiner::from_second))
            .transpose()
    }
}

pub(crate) fn validate_if_test<P: ComposablePass<H>, H: HugrMut>(
    pass: P,
    hugr: &mut H,
) -> Result<P::Result, ValidatePassError<H::Node, P::Error>> {
    if cfg!(test) {
        ValidatingPass::new(pass).run(hugr)
    } else {
        Ok(pass.run(hugr)?)
    }
}

#[cfg(test)]
mod test {
    use hugr_core::ops::Value;
    use itertools::{Either, Itertools};

    use hugr_core::builder::{
        Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder, ModuleBuilder,
    };
    use hugr_core::extension::prelude::{ConstUsize, MakeTuple, UnpackTuple, bool_t, usize_t};
    use hugr_core::hugr::hugrmut::HugrMut;
    use hugr_core::ops::{DFG, Input, OpType, Output, handle::NodeHandle};
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::types::{Signature, TypeRow};
    use hugr_core::{Hugr, HugrView, IncomingPort, Node, NodeIndex};

    use crate::const_fold::{ConstFoldError, ConstantFoldPass};
    use crate::dead_code::DeadCodeElimError;
    use crate::untuple::{UntupleRecursive, UntupleResult};
    use crate::{DeadCodeElimPass, ReplaceTypes, UntuplePass};

    use super::{ComposablePass, IfThen, ValidatePassError, ValidatingPass, validate_if_test};

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

        let c_usz = Value::from(ConstUsize::new(2));
        let not_a_node = Node::from(portgraph::NodeIndex::new(
            hugr.nodes().map(Node::index).max().unwrap() + 1,
        ));
        assert!(!hugr.contains_node(not_a_node));
        let dce = DeadCodeElimPass::default().with_entry_points([not_a_node]);
        let cfold = ConstantFoldPass::default().with_inputs(id2.node(), [(0, c_usz.clone())]);

        cfold.run(&mut hugr.clone()).unwrap();

        let dce_err = DeadCodeElimError::NodeNotFound(not_a_node);
        let r: Result<_, Either<DeadCodeElimError, ConstFoldError>> =
            dce.clone().then(cfold.clone()).run(&mut hugr.clone());
        assert_eq!(r, Err(Either::Left(dce_err.clone())));

        let r: Result<_, Either<_, _>> = cfold
            .clone()
            .with_inputs(id1.node(), [(0, c_usz)])
            .then(dce.clone())
            .run(&mut hugr.clone());
        assert_eq!(r, Err(Either::Right(dce_err)));

        // Avoid wrapping in Either by mapping both to same Error
        let r = dce
            .map_err(|e| match e {
                DeadCodeElimError::NodeNotFound(node) => ConstFoldError::MissingEntryPoint { node },
            })
            .then(cfold.clone())
            .run(&mut hugr.clone());
        assert_eq!(
            r,
            Err(ConstFoldError::MissingEntryPoint { node: not_a_node })
        );

        // Or where second supports Into first
        let v = ValidatingPass::new(cfold.clone());
        let r: Result<_, ValidatePassError<Node, ConstFoldError>> =
            v.then(cfold).run(&mut hugr.clone());
        r.unwrap();
    }

    #[test]
    fn test_validation() {
        let mut h = Hugr::new_with_entrypoint(DFG {
            signature: Signature::new(usize_t(), bool_t()),
        })
        .unwrap();
        let inp = h.add_node_with_parent(
            h.entrypoint(),
            Input {
                types: usize_t().into(),
            },
        );
        let outp = h.add_node_with_parent(
            h.entrypoint(),
            Output {
                types: bool_t().into(),
            },
        );
        h.connect(inp, 0, outp, 0);
        let backup = h.clone();
        let err = backup.validate().unwrap_err();

        let no_inputs: [(IncomingPort, _); 0] = [];
        let cfold = ConstantFoldPass::default().with_inputs(backup.entrypoint(), no_inputs);
        cfold.run(&mut h).unwrap();
        assert_eq!(h, backup); // Did nothing

        let r = ValidatingPass::new(cfold).run(&mut h);
        assert!(matches!(r, Err(ValidatePassError::Input { err: e, .. }) if *e == err));
    }

    #[test]
    fn test_if_then() {
        let tr = TypeRow::from(vec![usize_t(); 2]);

        let h = {
            let sig = Signature::new_endo(tr.clone());
            let mut fb = FunctionBuilder::new("tupuntup", sig).unwrap();
            let [a, b] = fb.input_wires_arr();
            let tup = fb
                .add_dataflow_op(MakeTuple::new(tr.clone()), [a, b])
                .unwrap();
            let untup = fb
                .add_dataflow_op(UnpackTuple::new(tr.clone()), tup.outputs())
                .unwrap();
            fb.finish_hugr_with_outputs(untup.outputs()).unwrap()
        };

        let untup = UntuplePass::new(UntupleRecursive::Recursive);
        {
            // Change usize_t to INT_TYPES[6], and if that did anything (it will!), then Untuple
            let mut repl = ReplaceTypes::default();
            let usize_custom_t = usize_t().as_extension().unwrap().clone();
            repl.replace_type(usize_custom_t, INT_TYPES[6].clone());
            let ifthen = IfThen::<Either<_, _>, _, _, _>::new(repl, untup.clone());

            let mut h = h.clone();
            let r = validate_if_test(ifthen, &mut h).unwrap();
            assert_eq!(
                r,
                Some(UntupleResult {
                    rewrites_applied: 1
                })
            );
            let [tuple_in, tuple_out] = h.children(h.entrypoint()).collect_array().unwrap();
            assert_eq!(h.output_neighbours(tuple_in).collect_vec(), [tuple_out; 2]);
        }

        // Change INT_TYPES[5] to INT_TYPES[6]; that won't do anything, so don't Untuple
        let mut repl = ReplaceTypes::default();
        let i32_custom_t = INT_TYPES[5].as_extension().unwrap().clone();
        repl.replace_type(i32_custom_t, INT_TYPES[6].clone());
        let ifthen = IfThen::<Either<_, _>, _, _, _>::new(repl, untup);
        let mut h = h;
        let r = validate_if_test(ifthen, &mut h).unwrap();
        assert_eq!(r, None);
        assert_eq!(h.children(h.entrypoint()).count(), 4);
        let mktup = h
            .output_neighbours(h.first_child(h.entrypoint()).unwrap())
            .next()
            .unwrap();
        assert_eq!(h.get_optype(mktup), &OpType::from(MakeTuple::new(tr)));
    }
}
