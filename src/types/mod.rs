//! General wire types used in the compiler

pub mod custom;
pub mod simple;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use custom::CustomType;
pub use simple::{ClassicType, QuantumType, SimpleType, TypeRow};
use smol_str::SmolStr;

use crate::resource::ResourceSet;

/// The wire types
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region
    ControlFlow,
    /// Data edges of a DDG region
    Value(SimpleType),
    /// A reference to a constant value definition, used in the module region
    Const(ClassicType),
    /// A strict ordering between nodes
    StateOrder,
    // An edge specifying a resource set
    Resource(ResourceSet),
}

/// A function signature with dataflow types. This does not specify control flow
/// ports nor state ordering
///
/// TODO: Here we split the input and output into two parts, one for value wires and
/// one for constant definitions. This allows us to reuse the `RowType` type,
/// but requires that the value ports come all before the constants. We could
/// change this by redefining
/// ```text
/// enum RowTypeVariant {df: DataType, const: DataType}
/// struct RowType(Vec<RowTypeVariant>);
/// ```
/// but that seems more annoying to work with.
/// That would reduce the size of `OpType` by about 50%, so it's worth considering.
///
/// TODO: Option2 is using Cow here instead of in the TypeRow.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    /// Input of the function
    pub input: TypeRow,
    /// Output of the function
    pub output: TypeRow,
    /// Constant data references used by the function
    pub const_input: TypeRow,
    /// Constant data references defined by the function
    pub const_output: TypeRow,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    /// The number of wires in the signature
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_empty()
            && self.const_output.is_empty()
            && self.input.is_empty()
            && self.output.is_empty()
    }

    /// Returns whether the data wires in the signature are purely linear
    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.input.purely_linear() && self.output.purely_linear()
    }

    /// Returns whether the data wires in the signature are purely classical
    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        self.input.purely_classical() && self.output.purely_classical()
    }
}
impl Signature {
    /// Returns the linear part of the signature
    #[inline(always)]
    pub fn linear(&self) -> impl Iterator<Item = &SimpleType> {
        debug_assert_eq!(
            self.input
                .iter()
                .filter(|t| t.is_linear())
                .collect::<Vec<_>>(),
            self.output
                .iter()
                .filter(|t| t.is_linear())
                .collect::<Vec<_>>()
        );
        self.input.iter().filter(|t| t.is_linear())
    }
}

impl Signature {
    /// Create a new signature
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
        const_input: impl Into<TypeRow>,
        const_output: impl Into<TypeRow>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
            const_output: const_output.into(),
        }
    }

    /// Create a new signature with the same input and output types
    pub fn new_linear(linear: impl Into<TypeRow>) -> Self {
        let linear = linear.into();
        Self {
            input: linear.clone(),
            output: linear,
            ..Default::default()
        }
    }

    /// Create a new signature with only dataflow inputs and outputs
    pub fn new_df(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            ..Default::default()
        }
    }

    /// Create a new signature with only constant outputs
    pub fn new_const(const_output: impl Into<TypeRow>) -> Self {
        Self {
            const_output: const_output.into(),
            ..Default::default()
        }
    }
}

/// Descriptive names for the ports in a [`Signature`].
///
/// This is a separate type from [`Signature`] as it is not normally used during the compiler operations.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SignatureDescription {
    /// Input of the function
    pub input: Vec<SmolStr>,
    /// Output of the function
    pub output: Vec<SmolStr>,
    /// Constant data references used by the function
    pub const_input: Vec<SmolStr>,
    /// Constant data references defined by the function
    pub const_output: Vec<SmolStr>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl SignatureDescription {
    /// The number of wires in the signature
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_empty()
            && self.const_output.is_empty()
            && self.input.is_empty()
            && self.output.is_empty()
    }
}

impl SignatureDescription {
    /// Create a new signature
    pub fn new(
        input: impl Into<Vec<SmolStr>>,
        output: impl Into<Vec<SmolStr>>,
        const_input: impl Into<Vec<SmolStr>>,
        const_output: impl Into<Vec<SmolStr>>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
            const_output: const_output.into(),
        }
    }

    /// Create a new signature with only linear dataflow inputs and outputs
    pub fn new_linear(linear: impl Into<Vec<SmolStr>>) -> Self {
        let linear = linear.into();
        Self {
            input: linear.clone(),
            output: linear,
            ..Default::default()
        }
    }

    /// Create a new signature with only dataflow inputs and outputs
    pub fn new_df(input: impl Into<Vec<SmolStr>>, output: impl Into<Vec<SmolStr>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            ..Default::default()
        }
    }

    /// Create a new signature with only constant outputs
    pub fn new_const(const_output: impl Into<Vec<SmolStr>>) -> Self {
        Self {
            const_output: const_output.into(),
            ..Default::default()
        }
    }

    /// Iterate over the input wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    ///
    /// TODO: Return Option<&String> instead of &String for the description
    pub fn input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        self.input
            .iter()
            .chain(&EmptyStringIterator)
            .zip(signature.input.iter())
    }

    /// Iterate over the output wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    pub fn output_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        self.output
            .iter()
            .chain(&EmptyStringIterator)
            .zip(signature.output.iter())
    }

    /// Iterate over the constant input wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    pub fn const_input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        self.const_input
            .iter()
            .chain(&EmptyStringIterator)
            .zip(signature.const_input.iter())
    }

    /// Iterate over the constant output wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    pub fn const_output_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        self.const_output
            .iter()
            .chain(&EmptyStringIterator)
            .zip(signature.const_output.iter())
    }
}

/// An iterator that always returns the an empty string.
struct EmptyStringIterator;

/// A reference to an empty string. Used by [`EmptyStringIterator`].
const EMPTY_STRING_REF: &SmolStr = &SmolStr::new_inline("");

impl<'a> Iterator for &'a EmptyStringIterator {
    type Item = &'a SmolStr;

    fn next(&mut self) -> Option<Self::Item> {
        Some(EMPTY_STRING_REF)
    }
}
