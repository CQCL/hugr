//! General wire types used in the compiler

pub mod custom;
pub mod simple;

use std::ops::Index;

use portgraph::Direction;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use custom::CustomType;
pub use simple::{ClassicType, Container, LinearType, SimpleType, TypeRow};

pub use portgraph::PortOffset;

use smol_str::SmolStr;

use crate::resource::ResourceSet;

/// The kinds of edges in a HUGR, excluding Hierarchy.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires"
    Value(SimpleType),
    /// A reference to a constant value definition, used in the module region
    Const(ClassicType),
    /// Explicitly enforce an ordering between nodes in a DDG
    StateOrder,
    // An edge specifying a resource set
    Resource(ResourceSet),
}

/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (constant).
///
/// TODO: Consider using Cow here instead of in the TypeRow.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    /// Value inputs of the function
    pub input: TypeRow,
    /// Value outputs of the function
    pub output: TypeRow,
    /// Possible constE input (for call / load-constant)
    pub const_input: Option<ClassicType>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    /// The number of wires in the signature
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_none() && self.input.is_empty() && self.output.is_empty()
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
    /// TODO: This fails when mixing different linear types
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

    /// Returns the port type given a [`PortOffset`]. Returns `None` if the offset is out of bounds.
    pub fn get(&self, offset: PortOffset) -> Option<EdgeKind> {
        if offset.direction() == Direction::Incoming && offset.index() == self.input.len() {
            self.const_input.clone().map(EdgeKind::Const)
        } else {
            self.get_df(offset).cloned().map(EdgeKind::Value)
        }
    }

    /// Returns the port type given a [`PortOffset`]. Returns `None` if the offset is out of bounds.
    pub fn get_df(&self, offset: PortOffset) -> Option<&SimpleType> {
        match offset.direction() {
            Direction::Incoming => self.input.get(offset.index()),
            Direction::Outgoing => self.output.get(offset.index()),
        }
    }

    /// Returns the port type given a [`PortOffset`]. Returns `None` if the offset is out of bounds.
    pub fn get_df_mut(&mut self, offset: PortOffset) -> Option<&mut SimpleType> {
        match offset.direction() {
            Direction::Incoming => self.input.get_mut(offset.index()),
            Direction::Outgoing => self.output.get_mut(offset.index()),
        }
    }
}

impl Signature {
    /// Create a new signature
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
        const_input: impl Into<Option<ClassicType>>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
        }
    }

    /// Create a new signature with the same input and output types
    pub fn new_linear(linear: impl Into<TypeRow>) -> Self {
        let linear = linear.into();
        Signature::new_df(linear.clone(), linear)
    }

    /// Create a new signature with only dataflow inputs and outputs
    pub fn new_df(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        Signature::new(input, output, None)
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
    pub const_input: Option<SmolStr>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl SignatureDescription {
    /// The number of wires in the signature
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_none() && self.input.is_empty() && self.output.is_empty()
    }
}

impl SignatureDescription {
    /// Create a new signature
    pub fn new(
        input: impl Into<Vec<SmolStr>>,
        output: impl Into<Vec<SmolStr>>,
        const_input: impl Into<Option<SmolStr>>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
        }
    }

    /// Create a new signature with only linear dataflow inputs and outputs
    pub fn new_linear(linear: impl Into<Vec<SmolStr>>) -> Self {
        let linear = linear.into();
        SignatureDescription::new_df(linear.clone(), linear)
    }

    /// Create a new signature with only dataflow inputs and outputs
    pub fn new_df(input: impl Into<Vec<SmolStr>>, output: impl Into<Vec<SmolStr>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
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
    pub fn const_input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> Option<(&'a SmolStr, &'a ClassicType)> {
        signature
            .const_input
            .as_ref()
            .map(|t| (self.const_input.as_ref().unwrap_or(EMPTY_STRING_REF), t))
    }
}

impl Index<PortOffset> for SignatureDescription {
    type Output = SmolStr;

    fn index(&self, index: PortOffset) -> &Self::Output {
        match index.direction() {
            Direction::Incoming => self.input.get(index.index()).unwrap_or(EMPTY_STRING_REF),
            Direction::Outgoing => self.output.get(index.index()).unwrap_or(EMPTY_STRING_REF),
        }
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
