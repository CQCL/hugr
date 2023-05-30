//! General wire types used in the compiler

pub mod custom;
pub mod simple;

use std::fmt::{self, Display, Write};
use std::ops::Index;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use custom::CustomType;
pub use simple::{ClassicType, Container, LinearType, SimpleType, TypeRow};

use smol_str::SmolStr;

use crate::hugr::{Direction, Port};
use crate::{resource::ResourceSet, type_row};
use crate::utils::display_list;

/// The kinds of edges in a HUGR, excluding Hierarchy.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(SimpleType),
    /// A reference to a constant value definition, used in the module region.
    Const(ClassicType),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
    /// An edge specifying a resource set.
    Resource(ResourceSet),
}

/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (constant).
///
/// TODO: Consider using Cow here instead of in the TypeRow.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    /// Value inputs of the function.
    pub input: TypeRow,
    /// Value outputs of the function.
    pub output: TypeRow,
    /// Possible constE input (for call / load-constant).
    pub const_input: TypeRow,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_empty() && self.input.is_empty() && self.output.is_empty()
    }

    /// Returns whether the data wires in the signature are purely linear.
    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.input.purely_linear() && self.output.purely_linear()
    }

    /// Returns whether the data wires in the signature are purely classical.
    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        self.input.purely_classical() && self.output.purely_classical()
    }
}
impl Signature {
    /// Returns the linear part of the signature
    /// TODO: This fails when mixing different linear types.
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

    /// Returns the type of a [`Port`]. Returns `None` if the port is out of bounds.
    pub fn get(&self, port: Port) -> Option<EdgeKind> {
        if port.direction() == Direction::Incoming && port.index() >= self.input.len() {
            self.const_input
                .get(port.index() - self.input.len())?
                .clone()
                .try_into()
                .ok()
                .map(EdgeKind::Const)
        } else {
            self.get_df(port).cloned().map(EdgeKind::Value)
        }
    }

    /// Returns the type of a [`Port`]. Returns `None` if the port is out of bounds.
    #[inline]
    pub fn get_df(&self, port: Port) -> Option<&SimpleType> {
        match port.direction() {
            Direction::Incoming => self.input.get(port.index()),
            Direction::Outgoing => self.output.get(port.index()),
        }
    }

    /// Returns the type of a [`Port`]. Returns `None` if the port is out of bounds.
    #[inline]
    pub fn get_df_mut(&mut self, port: Port) -> Option<&mut SimpleType> {
        match port.direction() {
            Direction::Incoming => self.input.get_mut(port.index()),
            Direction::Outgoing => self.output.get_mut(port.index()),
        }
    }
}

impl Signature {
    /// Create a new signature.
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
        const_input: impl Into<TypeRow>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
        }
    }

    /// Create a new signature with the same input and output types.
    pub fn new_linear(linear: impl Into<TypeRow>) -> Self {
        let linear = linear.into();
        Signature::new_df(linear.clone(), linear)
    }

    /// Create a new signature with only dataflow inputs and outputs.
    pub fn new_df(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        Signature::new(input, output, type_row![])
    }
}

impl Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let has_inputs = !(self.const_input.is_empty() && self.input.is_empty());
        if has_inputs {
            self.input.fmt(f)?;
            if !self.const_input.is_empty() {
                f.write_char('<')?;
                display_list(&self.const_input, f)?;
                f.write_char('>')?;
            }
            f.write_str(" -> ")?;
        }
        self.output.fmt(f)
    }
}

/// Descriptive names for the ports in a [`Signature`].
///
/// This is a separate type from [`Signature`] as it is not normally used during the compiler operations.
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SignatureDescription {
    /// Input of the function.
    pub input: Vec<SmolStr>,
    /// Output of the function.
    pub output: Vec<SmolStr>,
    /// Constant data references used by the function.
    pub const_input: Vec<SmolStr>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl SignatureDescription {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.const_input.is_empty() && self.input.is_empty() && self.output.is_empty()
    }
}

impl SignatureDescription {
    /// Create a new signature.
    pub fn new(
        input: impl Into<Vec<SmolStr>>,
        output: impl Into<Vec<SmolStr>>,
        const_input: impl Into<Vec<SmolStr>>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            const_input: const_input.into(),
        }
    }

    /// Create a new signature with only linear dataflow inputs and outputs.
    pub fn new_linear(linear: impl Into<Vec<SmolStr>>) -> Self {
        let linear = linear.into();
        SignatureDescription::new_df(linear.clone(), linear)
    }

    /// Create a new signature with only dataflow inputs and outputs.
    pub fn new_df(input: impl Into<Vec<SmolStr>>, output: impl Into<Vec<SmolStr>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            ..Default::default()
        }
    }

    fn row_zip<'a>(
        type_row: &'a TypeRow,
        name_row: &'a [SmolStr],
    ) -> impl Iterator<Item = (&'a SmolStr, &'a SimpleType)> {
        name_row
            .iter()
            .chain(&EmptyStringIterator)
            .zip(type_row.iter())
    }

    /// Iterate over the input wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    ///
    /// TODO: Return Option<&String> instead of &String for the description.
    pub fn input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        Self::row_zip(&signature.input, &self.input)
    }

    /// Iterate over the output wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    pub fn output_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        Self::row_zip(&signature.output, &self.output)
    }

    /// Iterate over the constant input wires of the signature and their names.
    pub fn const_input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        Self::row_zip(&signature.const_input, &self.const_input)
    }
}

impl Index<Port> for SignatureDescription {
    type Output = SmolStr;

    fn index(&self, index: Port) -> &Self::Output {
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
