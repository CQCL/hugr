//! General wire types used in the compiler

pub mod custom;
pub mod simple;
pub mod type_param;

use std::fmt::{self, Display, Write};
use std::ops::Index;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use custom::CustomType;
pub use simple::{ClassicType, Container, SimpleType, TypeRow};

use smol_str::SmolStr;

use crate::hugr::{Direction, Port};
use crate::utils::display_list;
use crate::{resource::ResourceSet, type_row};

/// The kinds of edges in a HUGR, excluding Hierarchy.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(SimpleType),
    /// A reference to a static value definition.
    Static(ClassicType),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
}

impl EdgeKind {
    /// Returns whether the type contains only linear data.
    pub fn is_linear(&self) -> bool {
        match self {
            EdgeKind::Value(t) => t.is_linear(),
            _ => false,
        }
    }
}

/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (static).
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    /// Value inputs of the function.
    pub input: TypeRow,
    /// Value outputs of the function.
    pub output: TypeRow,
    /// Possible static input (for call / load-constant).
    pub static_input: TypeRow,
    /// The resource requirements of all the inputs
    pub input_resources: ResourceSet,
    /// The resource requirements of all the outputs
    pub output_resources: ResourceSet,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.static_input.is_empty() && self.input.is_empty() && self.output.is_empty()
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
            self.static_input
                .get(port.index() - self.input.len())?
                .clone()
                .try_into()
                .ok()
                .map(EdgeKind::Static)
        } else {
            self.get_df(port).cloned().map(EdgeKind::Value)
        }
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds or if it is not a value.
    #[inline]
    pub fn get_df(&self, port: Port) -> Option<&SimpleType> {
        match port.direction() {
            Direction::Incoming => self.input.get(port.index()),
            Direction::Outgoing => self.output.get(port.index()),
        }
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds or if it is not a value.
    #[inline]
    pub fn get_df_mut(&mut self, port: Port) -> Option<&mut SimpleType> {
        match port.direction() {
            Direction::Incoming => self.input.get_mut(port.index()),
            Direction::Outgoing => self.output.get_mut(port.index()),
        }
    }

    /// Returns the number of value and static ports in the signature.
    #[inline]
    pub fn port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => self.input.len() + self.static_input.len(),
            Direction::Outgoing => self.output.len(),
        }
    }

    /// Returns the number of input value and static ports in the signature.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of output value and static ports in the signature.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Returns the number of value ports in the signature.
    #[inline]
    pub fn df_port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => self.input.len(),
            Direction::Outgoing => self.output.len(),
        }
    }

    /// Returns a reference to the resource set for the ports of the
    /// signature in a given direction
    pub fn get_resources(&self, dir: &Direction) -> &ResourceSet {
        match dir {
            Direction::Incoming => &self.input_resources,
            Direction::Outgoing => &self.output_resources,
        }
    }

    /// Returns the value `Port`s in the signature for a given direction.
    #[inline]
    pub fn ports_df(&self, dir: Direction) -> impl Iterator<Item = Port> {
        (0..self.df_port_count(dir)).map(move |i| Port::new(dir, i))
    }

    /// Returns the incoming value `Port`s in the signature.
    #[inline]
    pub fn input_ports_df(&self) -> impl Iterator<Item = Port> {
        self.ports_df(Direction::Incoming)
    }

    /// Returns the outgoing value `Port`s in the signature.
    #[inline]
    pub fn output_ports_df(&self) -> impl Iterator<Item = Port> {
        self.ports_df(Direction::Outgoing)
    }

    /// Returns the `Port`s in the signature for a given direction.
    #[inline]
    pub fn ports(&self, dir: Direction) -> impl Iterator<Item = Port> {
        (0..self.port_count(dir)).map(move |i| Port::new(dir, i))
    }

    /// Returns the incoming `Port`s in the signature.
    #[inline]
    pub fn input_ports(&self) -> impl Iterator<Item = Port> {
        self.ports(Direction::Incoming)
    }

    /// Returns the outgoing `Port`s in the signature.
    #[inline]
    pub fn output_ports(&self) -> impl Iterator<Item = Port> {
        self.ports(Direction::Outgoing)
    }

    /// Returns a slice of the value types for the given direction.
    #[inline]
    pub fn df_types(&self, dir: Direction) -> &[SimpleType] {
        match dir {
            Direction::Incoming => &self.input,
            Direction::Outgoing => &self.output,
        }
    }

    /// Returns a slice of the input value types.
    #[inline]
    pub fn input_df_types(&self) -> &[SimpleType] {
        self.df_types(Direction::Incoming)
    }

    /// Returns a slice of the output value types.
    #[inline]
    pub fn output_df_types(&self) -> &[SimpleType] {
        self.df_types(Direction::Outgoing)
    }
}

impl Signature {
    /// Create a new signature.
    pub fn new(
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
        static_input: impl Into<TypeRow>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            static_input: static_input.into(),
            input_resources: ResourceSet::new(),
            output_resources: ResourceSet::new(),
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
        let has_inputs = !(self.static_input.is_empty() && self.input.is_empty());
        if has_inputs {
            self.input.fmt(f)?;
            if !self.static_input.is_empty() {
                f.write_char('<')?;
                display_list(&self.static_input, f)?;
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
    /// Static data references used by the function.
    pub static_input: Vec<SmolStr>,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl SignatureDescription {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.static_input.is_empty() && self.input.is_empty() && self.output.is_empty()
    }
}

impl SignatureDescription {
    /// Create a new signature.
    pub fn new(
        input: impl Into<Vec<SmolStr>>,
        output: impl Into<Vec<SmolStr>>,
        static_input: impl Into<Vec<SmolStr>>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            static_input: static_input.into(),
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

    /// Iterate over the static input wires of the signature and their names.
    pub fn static_input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &SimpleType)> {
        Self::row_zip(&signature.static_input, &self.static_input)
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
