//! Abstract and concrete Signature types.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use std::ops::Index;

use smol_str::SmolStr;

use crate::utils::display_list;

use std::fmt::{self, Display, Write};

use crate::hugr::Direction;

use super::{EdgeKind, Type, TypeRow};

use crate::hugr::Port;

use crate::type_row;

use crate::resource::ResourceSet;
use delegate::delegate;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (static).
pub struct AbstractSignature {
    /// Value inputs of the function.
    pub input: TypeRow,
    /// Value outputs of the function.
    pub output: TypeRow,
    /// Possible static input (for call / load-constant).
    pub static_input: TypeRow,
    /// The resource requirements which are added by the operation
    pub resource_reqs: ResourceSet,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// A concrete signature, which has been instantiated with a set of input resources
pub struct Signature {
    /// The underlying signature
    pub signature: AbstractSignature,
    /// The resources which are associated with all the inputs and carried through
    pub input_resources: ResourceSet,
}

impl AbstractSignature {
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
            resource_reqs: ResourceSet::new(),
        }
    }

    /// Builder method, add resource_reqs to an AbstractSignature
    pub fn with_resource_delta(mut self, rs: &ResourceSet) -> Self {
        self.resource_reqs = self.resource_reqs.union(rs);
        self
    }

    /// Instantiate an AbstractSignature, converting it to a concrete one
    pub fn with_input_resources(self, rs: ResourceSet) -> Signature {
        Signature {
            signature: self,
            input_resources: rs,
        }
    }

    /// Instantiate a signature with the empty set of resources
    pub fn pure(self) -> Signature {
        self.with_input_resources(ResourceSet::new())
    }
}

impl From<Signature> for AbstractSignature {
    fn from(sig: Signature) -> Self {
        sig.signature
    }
}

impl Signature {
    /// Calculate the resource requirements of the output wires
    pub fn output_resources(&self) -> ResourceSet {
        self.input_resources
            .clone()
            .union(&self.signature.resource_reqs)
    }
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl AbstractSignature {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.static_input.is_empty() && self.input.is_empty() && self.output.is_empty()
    }
}

impl AbstractSignature {
    /// Create a new signature with only dataflow inputs and outputs.
    pub fn new_df(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        Self::new(input, output, type_row![])
    }
    /// Create a new signature with the same input and output types.
    pub fn new_linear(linear: impl Into<TypeRow>) -> Self {
        let linear = linear.into();
        Self::new_df(linear.clone(), linear)
    }

    /// Returns the type of a [`Port`]. Returns `None` if the port is out of bounds.
    pub fn get(&self, port: Port) -> Option<EdgeKind> {
        if port.direction() == Direction::Incoming && port.index() >= self.input.len() {
            Some(EdgeKind::Static(
                self.static_input
                    .get(port.index() - self.input.len())?
                    .clone(),
            ))
        } else {
            self.get_df(port).cloned().map(EdgeKind::Value)
        }
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds or if it is not a value.
    #[inline]
    pub fn get_df(&self, port: Port) -> Option<&Type> {
        match port.direction() {
            Direction::Incoming => self.input.get(port.index()),
            Direction::Outgoing => self.output.get(port.index()),
        }
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds or if it is not a value.
    #[inline]
    pub fn get_df_mut(&mut self, port: Port) -> Option<&mut Type> {
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

    /// Returns a slice of the value types for the given direction.
    #[inline]
    pub fn df_types(&self, dir: Direction) -> &[Type] {
        match dir {
            Direction::Incoming => &self.input,
            Direction::Outgoing => &self.output,
        }
    }

    /// Returns a slice of the input value types.
    #[inline]
    pub fn input_df_types(&self) -> &[Type] {
        self.df_types(Direction::Incoming)
    }

    /// Returns a slice of the output value types.
    #[inline]
    pub fn output_df_types(&self) -> &[Type] {
        self.df_types(Direction::Outgoing)
    }

    #[inline]
    /// Returns the input row
    pub fn input(&self) -> &TypeRow {
        &self.input
    }

    #[inline]
    /// Returns the output row
    pub fn output(&self) -> &TypeRow {
        &self.output
    }

    #[inline]
    /// Returns the row of static inputs
    pub fn static_input(&self) -> &TypeRow {
        &self.static_input
    }
}

impl AbstractSignature {
    /// Returns the linear part of the signature
    /// TODO: This fails when mixing different linear types.
    #[inline(always)]
    pub fn linear(&self) -> impl Iterator<Item = &Type> {
        debug_assert_eq!(
            self.input
                .iter()
                .filter(|t| t.least_upper_bound().is_none())
                .collect::<Vec<_>>(),
            self.output
                .iter()
                .filter(|t| t.least_upper_bound().is_none())
                .collect::<Vec<_>>()
        );
        self.input
            .iter()
            .filter(|t| t.least_upper_bound().is_none())
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
}

impl Signature {
    /// Returns a reference to the resource set for the ports of the
    /// signature in a given direction
    pub fn get_resources(&self, dir: &Direction) -> ResourceSet {
        match dir {
            Direction::Incoming => self.input_resources.clone(),
            Direction::Outgoing => self.output_resources(),
        }
    }

    delegate! {
        to self.signature {
            /// Inputs of the abstract signature
            pub fn input(&self) -> &TypeRow;
            /// Outputs of the abstract signature
            pub fn output(&self) -> &TypeRow;
            /// Static inputs of the abstract signature
            pub fn static_input(&self) -> &TypeRow;
        }
    }
}

impl Display for AbstractSignature {
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
        f.write_char('[')?;
        self.resource_reqs.fmt(f)?;
        f.write_char(']')?;
        self.output.fmt(f)
    }
}

impl Display for Signature {
    delegate! {
        to self.signature {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result;
        }
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

    pub(crate) fn row_zip<'a>(
        type_row: &'a TypeRow,
        name_row: &'a [SmolStr],
    ) -> impl Iterator<Item = (&'a SmolStr, &'a Type)> {
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
    ) -> impl Iterator<Item = (&SmolStr, &Type)> {
        Self::row_zip(signature.input(), &self.input)
    }

    /// Iterate over the output wires of the signature and their names.
    ///
    /// Unnamed wires are given an empty string name.
    pub fn output_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &Type)> {
        Self::row_zip(signature.output(), &self.output)
    }

    /// Iterate over the static input wires of the signature and their names.
    pub fn static_input_zip<'a>(
        &'a self,
        signature: &'a Signature,
    ) -> impl Iterator<Item = (&SmolStr, &Type)> {
        Self::row_zip(signature.static_input(), &self.static_input)
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
pub(crate) struct EmptyStringIterator;

/// A reference to an empty string. Used by [`EmptyStringIterator`].
pub(crate) const EMPTY_STRING_REF: &SmolStr = &SmolStr::new_inline("");

impl<'a> Iterator for &'a EmptyStringIterator {
    type Item = &'a SmolStr;

    fn next(&mut self) -> Option<Self::Item> {
        Some(EMPTY_STRING_REF)
    }
}
