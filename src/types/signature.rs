//! Abstract and concrete Signature types.

#[cfg(feature = "pyo3")]
use pyo3::{pyclass, pymethods};

use std::ops::Index;

use smol_str::SmolStr;

use std::fmt::{self, Display, Write};

use crate::hugr::{Direction, PortIndex};

use super::type_param::TypeParam;
use super::{Type, TypeRow};

use crate::hugr::Port;

use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use delegate::delegate;

#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (static).
pub struct FunctionType {
    /// Value inputs of the function.
    pub input: TypeRow,
    /// Value outputs of the function.
    pub output: TypeRow,
    /// The extension requirements which are added by the operation
    pub extension_reqs: ExtensionSet,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// A concrete signature, which has been instantiated with a set of input extensions
pub struct Signature {
    /// The underlying signature
    pub signature: FunctionType,
    /// The extensions which are associated with all the inputs and carried through
    pub input_extensions: ExtensionSet,
}

impl FunctionType {
    /// Builder method, add extension_reqs to an FunctionType
    pub fn with_extension_delta(mut self, rs: &ExtensionSet) -> Self {
        self.extension_reqs = self.extension_reqs.union(rs);
        self
    }

    /// Instantiate an FunctionType, converting it to a concrete one
    pub fn with_input_extensions(self, es: ExtensionSet) -> Signature {
        Signature {
            signature: self,
            input_extensions: es,
        }
    }

    /// Instantiate a signature with the empty set of extensions
    pub fn pure(self) -> Signature {
        self.with_input_extensions(ExtensionSet::new())
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.input
            .iter()
            .chain(self.output.iter())
            .try_for_each(|t| t.validate(extension_registry, var_decls))?;
        self.extension_reqs.validate(var_decls)
    }
}

impl From<Signature> for FunctionType {
    fn from(sig: Signature) -> Self {
        sig.signature
    }
}

impl Signature {
    /// Calculate the extension requirements of the output wires
    pub fn output_extensions(&self) -> ExtensionSet {
        self.input_extensions
            .clone()
            .union(&self.signature.extension_reqs)
    }
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl FunctionType {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }
}

impl FunctionType {
    /// Create a new signature with specified inputs and outputs.
    pub fn new(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            extension_reqs: ExtensionSet::new(),
        }
    }
    /// Create a new signature with the same input and output types.
    pub fn new_linear(linear: impl Into<TypeRow>) -> Self {
        let linear = linear.into();
        Self::new(linear.clone(), linear)
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn get(&self, port: Port) -> Option<&Type> {
        match port.direction() {
            Direction::Incoming => self.input.get(port),
            Direction::Outgoing => self.output.get(port),
        }
    }

    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn get_mut(&mut self, port: Port) -> Option<&mut Type> {
        match port.direction() {
            Direction::Incoming => self.input.get_mut(port),
            Direction::Outgoing => self.output.get_mut(port),
        }
    }

    /// Returns the number of ports in the signature.
    #[inline]
    pub fn port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => self.input.len(),
            Direction::Outgoing => self.output.len(),
        }
    }

    /// Returns the number of input ports in the signature.
    #[inline]
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of output ports in the signature.
    #[inline]
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Returns a slice of the types for the given direction.
    #[inline]
    pub fn types(&self, dir: Direction) -> &[Type] {
        match dir {
            Direction::Incoming => &self.input,
            Direction::Outgoing => &self.output,
        }
    }

    /// Returns a slice of the input types.
    #[inline]
    pub fn input_types(&self) -> &[Type] {
        self.types(Direction::Incoming)
    }

    /// Returns a slice of the output types.
    #[inline]
    pub fn output_types(&self) -> &[Type] {
        self.types(Direction::Outgoing)
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
}

impl FunctionType {
    /// Returns the linear part of the signature
    /// TODO: This fails when mixing different linear types.
    #[inline(always)]
    pub fn linear(&self) -> impl Iterator<Item = &Type> {
        debug_assert_eq!(
            self.input
                .iter()
                .filter(|t| !t.copyable())
                .collect::<Vec<_>>(),
            self.output
                .iter()
                .filter(|t| !t.copyable())
                .collect::<Vec<_>>()
        );
        self.input.iter().filter(|t| !t.copyable())
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
    /// Returns a reference to the extension set for the ports of the
    /// signature in a given direction
    pub fn get_extension(&self, dir: &Direction) -> ExtensionSet {
        match dir {
            Direction::Incoming => self.input_extensions.clone(),
            Direction::Outgoing => self.output_extensions(),
        }
    }

    delegate! {
        to self.signature {
            /// Inputs of the function type
            pub fn input(&self) -> &TypeRow;
            /// Outputs of the function type
            pub fn output(&self) -> &TypeRow;
        }
    }
}

impl Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.input.is_empty() {
            self.input.fmt(f)?;
            f.write_str(" -> ")?;
        }
        f.write_char('[')?;
        self.extension_reqs.fmt(f)?;
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
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl SignatureDescription {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }
}

impl SignatureDescription {
    /// Create a new signature.
    pub fn new(input: impl Into<Vec<SmolStr>>, output: impl Into<Vec<SmolStr>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
        }
    }

    /// Create a new signature with only linear inputs and outputs.
    pub fn new_linear(linear: impl Into<Vec<SmolStr>>) -> Self {
        let linear = linear.into();
        SignatureDescription::new(linear.clone(), linear)
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
