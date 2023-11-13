//! Abstract and concrete Signature types.

#[cfg(feature = "pyo3")]
use pyo3::{pyclass, pymethods};

use delegate::delegate;
use std::fmt::{self, Display, Write};

use super::type_param::TypeParam;
use super::{subst_row, Substitution, Type, TypeRow};

use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

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

    pub(crate) fn substitute(&self, tr: &impl Substitution) -> Self {
        FunctionType {
            input: subst_row(&self.input, tr),
            output: subst_row(&self.output, tr),
            extension_reqs: self.extension_reqs.substitute(tr),
        }
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
    pub fn port_type(&self, port: impl Into<Port>) -> Option<&Type> {
        let port = port.into();
        match port.direction() {
            Direction::Incoming => self.input.get(port),
            Direction::Outgoing => self.output.get(port),
        }
    }

    /// Returns a mutable reference to the type of a value [`Port`].
    /// Returns `None` if the port is out of bounds.
    #[inline]
    pub fn port_type_mut(&mut self, port: impl Into<Port>) -> Option<&mut Type> {
        let port = port.into();
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
    /// Returns the `Port`s in the signature for a given direction.
    #[inline]
    pub fn ports(&self, dir: Direction) -> impl Iterator<Item = Port> {
        (0..self.port_count(dir)).map(move |i| Port::new(dir, i))
    }

    /// Returns the incoming `Port`s in the signature.
    #[inline]
    pub fn input_ports(&self) -> impl Iterator<Item = IncomingPort> {
        self.ports(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Returns the outgoing `Port`s in the signature.
    #[inline]
    pub fn output_ports(&self) -> impl Iterator<Item = OutgoingPort> {
        self.ports(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
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

#[cfg(test)]
mod test {
    use crate::{
        extension::{prelude::USIZE_T, ExtensionId},
        type_row,
    };

    use super::*;
    #[test]
    fn test_function_type() {
        let mut f_type = FunctionType::new(type_row![Type::UNIT], type_row![Type::UNIT]);
        assert_eq!(f_type.input_count(), 1);
        assert_eq!(f_type.output_count(), 1);

        assert_eq!(f_type.input_types(), &[Type::UNIT]);

        assert_eq!(
            f_type.port_type(Port::new(Direction::Incoming, 0)),
            Some(&Type::UNIT)
        );

        let out = Port::new(Direction::Outgoing, 0);
        *(f_type.port_type_mut(out).unwrap()) = USIZE_T;

        assert_eq!(f_type.port_type(out), Some(&USIZE_T));

        assert_eq!(f_type.input_types(), &[Type::UNIT]);
        assert_eq!(f_type.output_types(), &[USIZE_T]);
    }

    #[test]
    fn test_signature() {
        let f_type = FunctionType::new(type_row![Type::UNIT], type_row![USIZE_T]);

        let sig: Signature = f_type.pure();

        assert_eq!(sig.input(), &type_row![Type::UNIT]);
        assert_eq!(sig.output(), &type_row![USIZE_T]);
    }

    #[test]
    fn test_display() {
        let f_type = FunctionType::new(type_row![Type::UNIT], type_row![USIZE_T]);
        assert_eq!(f_type.to_string(), "[Tuple([])] -> [[]][usize([])]");
        let sig: Signature = f_type.with_input_extensions(ExtensionSet::singleton(
            &ExtensionId::new("Example").unwrap(),
        ));
        assert_eq!(sig.to_string(), "[Tuple([])] -> [[]][usize([])]");
    }
}
