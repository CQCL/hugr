//! Abstract and concrete Signature types.

use itertools::Either;

use std::fmt::{self, Display, Write};

use super::type_param::TypeParam;
use super::{Substitution, Type, TypeBound, TypeEnum, TypeRow};

use crate::core::PortIndex;
use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

#[cfg(test)]
use {crate::proptest::RecursionDepth, ::proptest::prelude::*, proptest_derive::Arbitrary};

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
/// Describes the edges required to/from a node (when ROWVARS=false);
/// or (when ROWVARS=true) the type of a [Graph] or the inputs/outputs from an OpDef
///
/// ROWVARS specifies whether it may contain [RowVariable]s or not.
///
/// [Graph]: crate::ops::constant::Value::Function
/// [RowVariable]: crate::types::TypeEnum::RowVariable
pub struct FunctionType<const ROWVARS: bool = true> {
    /// Value inputs of the function.
    #[cfg_attr(test, proptest(strategy = "any_with::<TypeRow>(params)"))]
    pub input: TypeRow<ROWVARS>,
    /// Value outputs of the function.
    #[cfg_attr(test, proptest(strategy = "any_with::<TypeRow>(params)"))]
    pub output: TypeRow<ROWVARS>,
    /// The extension requirements which are added by the operation
    pub extension_reqs: ExtensionSet,
}

/// The concept of "signature" in the spec - the edges required to/from a node or graph
/// and also the target (value) of a call (static).
pub type Signature = FunctionType<false>;

impl<const RV: bool> FunctionType<RV> {
    /// Builder method, add extension_reqs to an FunctionType
    pub fn with_extension_delta(mut self, rs: impl Into<ExtensionSet>) -> Self {
        self.extension_reqs = self.extension_reqs.union(rs.into());
        self
    }

    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self {
            input: self.input.substitute(tr),
            output: self.output.substitute(tr),
            extension_reqs: self.extension_reqs.substitute(tr),
        }
    }

    pub fn new(input: impl Into<TypeRow<RV>>, output: impl Into<TypeRow<RV>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            extension_reqs: ExtensionSet::default(),
        }
    }

    pub fn new_endo(row: impl Into<TypeRow<RV>>) -> Self {
        let row = row.into();
        Self::new(row.clone(), row)
    }

    /// True if both inputs and outputs are necessarily empty.
    /// (For [FunctionType], even after any possible substitution of row variables)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }

    #[inline]
    /// Returns a row of the value inputs of the function.
    pub fn input(&self) -> &TypeRow<RV> {
        &self.input
    }

    #[inline]
    /// Returns a row of the value outputs of the function.
    pub fn output(&self) -> &TypeRow<RV> {
        &self.output
    }

    pub(super) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.input.validate(extension_registry, var_decls)?;
        self.output.validate(extension_registry, var_decls)?;
        self.extension_reqs.validate(var_decls)
    }

    pub(crate) fn into_rv(self) -> FunctionType<true> {
        FunctionType {
            input: self.input.into_rv(),
            output: self.output.into_rv(),
            extension_reqs: self.extension_reqs
        }
    }
}

impl FunctionType<true> {
    /// If this FunctionType contains any row variables, return one.
    pub fn find_rowvar(&self) -> Option<(usize, TypeBound)> {
        self.input
            .iter()
            .chain(self.output.iter())
            .find_map(|t| match t.0 {
                TypeEnum::RowVariable(idx, bound) => Some((idx, bound)),
                _ => None,
            })
    }
}

impl Signature {
    /// Returns the type of a value [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn port_type(&self, port: impl Into<Port>) -> Option<&Type> {
        let port: Port = port.into();
        match port.as_directed() {
            Either::Left(port) => self.in_port_type(port),
            Either::Right(port) => self.out_port_type(port),
        }
    }

    /// Returns the type of a value input [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn in_port_type(&self, port: impl Into<IncomingPort>) -> Option<&Type> {
        self.input.get(port.into().index())
    }

    /// Returns the type of a value output [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn out_port_type(&self, port: impl Into<OutgoingPort>) -> Option<&Type> {
        self.output.get(port.into().index())
    }

    /// Returns a mutable reference to the type of a value input [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn in_port_type_mut(&mut self, port: impl Into<IncomingPort>) -> Option<&mut Type> {
        self.input.get_mut(port.into().index())
    }

    /// Returns the type of a value output [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn out_port_type_mut(&mut self, port: impl Into<OutgoingPort>) -> Option<&mut Type> {
        self.output.get_mut(port.into().index())
    }

    /// Returns a mutable reference to the type of a value [`Port`].
    /// Returns `None` if the port is out of bounds.
    #[inline]
    pub fn port_type_mut(&mut self, port: impl Into<Port>) -> Option<&mut Type> {
        let port = port.into();
        match port.as_directed() {
            Either::Left(port) => self.in_port_type_mut(port),
            Either::Right(port) => self.out_port_type_mut(port),
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

impl<const RV: bool> Display for FunctionType<RV> {
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

impl TryFrom<FunctionType> for Signature {
    type Error = SignatureError;

    fn try_from(value: FunctionType) -> Result<Self, Self::Error> {
        let input: TypeRow<false> = value.input.try_into()?;
        let output: TypeRow<false> = value.output.try_into()?;
        Ok(Self::new(input, output).with_extension_delta(value.extension_reqs))
    }
}

impl From<Signature> for FunctionType {
    fn from(value: Signature) -> Self {
        Self::new(value.input.into_rv(), value.output.into_rv())
            .with_extension_delta(value.extension_reqs)
    }
}

impl PartialEq<FunctionType> for Signature {
    fn eq(&self, other: &FunctionType) -> bool {
        self.input == other.input
            && self.output == other.output
            && self.extension_reqs == other.extension_reqs
    }
}

#[cfg(test)]
mod test {
    use crate::extension::prelude::USIZE_T;

    use super::*;
    #[test]
    fn test_function_type() {
        let mut f_type = FunctionType::try_new(Type::UNIT, Type::UNIT).unwrap();
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
}
