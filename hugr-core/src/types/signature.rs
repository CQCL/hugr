//! Abstract and concrete Signature types.

use itertools::Either;

use super::type_param::TypeParam;
use super::{RowVariable, Substitution, Transformable, Type, TypeRow, TypeRowRV, TypeTransformer};

use crate::core::PortIndex;
use crate::extension::resolution::{
    ExtensionCollectionError, WeakExtensionRegistry, collect_signature_exts,
};
use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

#[cfg(test)]
use {crate::proptest::RecursionDepth, proptest::prelude::*, proptest_derive::Arbitrary};

/// The concept of "signature" in the spec - the edges required to/from a node
/// or within a [`FuncDefn`], also the target (value) of a call (static).
///
/// [`FuncDefn`]: crate::ops::FuncDefn
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    derive_more::Display,
)]
#[display("{input} -> {output}")]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
pub struct Signature {
    /// The input types of the signature.
    pub input: TypeRow,

    /// The output types of the signature.
    pub output: TypeRow,
}

impl Signature {
    /// Create a new signature with specified inputs and outputs.
    pub fn new(input: impl Into<TypeRow>, output: impl Into<TypeRow>) -> Self {
        let input = input.into();
        let output = output.into();
        Self { input, output }
    }

    /// Create a new signature with the same input and output types (signature of an endomorphic
    /// function).
    pub fn new_endo(row: impl Into<TypeRow>) -> Self {
        let row = row.into();
        Self::new(row.clone(), row)
    }

    /// True if both inputs and outputs are necessarily empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }

    #[inline]
    /// Returns a row of the value inputs of the function.
    #[must_use]
    pub fn input(&self) -> &TypeRow {
        &self.input
    }

    #[inline]
    /// Returns a row of the value outputs of the function.
    #[must_use]
    pub fn output(&self) -> &TypeRow {
        &self.output
    }

    #[inline]
    /// Returns a tuple with the input and output rows of the function.
    #[must_use]
    pub fn io(&self) -> (&TypeRow, &TypeRow) {
        (&self.input, &self.output)
    }

    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self {
            input: self.input.substitute(tr),
            output: self.output.substitute(tr),
        }
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        self.input.validate(var_decls)?;
        self.output.validate(var_decls)
    }

    /// Returns a registry with the concrete extensions used by this signature.
    pub fn used_extensions(&self) -> Result<ExtensionRegistry, ExtensionCollectionError> {
        let mut used = WeakExtensionRegistry::default();
        let mut missing = ExtensionSet::new();

        collect_signature_exts(self, &mut used, &mut missing);

        if missing.is_empty() {
            Ok(used.try_into().expect("all extensions are present"))
        } else {
            Err(ExtensionCollectionError::dropped_signature(self, missing))
        }
    }

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
    #[must_use]
    pub fn port_count(&self, dir: Direction) -> usize {
        match dir {
            Direction::Incoming => self.input.len(),
            Direction::Outgoing => self.output.len(),
        }
    }

    /// Returns the number of input ports in the signature.
    #[inline]
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.port_count(Direction::Incoming)
    }

    /// Returns the number of output ports in the signature.
    #[inline]
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.port_count(Direction::Outgoing)
    }

    /// Returns a slice of the types for the given direction.
    #[inline]
    #[must_use]
    pub fn types(&self, dir: Direction) -> &[Type] {
        match dir {
            Direction::Incoming => &self.input,
            Direction::Outgoing => &self.output,
        }
    }

    /// Returns a slice of the input types.
    #[inline]
    #[must_use]
    pub fn input_types(&self) -> &[Type] {
        self.types(Direction::Incoming)
    }

    /// Returns a slice of the output types.
    #[inline]
    #[must_use]
    pub fn output_types(&self) -> &[Type] {
        self.types(Direction::Outgoing)
    }

    /// Returns the `Port`s in the signature for a given direction.
    #[inline]
    pub fn ports(&self, dir: Direction) -> impl Iterator<Item = Port> + use<> {
        (0..self.port_count(dir)).map(move |i| Port::new(dir, i))
    }

    /// Returns the incoming `Port`s in the signature.
    #[inline]
    pub fn input_ports(&self) -> impl Iterator<Item = IncomingPort> + use<> {
        self.ports(Direction::Incoming)
            .map(|p| p.as_incoming().unwrap())
    }

    /// Returns the outgoing `Port`s in the signature.
    #[inline]
    pub fn output_ports(&self) -> impl Iterator<Item = OutgoingPort> + use<> {
        self.ports(Direction::Outgoing)
            .map(|p| p.as_outgoing().unwrap())
    }
}

impl Transformable for Signature {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        // TODO handle extension sets?
        Ok(self.input.transform(tr)? | self.output.transform(tr)?)
    }
}

/// A function that may contain [`RowVariable`]s and thus has potentially-unknown arity;
/// used for [`OpDef`]'s and passable as a value round a Hugr (see [`Type::new_function`])
/// but not a valid node type.
///
/// [`OpDef`]: crate::extension::OpDef
#[derive(
    Debug,
    Clone,
    Default,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    derive_more::Display,
)]
#[display("{input} -> {output}")]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
pub struct FuncValueType {
    /// The inputs of the function type.
    pub input: TypeRowRV,
    /// The outputs of the function type.
    pub output: TypeRowRV,
}

impl FuncValueType {
    /// Create a new function type with specified inputs and outputs.
    pub fn new(input: impl Into<TypeRowRV>, output: impl Into<TypeRowRV>) -> Self {
        let input = input.into();
        let output = output.into();
        Self { input, output }
    }

    /// Create a new function type with the same input and output types (signature of an endomorphic
    /// function).
    pub fn new_endo(row: impl Into<TypeRowRV>) -> Self {
        let row = row.into();
        Self::new(row.clone(), row)
    }

    /// True if both inputs and outputs are necessarily empty, even after any possible substitution of row variables.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }

    #[inline]
    /// Returns a row of the value inputs of the function.
    #[must_use]
    pub fn input(&self) -> &TypeRowRV {
        &self.input
    }

    #[inline]
    /// Returns a row of the value outputs of the function.
    #[must_use]
    pub fn output(&self) -> &TypeRowRV {
        &self.output
    }

    #[inline]
    /// Returns a tuple with the input and output rows of the function.
    #[must_use]
    pub fn io(&self) -> (&TypeRowRV, &TypeRowRV) {
        (&self.input, &self.output)
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        self.input.validate(var_decls)?;
        self.output.validate(var_decls)
    }

    /// If this `FuncValueType` contains any row variables, return one.
    #[must_use]
    pub fn find_rowvar(&self) -> Option<RowVariable> {
        self.input
            .iter()
            .chain(self.output.iter())
            .find_map(|t| Type::try_from(t.clone()).err())
    }

    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self {
            input: self.input.substitute(tr),
            output: self.output.substitute(tr),
        }
    }
}

impl Transformable for FuncValueType {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        // TODO handle extension sets?
        Ok(self.input.transform(tr)? | self.output.transform(tr)?)
    }
}

impl TryFrom<FuncValueType> for Signature {
    type Error = SignatureError;

    fn try_from(value: FuncValueType) -> Result<Self, Self::Error> {
        let input: TypeRow = value.input.try_into()?;
        let output: TypeRow = value.output.try_into()?;
        Ok(Self::new(input, output))
    }
}

impl From<Signature> for FuncValueType {
    fn from(value: Signature) -> Self {
        Self {
            input: value.input.into(),
            output: value.output.into(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::extension::prelude::{bool_t, qb_t, usize_t};
    use crate::type_row;
    use crate::types::{CustomType, TypeEnum, test::FnTransformer};

    use super::*;
    #[test]
    fn test_function_type() {
        let mut f_type = Signature::new(type_row![Type::UNIT], type_row![Type::UNIT]);
        assert_eq!(f_type.input_count(), 1);
        assert_eq!(f_type.output_count(), 1);

        assert_eq!(f_type.input_types(), &[Type::UNIT]);

        assert_eq!(
            f_type.port_type(Port::new(Direction::Incoming, 0)),
            Some(&Type::UNIT)
        );

        let out = Port::new(Direction::Outgoing, 0);
        *(f_type.port_type_mut(out).unwrap()) = usize_t();

        assert_eq!(f_type.port_type(out), Some(&usize_t()));

        assert_eq!(f_type.input_types(), &[Type::UNIT]);
        assert_eq!(f_type.output_types(), &[usize_t()]);
        assert_eq!(
            f_type.io(),
            (&type_row![Type::UNIT], &vec![usize_t()].into())
        );
    }

    #[test]
    fn test_transform() {
        let TypeEnum::Extension(usz_t) = usize_t().as_type_enum().clone() else {
            panic!()
        };
        let tr = FnTransformer(|ct: &CustomType| (ct == &usz_t).then_some(bool_t()));
        let row_with = || TypeRow::from(vec![usize_t(), qb_t(), bool_t()]);
        let row_after = || TypeRow::from(vec![bool_t(), qb_t(), bool_t()]);
        let mut sig = Signature::new(row_with(), row_after());
        let exp = Signature::new(row_after(), row_after());
        assert_eq!(sig.transform(&tr), Ok(true));
        assert_eq!(sig, exp);
        assert_eq!(sig.transform(&tr), Ok(false));
        assert_eq!(sig, exp);
        let exp = Type::new_function(exp);
        for fty in [
            FuncValueType::new(row_after(), row_with()),
            FuncValueType::new(row_with(), row_with()),
        ] {
            let mut t = Type::new_function(fty);
            assert_eq!(t.transform(&tr), Ok(true));
            assert_eq!(t, exp);
            assert_eq!(t.transform(&tr), Ok(false));
            assert_eq!(t, exp);
        }
    }
}
