//! Abstract and concrete Signature types.

use itertools::Either;

use std::fmt::{self, Display, Write};

use super::type_param::TypeParam;
use super::type_row::TypeRowBase;
use super::{MaybeRV, NoRV, RowVariable, Substitution, Type, TypeRow};

use crate::core::PortIndex;
use crate::extension::resolution::{collect_signature_exts, ExtensionCollectionError};
use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

#[cfg(test)]
use {crate::proptest::RecursionDepth, ::proptest::prelude::*, proptest_derive::Arbitrary};

#[derive(Clone, Debug, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(Arbitrary), proptest(params = "RecursionDepth"))]
/// Describes the edges required to/from a node or inside a [FuncDefn] (when ROWVARS=[NoRV]);
/// or (when ROWVARS=[RowVariable]) the type of a higher-order [function value] or the inputs/outputs from an OpDef
///
/// ROWVARS specifies whether it may contain [RowVariable]s or not.
///
/// [function value]: crate::ops::constant::Value::Function
/// [FuncDefn]: crate::ops::FuncDefn
pub struct FuncTypeBase<ROWVARS: MaybeRV> {
    /// Value inputs of the function.
    #[cfg_attr(test, proptest(strategy = "any_with::<TypeRowBase<ROWVARS>>(params)"))]
    pub input: TypeRowBase<ROWVARS>,
    /// Value outputs of the function.
    #[cfg_attr(test, proptest(strategy = "any_with::<TypeRowBase<ROWVARS>>(params)"))]
    pub output: TypeRowBase<ROWVARS>,
    /// The extension requirements which are added by the operation
    pub extension_reqs: ExtensionSet,
}

/// The concept of "signature" in the spec - the edges required to/from a node
/// or within a [FuncDefn], also the target (value) of a call (static).
///
/// [FuncDefn]: crate::ops::FuncDefn
pub type Signature = FuncTypeBase<NoRV>;

/// A function that may contain [RowVariable]s and thus has potentially-unknown arity;
/// used for [OpDef]'s and passable as a value round a Hugr (see [Type::new_function])
/// but not a valid node type.
///
/// [OpDef]: crate::extension::OpDef
pub type FuncValueType = FuncTypeBase<RowVariable>;

impl<RV: MaybeRV> FuncTypeBase<RV> {
    /// Builder method, add extension_reqs to a FunctionType
    pub fn with_extension_delta(mut self, rs: impl Into<ExtensionSet>) -> Self {
        self.extension_reqs = self.extension_reqs.union(rs.into());
        self
    }

    /// Shorthand for adding the prelude extension to a FunctionType.
    pub fn with_prelude(self) -> Self {
        self.with_extension_delta(crate::extension::prelude::PRELUDE_ID)
    }

    pub(crate) fn substitute(&self, tr: &Substitution) -> Self {
        Self {
            input: self.input.substitute(tr),
            output: self.output.substitute(tr),
            extension_reqs: self.extension_reqs.substitute(tr),
        }
    }

    /// Create a new signature with specified inputs and outputs.
    pub fn new(input: impl Into<TypeRowBase<RV>>, output: impl Into<TypeRowBase<RV>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            extension_reqs: ExtensionSet::new(),
        }
    }

    /// Create a new signature with the same input and output types (signature of an endomorphic
    /// function).
    pub fn new_endo(row: impl Into<TypeRowBase<RV>>) -> Self {
        let row = row.into();
        Self::new(row.clone(), row)
    }

    /// True if both inputs and outputs are necessarily empty.
    /// (For [FuncValueType], even after any possible substitution of row variables)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
    }

    #[inline]
    /// Returns a row of the value inputs of the function.
    pub fn input(&self) -> &TypeRowBase<RV> {
        &self.input
    }

    #[inline]
    /// Returns a row of the value outputs of the function.
    pub fn output(&self) -> &TypeRowBase<RV> {
        &self.output
    }

    #[inline]
    /// Returns a tuple with the input and output rows of the function.
    pub fn io(&self) -> (&TypeRowBase<RV>, &TypeRowBase<RV>) {
        (&self.input, &self.output)
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

    /// Returns a registry with the concrete extensions used by this signature.
    ///
    /// Note that extension type parameters are not included, as they have not
    /// been instantiated yet.
    ///
    /// This method only returns extensions actually used by the types in the
    /// signature. The extension deltas added via [`Self::with_extension_delta`]
    /// refer to _runtime_ extensions, which may not be in all places that
    /// manipulate a HUGR.
    pub fn used_extensions(&self) -> Result<ExtensionRegistry, ExtensionCollectionError> {
        let mut used = ExtensionRegistry::default();
        let mut missing = ExtensionSet::new();

        collect_signature_exts(self, &mut used, &mut missing);

        if missing.is_empty() {
            Ok(used)
        } else {
            Err(ExtensionCollectionError::dropped_signature(self, missing))
        }
    }
}

impl FuncValueType {
    /// If this FuncValueType contains any row variables, return one.
    pub fn find_rowvar(&self) -> Option<RowVariable> {
        self.input
            .iter()
            .chain(self.output.iter())
            .find_map(|t| Type::try_from(t.clone()).err())
    }
}

// deriving Default leads to an impl that only applies for RV: Default
impl<RV: MaybeRV> Default for FuncTypeBase<RV> {
    fn default() -> Self {
        Self {
            input: Default::default(),
            output: Default::default(),
            extension_reqs: Default::default(),
        }
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

impl<RV: MaybeRV> Display for FuncTypeBase<RV> {
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

impl TryFrom<FuncValueType> for Signature {
    type Error = SignatureError;

    fn try_from(value: FuncValueType) -> Result<Self, Self::Error> {
        let input: TypeRow = value.input.try_into()?;
        let output: TypeRow = value.output.try_into()?;
        Ok(Self::new(input, output).with_extension_delta(value.extension_reqs))
    }
}

impl From<Signature> for FuncValueType {
    fn from(value: Signature) -> Self {
        Self {
            input: value.input.into(),
            output: value.output.into(),
            extension_reqs: value.extension_reqs,
        }
    }
}

impl<RV1: MaybeRV, RV2: MaybeRV> PartialEq<FuncTypeBase<RV1>> for FuncTypeBase<RV2> {
    fn eq(&self, other: &FuncTypeBase<RV1>) -> bool {
        self.input == other.input
            && self.output == other.output
            && self.extension_reqs == other.extension_reqs
    }
}

#[cfg(test)]
mod test {
    use crate::{extension::prelude::usize_t, type_row};

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
}
