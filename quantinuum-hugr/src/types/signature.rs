//! Abstract and concrete Signature types.

use itertools::Either;

use std::fmt::{self, Display, Write};

use super::type_param::TypeParam;
use super::type_row::{RowVarOrType, TypeRowBase};
use super::{check_typevar_decl, Substitution, Type, TypeBound};

use crate::extension::{ExtensionRegistry, ExtensionSet, SignatureError};
use crate::{Direction, IncomingPort, OutgoingPort, Port};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Describes the edges required to/from a node. This includes both the concept of "signature" in the spec,
/// and also the target (value) of a call (static).
pub struct FuncTypeBase<T>
where
    T: TypeRowElem,
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Value inputs of the function.
    pub input: TypeRowBase<T>,
    /// Value outputs of the function.
    pub output: TypeRowBase<T>,
    /// The extension requirements which are added by the operation
    pub extension_reqs: ExtensionSet,
}

pub(super) trait TypeRowElem: 'static + Sized + Clone
where
    [Self]: ToOwned<Owned = Vec<Self>>,
{
    fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError>;

    fn subst_row(row: &TypeRowBase<Self>, tr: &impl Substitution) -> TypeRowBase<Self>;
}

impl<T> Default for FuncTypeBase<T>
where
    T: TypeRowElem,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn default() -> Self {
        Self {
            input: Default::default(),
            output: Default::default(),
            extension_reqs: Default::default(),
        }
    }
}

/// The type of a function, e.g. passing around a pointer/static ref to it.
pub type FunctionType = FuncTypeBase<RowVarOrType>;

impl TypeRowElem for RowVarOrType {
    fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        match self {
            RowVarOrType::T(t) => t.validate(extension_registry, var_decls),
            RowVarOrType::RV(idx, bound) => {
                let t = TypeParam::List {
                    param: Box::new((*bound).into()),
                };
                check_typevar_decl(var_decls, *idx, &t)
            }
        }
    }

    fn subst_row(row: &TypeRowBase<Self>, tr: &impl Substitution) -> TypeRowBase<Self> {
        row.iter()
            .flat_map(|ty| match ty {
                RowVarOrType::RV(idx, bound) => tr.apply_rowvar(*idx, *bound),
                RowVarOrType::T(t) => vec![RowVarOrType::T(t.substitute(tr))],
            })
            .collect::<Vec<_>>()
            .into()
    }
}

/// The type of a node. Fixed/known arity of inputs + outputs.
pub type Signature = FuncTypeBase<Type>;

impl TypeRowElem for Type {
    fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.validate(extension_registry, var_decls)
    }

    fn subst_row(row: &TypeRowBase<Self>, tr: &impl Substitution) -> TypeRowBase<Self> {
        row.iter()
            .map(|t| t.substitute(tr))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<T> FuncTypeBase<T>
where
    T: TypeRowElem,
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Builder method, add extension_reqs to an FunctionType
    pub fn with_extension_delta(mut self, rs: impl Into<ExtensionSet>) -> Self {
        self.extension_reqs = self.extension_reqs.union(rs.into());
        self
    }

    /// Create a new signature with specified inputs and outputs.
    pub fn new(input: impl Into<TypeRowBase<T>>, output: impl Into<TypeRowBase<T>>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            extension_reqs: ExtensionSet::new(),
        }
    }
    /// Create a new signature with the same input and output types (signature of an endomorphic
    /// function).
    pub fn new_endo(linear: impl Into<TypeRowBase<T>>) -> Self {
        let linear = linear.into();
        Self::new(linear.clone(), linear)
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.input
            .iter()
            .chain(self.output.iter())
            .try_for_each(|e| e.validate(extension_registry, var_decls))?;
        self.extension_reqs.validate(var_decls)
    }

    pub(crate) fn substitute(&self, tr: &impl Substitution) -> Self {
        Self {
            input: TypeRowElem::subst_row(&self.input, tr),
            output: TypeRowElem::subst_row(&self.output, tr),
            extension_reqs: self.extension_reqs.substitute(tr),
        }
    }

    #[inline]
    /// Returns the input row
    pub fn input(&self) -> &TypeRowBase<T> {
        &self.input
    }

    #[inline]
    /// Returns the output row
    pub fn output(&self) -> &TypeRowBase<T> {
        &self.output
    }
}

impl Signature {
    /// The number of wires in the signature.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.input.is_empty() && self.output.is_empty()
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
        self.input.get(port.into())
    }

    /// Returns the type of a value output [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn out_port_type(&self, port: impl Into<OutgoingPort>) -> Option<&Type> {
        self.output.get(port.into())
    }

    /// Returns a mutable reference to the type of a value input [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn in_port_type_mut(&mut self, port: impl Into<IncomingPort>) -> Option<&mut Type> {
        self.input.get_mut(port.into())
    }

    /// Returns the type of a value output [`Port`]. Returns `None` if the port is out
    /// of bounds.
    #[inline]
    pub fn out_port_type_mut(&mut self, port: impl Into<OutgoingPort>) -> Option<&mut Type> {
        self.output.get_mut(port.into())
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

impl From<Signature> for FunctionType {
    fn from(sig: Signature) -> Self {
        Self {
            input: sig.input.into(),
            output: sig.output.into(),
            extension_reqs: sig.extension_reqs,
        }
    }
}

impl TryFrom<FunctionType> for Signature {
    type Error = (usize, TypeBound);

    fn try_from(funty: FunctionType) -> Result<Self, Self::Error> {
        Ok(Self {
            input: funty.input.try_into()?,
            output: funty.output.try_into()?,
            extension_reqs: funty.extension_reqs,
        })
    }
}

impl<T: Display + TypeRowElem> Display for FuncTypeBase<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
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

#[cfg(test)]
mod test {
    use crate::{extension::prelude::USIZE_T, type_row};

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
        *(f_type.port_type_mut(out).unwrap()) = USIZE_T;

        assert_eq!(f_type.port_type(out), Some(&USIZE_T));

        assert_eq!(f_type.input_types(), &[Type::UNIT]);
        assert_eq!(f_type.output_types(), &[USIZE_T]);
    }
}
