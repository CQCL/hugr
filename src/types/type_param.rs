//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::extension::TypeDef

use std::num::NonZeroU64;

use thiserror::Error;

use crate::extension::ExtensionSet;

use super::CustomType;
use super::Type;
use super::TypeBound;

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
/// The upper non-inclusive bound of a [`TypeParam::BoundedNat`]
// A None inner value implies the maximum bound: u64::MAX + 1 (all u64 values valid)
pub struct UpperBound(Option<NonZeroU64>);
impl UpperBound {
    fn valid_value(&self, val: u64) -> bool {
        match (val, self.0) {
            (0, _) | (_, None) => true,
            (val, Some(inner)) if NonZeroU64::new(val).unwrap() < inner => true,
            _ => false,
        }
    }
}

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type].
    Type(TypeBound),
    /// Argument is a [TypeArg::BoundedNat] that is less than the upper bound.
    BoundedNat(UpperBound),
    /// Argument is a [TypeArg::Opaque], defined by a [CustomType].
    Opaque(CustomType),
    /// Argument is a [TypeArg::Sequence]. A list of indeterminate size containing parameters.
    List(Box<TypeParam>),
    /// Argument is a [TypeArg::Sequence]. A tuple of parameters.
    Tuple(Vec<TypeParam>),
    /// Argument is a [TypeArg::Extensions]. A set of [ExtensionId]s.
    ///
    /// [ExtensionId]: crate::extension::ExtensionId
    Extensions,
}

impl TypeParam {
    /// [`TypeParam::BoundedNat`] with the maximum bound (`u64::MAX` + 1)
    pub const fn max_nat() -> Self {
        Self::BoundedNat(UpperBound(None))
    }

    /// [`TypeParam::BoundedNat`] with the stated upper bound (non-exclusive)
    pub const fn bounded_nat(upper_bound: NonZeroU64) -> Self {
        Self::BoundedNat(UpperBound(Some(upper_bound)))
    }
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(Type),
    /// Instance of [TypeParam::BoundedNat]. 64-bit unsigned integer.
    BoundedNat(u64),
    ///Instance of [TypeParam::Opaque] An opaque value, stored as serialized blob.
    Opaque(CustomTypeArg),
    /// Instance of [TypeParam::List] or [TypeParam::Tuple], defined by a
    /// sequence of arguments.
    Sequence(Vec<TypeArg>),
    /// Instance of [TypeParam::Extensions], providing the extension ids.
    Extensions(ExtensionSet),
}

/// A serialized representation of a value of a [CustomType]
/// restricted to equatable types.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustomTypeArg {
    /// The type of the constant.
    /// (Exact matches only - the constant is exactly this type.)
    pub typ: CustomType,
    /// Serialized representation.
    pub value: serde_yaml::Value,
}

impl CustomTypeArg {
    /// Create a new CustomTypeArg. Enforces that the type must be checkable for
    /// equality.
    pub fn new(typ: CustomType, value: serde_yaml::Value) -> Result<Self, &'static str> {
        if typ.bound() == TypeBound::Eq {
            Ok(Self { typ, value })
        } else {
            Err("Only TypeBound::Eq CustomTypes can be used as TypeArgs")
        }
    }
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    match (arg, param) {
        (TypeArg::Type(t), TypeParam::Type(bound)) if bound.contains(t.least_upper_bound()) => {
            Ok(())
        }
        (TypeArg::Sequence(items), TypeParam::List(param)) => {
            items.iter().try_for_each(|arg| check_type_arg(arg, param))
        }
        (TypeArg::Sequence(items), TypeParam::Tuple(types)) => {
            if items.len() != types.len() {
                Err(TypeArgError::WrongNumberTuple(items.len(), types.len()))
            } else {
                items
                    .iter()
                    .zip(types.iter())
                    .try_for_each(|(arg, param)| check_type_arg(arg, param))
            }
        }
        (TypeArg::BoundedNat(val), TypeParam::BoundedNat(bound)) if bound.valid_value(*val) => {
            Ok(())
        }

        (TypeArg::Opaque(arg), TypeParam::Opaque(param))
            if param.bound() == TypeBound::Eq && &arg.typ == param =>
        {
            Ok(())
        }
        (TypeArg::Extensions(_), TypeParam::Extensions) => Ok(()),
        _ => Err(TypeArgError::TypeMismatch {
            arg: arg.clone(),
            param: param.clone(),
        }),
    }
}

/// Errors that can occur fitting a [TypeArg] into a [TypeParam]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeArgError {
    #[allow(missing_docs)]
    /// For now, general case of a type arg not fitting a param.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Type argument {arg:?} does not fit declared parameter {param:?}")]
    TypeMismatch { param: TypeParam, arg: TypeArg },
    /// Wrong number of type arguments (actual vs expected).
    // For now this only happens at the top level (TypeArgs of op/type vs TypeParams of Op/TypeDef).
    // However in the future it may be applicable to e.g. contents of Tuples too.
    #[error("Wrong number of type arguments: {0} vs expected {1} declared type parameters")]
    WrongNumberArgs(usize, usize),

    /// Wrong number of type arguments in tuple (actual vs expected).
    #[error("Wrong number of type arguments to tuple parameter: {0} vs expected {1} declared type parameters")]
    WrongNumberTuple(usize, usize),
    /// Opaque value type check error.
    #[error("Opaque type argument does not fit declared parameter type: {0:?}")]
    OpaqueTypeMismatch(#[from] crate::types::CustomCheckFailure),
    /// Invalid value
    #[error("Invalid value of type argument")]
    InvalidValue(TypeArg),
}
