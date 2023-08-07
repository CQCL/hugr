//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use thiserror::Error;

use crate::ops::constant::{CustomConst, CustomSerialized};

use super::CustomType;
use super::{ClassicType, PrimType, SimpleType, TypeTag};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type].
    Type(TypeTag),
    /// Argument is a [TypeArg::USize].
    USize,
    /// Argument is a [TypeArg::Opaque], defined by a [CustomType].
    Opaque(CustomType),
    /// Argument is a [TypeArg::Sequence]. A list of indeterminate size containing parameters.
    List(Box<TypeParam>),
    /// Argument is a [TypeArg::Sequence]. A tuple of parameters.
    Tuple(Vec<TypeParam>),
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Instance of [TypeParam::USize]. 64-bit unsigned integer.
    USize(u64),
    ///Instance of [TypeParam::Opaque] An opaque value, stored as serialized blob.
    Opaque(CustomSerialized),
    /// Instance of [TypeParam::List] or [TypeParam::Tuple], defined by a
    /// sequence of arguments.
    Sequence(Vec<TypeArg>),
}

impl TypeArg {
    /// Report [`TypeTag`] if param is a type
    pub fn tag_of_type(&self) -> Option<TypeTag> {
        match self {
            TypeArg::Type(s) => Some(s.tag()),
            _ => None,
        }
    }
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    match (arg, param) {
        (
            TypeArg::Type(SimpleType::Classic(ClassicType::Hashable(_))),
            TypeParam::Type(TypeTag::Hashable),
        ) => Ok(()),
        (TypeArg::Type(SimpleType::Classic(_)), TypeParam::Type(TypeTag::Classic)) => Ok(()),
        (TypeArg::Type(_), TypeParam::Type(TypeTag::Simple)) => Ok(()),
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
        (TypeArg::USize(_), TypeParam::USize) => Ok(()),
        (TypeArg::Opaque(arg), TypeParam::Opaque(param)) => {
            arg.check_custom_type(param)?;
            Ok(())
        }
        _ => Err(TypeArgError::TypeMismatch(arg.clone(), param.clone())),
    }
}

/// Errors that can occur fitting a [TypeArg] into a [TypeParam]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeArgError {
    /// For now, general case of a type arg not fitting a param.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Type argument {0:?} does not fit declared parameter {1:?}")]
    TypeMismatch(TypeArg, TypeParam),
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
    OpaqueTypeMismatch(#[from] crate::values::CustomCheckFail),
}
