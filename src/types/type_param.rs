//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use thiserror::Error;

use crate::ops::constant::HugrIntValueStore;

use super::{ClassicType, SimpleType};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
// TODO any other 'leaf' types? We specifically do not want float.
// bool should eventually be a Sum type (Container).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type] - classic or linear
    Type,
    /// Argument is a [TypeArg::ClassicType]
    ClassicType,
    /// Argument is an integer
    Int,
    /// Node must provide a [TypeArg::List] (of whatever length)
    /// TODO it'd be better to use [`Container`] here.
    ///
    /// [`Container`]: crate::types::simple::Container
    List(Box<TypeParam>),
    /// Argument is a [TypeArg::Value], containing a yaml-encoded object
    /// interpretable by the operation.
    Value,
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the TypeDef declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Where the TypeDef declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the TypeDef declares a [TypeParam::Int]
    Int(HugrIntValueStore),
    /// Where an argument has type [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArg, representing a `T`.
    List(Vec<TypeArg>),
    /// Where the TypeDef declares a [TypeParam::Value]
    Value(serde_yaml::Value),
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    match (arg, param) {
        (TypeArg::Type(_), TypeParam::Type) => (),
        (TypeArg::ClassicType(_), TypeParam::ClassicType) => (),
        (TypeArg::Int(_), TypeParam::Int) => (),
        (TypeArg::List(items), TypeParam::List(ty)) => {
            for item in items {
                check_type_arg(item, ty.as_ref())?;
            }
        }
        (TypeArg::Value(_), TypeParam::Value) => (),
        _ => {
            return Err(TypeArgError::TypeMismatch(arg.clone(), param.clone()));
        }
    };
    Ok(())
}

/// Errors that can occur fitting a [TypeArg] into a [TypeParam]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TypeArgError {
    /// For now, general case of a type arg not fitting a param.
    /// We'll have more cases when we allow general Containers.
    // TODO It may become possible to combine this with ConstTypeError.
    #[error("Type argument {0:?} does not fit declared parameter {1:?}")]
    TypeMismatch(TypeArg, TypeParam),
    /// Wrong number of type arguments.
    // For now this only happens at the top level (TypeArgs of Op vs TypeParams of OpDef).
    // However in the future it may be applicable to e.g. contents of Tuples too.
    #[error("Wrong number of type arguments: {0} vs expected {1} declared type parameters")]
    WrongNumber(usize, usize),
}
