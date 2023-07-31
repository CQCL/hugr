//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use thiserror::Error;

use crate::ops::constant::typecheck::ConstIntError;
use crate::ops::constant::HugrIntValueStore;

use super::{
    simple::{Container, PrimType, TypeRowElem},
    ClassicType, HashableType, SimpleType, TypeTag,
};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
// TODO any other 'leaf' types? We specifically do not want float.
// bool should eventually be a Sum type (Container).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(
    try_from = "super::serialize::SerSimpleType",
    into = "super::serialize::SerSimpleType"
)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type] - classic or linear
    Type,
    /// Argument is a [TypeArg::ClassicType] - hashable or otherwise
    ClassicType,
    /// Argument is a [TypeArg::HashableType]
    HashableType,
    /// Argument is an instance of a [Container] type (not an alias).
    /// Values will be of the corresponding variety of TypeArg.
    Container(Container<TypeParam>),
    /// Argument is a value of the specified type.
    Value(HashableType),
}

impl TypeRowElem for TypeParam {}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::HashableType],
    /// this is the value.
    HashableType(HashableType),
    /// Where the (Type/Op)Def declares a [TypeParam::Value] of type [HashableType::Int], a constant value thereof
    Int(HugrIntValueStore),
    /// Where the (Type/Op)Def declares a [TypeParam::Value] of type [HashableType::String], here it is
    String(String),
    /// Where the (Type/Op)Def declares a [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArg, i.e. `T`s.
    List(Vec<TypeArg>),
    /// Where the TypeDef declares a [TypeParam::Value] of [Container::Opaque]
    CustomValue(serde_yaml::Value),
}

impl TypeArg {
    /// Report [`TypeTag`] if param is a type
    pub fn tag_of_type(&self) -> Option<TypeTag> {
        match self {
            TypeArg::Type(s) => Some(s.tag()),
            TypeArg::ClassicType(c) => Some(c.tag()),
            _ => None,
        }
    }
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_type_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), TypeArgError> {
    let _ = arg;
    let _ = param;
    todo!();
    /* Reinstate following checkable consts etc.
    match (arg, param) {
        (TypeArg::Type(_), TypeParam::Type) => Ok(()),
        (TypeArg::ClassicType(_), TypeParam::ClassicType) => Ok(()),
        (TypeArg::HashableType(_), TypeParam::HashableType) => Ok(()),
        (TypeArg::List(items), TypeParam::List(ty)) => {
            for item in items {
                check_type_arg(item, ty.as_ref())?;
            }
            Ok(())
        }
        (TypeArg::Int(v), TypeParam::Value(HashableType::Int(width))) => {
            check_int_fits_in_width(*v, *width).map_err(TypeArgError::Int)
        }
        (TypeArg::String(_), TypeParam::Value(HashableType::String)) => Ok(()),
        (arg, TypeParam::Value(HashableType::Container(ctr))) => match ctr {
            Container::Opaque(_) => match arg {
                TypeArg::CustomValue(_) => Ok(()), // Are there more checks we should do here?
                _ => Err(TypeArgError::TypeMismatch(arg.clone(), param.clone())),
            },
            Container::List(elem) => check_type_arg(
                arg,
                &TypeParam::List(Box::new(TypeParam::Value((**elem).clone()))),
            ),
            Container::Map(_) => unimplemented!(),
            Container::Tuple(_) => unimplemented!(),
            Container::Sum(_) => unimplemented!(),
            Container::Array(elem, sz) => {
                let TypeArg::List(items) = arg else {return Err(TypeArgError::TypeMismatch(arg.clone(), param.clone()))};
                if items.len() != *sz {
                    return Err(TypeArgError::WrongNumber(items.len(), *sz));
                }
                check_type_arg(
                    arg,
                    &TypeParam::List(Box::new(TypeParam::Value((**elem).clone()))),
                )
            }
            Container::Alias(n) => Err(TypeArgError::NoAliases(n.to_string())),
        },
        _ => Err(TypeArgError::TypeMismatch(arg.clone(), param.clone())),
    }*/
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
    /// The type declared for a TypeParam was an alias that was not resolved to an actual type
    #[error("TypeParam required an unidentified alias type {0}")]
    NoAliases(String),
    /// There was some problem fitting a const int into its declared size
    #[error("Error with int constant")]
    Int(#[from] ConstIntError),
}
