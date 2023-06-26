//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::ops::constant::HugrIntValueStore;

use super::{ClassicType, SimpleType};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
/// TODO any other 'leaf' types? We specifically do not want float.
/// bool should eventually be a Sum type (Container).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
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
}

/// Checks a [TypeArg] is as expected for a [TypeParam]
pub fn check_arg(arg: &TypeArg, param: &TypeParam) -> Result<(), String> {
    match (arg, param) {
        (TypeArg::Type(_), TypeParam::Type) => (),
        (TypeArg::ClassicType(_), TypeParam::ClassicType) => (),
        (TypeArg::Int(_), TypeParam::Int) => (),
        (TypeArg::List(items), TypeParam::List(ty)) => {
            for item in items {
                check_arg(item, ty.as_ref())?;
            }
        }
        _ => {
            return Err(format!("Mismatched {:?} vs {:?}", arg, param));
        }
    };
    Ok(())
}
