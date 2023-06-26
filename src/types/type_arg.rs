//! Type Parameters
//!
//! Parameters for [`OpDef`]s provided by extensions
//!
//! [`OpDef`]: crate::resource::OpDef

use crate::{ops::ConstValue, resource::ResourceSet};

use super::{ClassicType, SimpleType};

/// A Type Parameter declared by an OpDef. Specifies
/// the values that must be provided by each operation node.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum TypeParam {
    /// Node must provide a [TypeArgValue::Type] - classic or linear
    Type,
    /// Node must provide a [TypeArgValue::ClassicType]
    ClassicType,
    /// Node must provide a [TypeArgValue::ResourceSet]. For example,
    /// a definition of an operation that takes a Graph argument,
    /// could be polymorphic over the ResourceSet of that graph,
    /// in order to encode that ResourceSet in its [`output_resources`]
    ///
    /// [`output_resources`]: crate::types::Signature::output_resources
    ResourceSet,
    /// Node must provide a value of the specified [TypeArgValue::ClassicType]
    Value(ClassicType),
    /// Node must provide a [TypeArgValue::List] (of whatever length)
    List(Box<TypeParam>),
}

/// An argument value for a type parameter
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum TypeArgValue {
    /// Where the OpDef declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// Where the OpDef declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the OpDef is polymorphic over a [TypeParam::ResourceSet]
    ResourceSet(ResourceSet),
    /// Where the OpDef is polymorphic over a [TypeParam::Value](`t`); the value's
    /// [ConstValue::const_type] will be equal to the ClassicType `t`.
    Value(ConstValue),
    /// Where an argument has type [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArgValue, representing a `T`.
    List(Vec<TypeArgValue>),
}

/// Checks a [TypeArgValue] is as expected for a [TypeParam]
pub fn check_arg(arg: &TypeArgValue, param: &TypeParam) -> Result<(), String> {
    match (arg, param) {
        (TypeArgValue::Type(_), TypeParam::Type) => (),
        (TypeArgValue::ClassicType(_), TypeParam::ClassicType) => (),
        (TypeArgValue::ResourceSet(_), TypeParam::ResourceSet) => (),
        (TypeArgValue::Value(cst), TypeParam::Value(ty)) if cst.const_type() == *ty => (),
        (TypeArgValue::List(items), TypeParam::List(ty)) => {
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
