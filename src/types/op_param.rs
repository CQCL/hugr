//! Type Parameters
//!
//! Parameters for [`OpDef`]s provided by extensions
//!
//! [`OpDef`]: crate::resource::OpDef

use crate::{ops::ConstValue, resource::ResourceSet};

use super::{ClassicType, SimpleType};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum OpParam {
    /// Node must provide a [OpArg::Type] - classic or linear
    Type,
    /// Node must provide a [OpArg::ClassicType]
    ClassicType,
    /// Node must provide a [OpArg::ResourceSet]. For example,
    /// a definition of an operation that takes a Graph argument,
    /// could be polymorphic over the ResourceSet of that graph,
    /// in order to encode that ResourceSet in its [`output_resources`]
    ///
    /// [`output_resources`]: crate::types::Signature::output_resources
    ResourceSet,
    /// Node must provide a value of the specified [OpArg::ClassicType]
    Value(ClassicType),
    /// Node must provide a [OpArg::List] (of whatever length)
    /// TODO it'd be better to use [`Container`], and to exclude containerized
    /// classictypes from [OpParam::Value].
    ///
    /// [`Container`]: crate::types::simple::Container
    List(Box<OpParam>),
}

/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OpArg {
    /// Where the OpDef declares that an argument is a [OpParam::Type]
    Type(SimpleType),
    /// Where the OpDef declares that an argument is a [OpParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the OpDef is polymorphic over a [OpParam::ResourceSet]
    ResourceSet(ResourceSet),
    /// Where the OpDef is polymorphic over a [OpParam::Value] (`t`); the value's
    /// [ConstValue::const_type] will be equal to the ClassicType `t`.
    Value(ConstValue),
    /// Where an argument has type [OpParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of OpArg, representing a `T`.
    List(Vec<OpArg>),
}

/// Checks a [OpArg] is as expected for a [OpParam]
pub fn check_arg(arg: &OpArg, param: &OpParam) -> Result<(), String> {
    match (arg, param) {
        (OpArg::Type(_), OpParam::Type) => (),
        (OpArg::ClassicType(_), OpParam::ClassicType) => (),
        (OpArg::ResourceSet(_), OpParam::ResourceSet) => (),
        (OpArg::Value(cst), OpParam::Value(ty)) if cst.const_type() == *ty => (),
        (OpArg::List(items), OpParam::List(ty)) => {
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
