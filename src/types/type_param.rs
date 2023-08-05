//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::values::{HashableLeaf, ValueError, ValueOfType};

use super::simple::{HashableElem, Tagged};
use super::CustomType;
use super::{ClassicType, HashableType, SimpleType, TypeTag};

/// A parameter declared by an OpDef. Specifies a value
/// that must be provided by each operation node.
// TODO any other 'leaf' types? We specifically do not want float.
// bool should eventually be a Sum type (Container).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type] - classic or linear
    Type,
    /// Argument is a [TypeArg::ClassicType] - hashable or otherwise
    ClassicType,
    /// Argument is a [TypeArg::HashableType]
    HashableType,
    /// Node must provide a [TypeArg::List] (of whatever length)
    /// TODO it'd be better to use [`Container`] here, or a variant thereof
    /// (plus List minus Array).
    ///
    /// [`Container`]: crate::types::simple::Container
    List(Box<TypeParam>),
    /// Equivalent to [Container::Custom] (since we are not using [Container] here)
    CustomValue(CustomType),
    /// Argument is a value of the specified type. Note we do not use
    /// HashableValue here, so this loses some expressivity, especially
    /// until we have more containers above.
    Value(HashableElem),
}

impl From<HashableElem> for TypeParam {
    fn from(value: HashableElem) -> Self {
        Self::Value(value)
    }
}

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
    /// Where the (Type/Op)Def declares a [TypeParam::Value], an appropriate constant
    Value(HashableLeaf),
    /// Where the (Type/Op)Def declares a [TypeParam::List]`<T>` - all elements will implicitly
    /// be of the same variety of TypeArg, i.e. `T`s.
    List(Vec<TypeArg>),
    /// Where the TypeDef declares a [TypeParam::CustomValue]
    CustomValue(serde_yaml::Value),
}

impl From<HashableLeaf> for TypeArg {
    fn from(value: HashableLeaf) -> Self {
        Self::Value(value)
    }
}

pub type TypeArgError = ValueError<TypeArg>;

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

impl ValueOfType for TypeArg {
    type T = TypeParam;

    fn name(&self) -> String {
        todo!()
    }

    /// Checks a [TypeArg] is as expected for a [TypeParam]
    fn check_type(&self, param: &TypeParam) -> Result<(), TypeArgError> {
        match (self, param) {
            (TypeArg::Type(_), TypeParam::Type) => Ok(()),
            (TypeArg::ClassicType(_), TypeParam::ClassicType) => Ok(()),
            (TypeArg::HashableType(_), TypeParam::HashableType) => Ok(()),
            (TypeArg::List(items), TypeParam::List(ty)) => {
                for item in items {
                    item.check_type(ty.as_ref())?;
                }
                Ok(())
            }
            (TypeArg::CustomValue(v), TypeParam::CustomValue(ct)) => Ok(()), // TODO more checks here, e.g. storing CustomType in the value
            (TypeArg::Value(h_v), TypeParam::Value(h_t)) => h_v.check_type(h_t).map_err(|e|e.map_into()),
            (_, _) => Err(TypeArgError::ValueCheckFail(param.clone(), self.clone())),
        }
    }
}
