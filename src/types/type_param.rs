//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::values::ConstTypeError;
use crate::values::{ContainerValue, HashableValue, ValueOfType};

use super::CustomType;
use super::{
    simple::{Container, HashableType, PrimType},
    ClassicType, SimpleType, TypeTag,
};

/// A parameter declared by an [OpDef] - specifying an argument
/// that must be provided by each operation node - or by a [TypeDef]
/// - specifying an argument that must be provided to make a type.
///
/// [OpDef]: crate::resource::OpDef
/// [TypeDef]: crate::resource::TypeDef
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// A type as parameter, with given type tag.
    Type(TypeTag),
    /// A value of a hashable type.
    Value(HashableType),
    /// List of types.
    TypeList,
    /// List of Values.
    ValueList,
}

impl TypeParam {
    /// Creates a new TypeParam accepting values of a specified CustomType, which must be hashable.
    pub fn new_opaque(ct: CustomType) -> Result<Self, &'static str> {
        if ct.tag() == TypeTag::Hashable {
            Ok(Self::Value(HashableType::Container(Container::Opaque(ct))))
        } else {
            Err("CustomType not hashable")
        }
    }
}
/// A statically-known argument value to an operation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeArg {
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::Type]
    Type(SimpleType),
    /// List of types of arbitrary length
    TypeList(Vec<SimpleType>),
    /// List of values of arbitrary length
    ValueList(Vec<ArgValue>),
    /// Where the (Type/Op)Def declares a [TypeParam::Value], the corresponding value
    Value(ArgValue),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
/// Type argument specified as a value
pub enum ArgValue {
    /// Hashable value.
    Hashable(HashableValue),
    /// Opaque serialised custom value.
    Custom(CustomTypeArg),
}
impl TypeArg {
    /// Report [`TypeTag`] if param is a type
    pub fn tag_of_type(&self) -> Option<TypeTag> {
        if let TypeArg::Type(s) = self {
            Some(s.tag())
        } else {
            None
        }
    }
}

/// A serialized representation of a value of a [CustomType]
/// restricted to Hashable types.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CustomTypeArg {
    /// The type of the constant.
    /// (Exact matches only - the constant is exactly this type.)
    typ: CustomType,
    /// Serialized representation.
    pub value: serde_yaml::Value,
}

impl CustomTypeArg {
    /// Create a new CustomTypeArg. Enforces that the type must be Hashable.
    pub fn new(typ: CustomType, value: serde_yaml::Value) -> Result<Self, &'static str> {
        if typ.tag() == TypeTag::Hashable {
            Ok(Self { typ, value })
        } else {
            Err("Only Hashable CustomTypes can be used as TypeArgs")
        }
    }
}

impl ValueOfType for TypeArg {
    type T = TypeParam;

    fn check_type(&self, ty: &TypeParam) -> Result<(), ConstTypeError> {
        match (self, ty) {
            (
                TypeArg::Type(SimpleType::Classic(ClassicType::Hashable(_))),
                TypeParam::Type(TypeTag::Hashable),
            ) => Ok(()),
            (TypeArg::Type(SimpleType::Classic(_)), TypeParam::Type(TypeTag::Classic)) => Ok(()),
            (TypeArg::Type(_), TypeParam::Type(TypeTag::Simple)) => Ok(()),

            (TypeArg::TypeList(_), TypeParam::TypeList) => Ok(()),
            (TypeArg::ValueList(_), TypeParam::ValueList) => Ok(()),

            (TypeArg::Value(ArgValue::Hashable(hv)), TypeParam::Value(ht)) => hv.check_type(ht),
            (
                TypeArg::Value(ArgValue::Custom(cv)),
                TypeParam::Value(HashableType::Container(Container::Opaque(ct))),
            ) if &cv.typ == ct => Ok(()),
            _ => Err(ConstTypeError::TypeArgCheckFail(ty.clone(), self.clone())),
        }
    }

    fn name(&self) -> String {
        match self {
            TypeArg::Type(s) => format!("type:{}", s),
            TypeArg::TypeList(_) => "type_list".into(),
            TypeArg::ValueList(_) => "value_list".into(),
            TypeArg::Value(ArgValue::Hashable(hv)) => hv.name(),
            TypeArg::Value(ArgValue::Custom(cs)) => format!("yaml:{:?}", cs.value),
        }
    }

    fn container_error(_typ: Container<Self::T>, _vals: ContainerValue<Self>) -> ConstTypeError {
        unimplemented!("Shouldn't be called")
    }
}

#[cfg(test)]
mod test {
    use crate::{
        types::{type_param::ArgValue, CustomType, TypeTag},
        values::{ConstTypeError, ValueOfType},
    };
    use cool_asserts::assert_matches;

    use super::{CustomTypeArg, TypeArg, TypeParam};

    #[test]
    fn test_check_custom_type_arg() {
        let ct = CustomType::new("MyType", [], "MyRsrc", TypeTag::Hashable);
        let c_arg =
            CustomTypeArg::new(ct.clone(), serde_yaml::Value::String("foo".into())).unwrap();
        let c_param = TypeParam::new_opaque(ct).unwrap();
        TypeArg::Value(ArgValue::Custom(c_arg))
            .check_type(&c_param)
            .unwrap();
        let c_arg2 = CustomTypeArg::new(
            CustomType::new("MyType2", [], "MyRsrc", TypeTag::Hashable),
            serde_yaml::Value::Number(5.into()),
        )
        .unwrap();
        assert_matches!(
            TypeArg::Value(ArgValue::Custom(c_arg2)).check_type(&c_param),
            Err(ConstTypeError::TypeArgCheckFail(_, _))
        );
    }
}
