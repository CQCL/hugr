//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::values::{map_container_type, ConstTypeError};
use crate::values::{ContainerValue, HashableValue, ValueOfType};

use super::{
    simple::{Container, HashableType, PrimType},
    ClassicType, SimpleType, TypeTag,
};
use super::{CustomType, TypeRow};

/// A parameter declared by an [OpDef] - specifying an argument
/// that must be provided by each operation node - or by a [TypeDef]
/// - specifying an argument that must be provided to make a type.
///
/// [OpDef]: crate::resource::OpDef
/// [TypeDef]: crate::resource::TypeDef
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum TypeParam {
    /// Argument is a [TypeArg::Type] - classic or linear
    Type,
    /// Argument is a [TypeArg::ClassicType] - hashable or otherwise
    ClassicType,
    /// Argument is a [TypeArg::HashableType]
    HashableType,
    /// A nested definition containing other TypeParams (possibly including other [HashableType]s).
    /// Note that if all components are [TypeParam::Value]s, then the entire [Container] should be stored
    /// inside a [TypeParam::Value] instead; so there will never be a [Container::Opaque] here,
    /// as there are no [CustomType]s corresponding to TypeParam's except Hashable [TypeParam::Value]s.
    Container(Container<TypeParam>),
    /// Argument is a value of the specified type.
    Value(HashableType),
}

/// A trait just to restrict serde's (de)serialization of [Container].
pub(super) trait TypeParamMarker: serde::Serialize + for<'a> serde::Deserialize<'a> {}

impl TypeParamMarker for TypeParam {}

impl TypeParam {
    fn value_types(
        typarams: TypeRow<TypeParam>,
    ) -> Result<TypeRow<HashableType>, TypeRow<TypeParam>> {
        if typarams.iter().all(|e| matches!(e, TypeParam::Value(_))) {
            Ok(typarams
                .into_owned()
                .into_iter()
                .map(|e| match e {
                    TypeParam::Value(ht) => ht,
                    _ => panic!(), // We checked all matched above
                })
                .collect::<Vec<_>>()
                .into())
        } else {
            Err(typarams)
        }
    }

    /// New Tuple TypeParam, elements defined by TypeRow
    pub fn new_tuple(elems: impl Into<TypeRow<TypeParam>>) -> Self {
        match TypeParam::value_types(elems.into()) {
            Ok(h_tys) => {
                TypeParam::Value(HashableType::Container(Container::Tuple(Box::new(h_tys))))
            }
            Err(ty_params) => Self::Container(Container::Tuple(Box::new(ty_params))),
        }
    }

    /// New Tuple typeparam, elements defined by TypeRow
    pub fn new_sum(elems: impl Into<TypeRow<TypeParam>>) -> Self {
        match TypeParam::value_types(elems.into()) {
            Ok(h_tys) => TypeParam::Value(HashableType::Container(Container::Sum(Box::new(h_tys)))),
            Err(ty_params) => Self::Container(Container::Sum(Box::new(ty_params))),
        }
    }

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
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::ClassicType],
    /// it'll get one of these (rather than embedding inside a Type)
    ClassicType(ClassicType),
    /// Where the (Type/Op)Def declares that an argument is a [TypeParam::HashableType],
    /// this is the value.
    HashableType(HashableType),
    /// Where the (Type/Op)Def declares a [TypeParam::Container], this is the value
    /// (unless the param is a [Container::Opaque] - that'll be given as a [TypeArg::CustomValue])
    Container(ContainerValue<TypeArg>),
    /// Where the (Type/Op)Def declares a [TypeParam::Value], the corresponding value
    Value(HashableValue),
    /// Where the TypeDef declares a [TypeParam::Container] of [Container::Opaque]
    CustomValue(CustomTypeArg),
}

impl TypeArg {
    /// Report [`TypeTag`] if param is a type
    pub fn tag_of_type(&self) -> Option<TypeTag> {
        match self {
            TypeArg::Type(s) => Some(s.tag()),
            TypeArg::ClassicType(c) => Some(c.tag()),
            TypeArg::HashableType(h) => Some(h.tag()),
            _ => None,
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
            (TypeArg::Type(_), TypeParam::Type) => Ok(()),
            (TypeArg::ClassicType(_), TypeParam::ClassicType) => Ok(()),
            (TypeArg::HashableType(_), TypeParam::HashableType) => Ok(()),
            (TypeArg::Container(vals), TypeParam::Container(c_ty)) => vals.check_container(c_ty),
            // The argument might not be a HashableValue because it contains [TypeArg::Opaque]:
            (TypeArg::Container(vals), TypeParam::Value(HashableType::Container(c_ty))) => {
                vals.check_container(&map_container_type(c_ty, &TypeParam::Value))
            }
            (TypeArg::Value(hv), TypeParam::Value(ht)) => hv.check_type(ht),
            // A "hashable" value might be argument to a non-hashable TypeParam:
            // e.g. an empty list is a HashableValue, yet can be checked against a List<TypeParam::Type>.
            (TypeArg::Value(HashableValue::Container(vals)), TypeParam::Container(c_ty)) => {
                vals.map_vals(&TypeArg::Value).check_container(c_ty)
            }
            (
                TypeArg::CustomValue(cv),
                TypeParam::Value(HashableType::Container(Container::Opaque(ct))),
            ) if &cv.typ == ct => Ok(()),
            _ => Err(ConstTypeError::TypeArgCheckFail(ty.clone(), self.clone())),
        }
    }

    fn name(&self) -> String {
        match self {
            TypeArg::Type(s) => format!("type:{}", s),
            TypeArg::ClassicType(c) => format!("ctype:{}", c),
            TypeArg::HashableType(h) => format!("htype:{}", h),
            TypeArg::Container(ctr) => ctr.desc(),
            TypeArg::Value(hv) => hv.name(),
            TypeArg::CustomValue(cs) => format!("yaml:{:?}", cs.value),
        }
    }

    fn container_error(typ: Container<Self::T>, vals: ContainerValue<Self>) -> ConstTypeError {
        ConstTypeError::TypeArgCheckFail(TypeParam::Container(typ), TypeArg::Container(vals))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        types::{CustomType, TypeTag},
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
        TypeArg::CustomValue(c_arg).check_type(&c_param).unwrap();
        let c_arg2 = CustomTypeArg::new(
            CustomType::new("MyType2", [], "MyRsrc", TypeTag::Hashable),
            serde_yaml::Value::Number(5.into()),
        )
        .unwrap();
        assert_matches!(
            TypeArg::CustomValue(c_arg2).check_type(&c_param),
            Err(ConstTypeError::TypeArgCheckFail(_, _))
        );
    }
}
