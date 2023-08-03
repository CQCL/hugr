//! Type Parameters
//!
//! Parameters for [`TypeDef`]s provided by extensions
//!
//! [`TypeDef`]: crate::resource::TypeDef

use crate::values::{map_container_type, ConstTypeError};
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
    /// A nested definition containing other TypeParams (possibly including other [HashableType]s).
    /// Note that if all components are [TypeParam::Value]s, then the entire [Container] should be stored
    /// inside a [TypeParam::Value] instead.
    Container(Container<TypeParam>),
    /// Argument is a value of the specified type.
    Value(HashableType),
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
        match self {
            TypeArg::Type(_) => {
                if ty == &TypeParam::Type {
                    return Ok(());
                }
            }
            TypeArg::ClassicType(_) => {
                if ty == &TypeParam::ClassicType {
                    return Ok(());
                }
            }
            TypeArg::HashableType(_) => {
                if ty == &TypeParam::HashableType {
                    return Ok(());
                }
            }
            TypeArg::Container(vals) => {
                match ty {
                    TypeParam::Container(c_ty) => return vals.check_container(c_ty),
                    // We might have an argument *value* that is a TypeArg (but not a HashableValue)
                    // that fits a Hashable type because the argument contains an [TypeArg::Opaque].
                    TypeParam::Value(HashableType::Container(c_ty)) => {
                        return vals.check_container(&map_container_type(c_ty, &TypeParam::Value))
                    }
                    _ => (),
                };
            }
            TypeArg::Value(hv) => match ty {
                TypeParam::Value(ht) => return hv.check_type(ht),
                TypeParam::Container(c_ty) => {
                    // A "hashable" value might be argument to a non-hashable TypeParam:
                    // e.g. an empty list is hashable, yet can be checked against a List<SimpleType>.
                    if let HashableValue::Container(vals) = hv {
                        return vals.map_vals(&TypeArg::Value).check_container(c_ty);
                    }
                }
                _ => (),
            },
            TypeArg::CustomValue(cv) => {
                let maybe_ct = match ty {
                    TypeParam::Container(Container::Opaque(c)) => Some(c),
                    TypeParam::Value(HashableType::Container(Container::Opaque(c))) => Some(c),
                    _ => None,
                };
                if let Some(ct) = maybe_ct {
                    if &cv.typ == ct {
                        return Ok(());
                    }
                }
            }
        };
        Err(ConstTypeError::TypeArgCheckFail(ty.clone(), self.clone()))
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
