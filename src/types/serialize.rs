use serde_repr::{Deserialize_repr, Serialize_repr};
use smol_str::SmolStr;

use super::custom::CustomType;

use super::type_param::TypeParam;
use super::type_row::TypeRowElem;
use super::{
    AbstractSignature, ClassicType, Container, HashableType, SimpleType, TypeRow, TypeTag,
};

use crate::ops::constant::HugrIntWidthStore;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub(crate) enum SerializableTag {
    Simple = 0,
    Classic = 1,
    Hashable = 2,
    TypeParam = 3,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
#[serde(tag = "t")]
pub(crate) enum SerSimpleType {
    Q,
    I {
        width: HugrIntWidthStore,
    },
    F,
    S,
    G {
        signature: Box<AbstractSignature>,
    },
    List {
        inner: Box<SerSimpleType>,
        c: SerializableTag,
    },
    Map {
        k: Box<SerSimpleType>,
        v: Box<SerSimpleType>,
        c: SerializableTag,
    },
    Tuple {
        row: Box<TypeRow<SerSimpleType>>,
        c: SerializableTag,
    },
    Sum {
        row: Box<TypeRow<SerSimpleType>>,
        c: SerializableTag,
    },
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
        c: SerializableTag,
    },
    Opaque {
        custom: CustomType,
        c: SerializableTag,
    },
    Alias {
        name: SmolStr,
        c: TypeTag, // not a SerializableTag - there are no TypeParam aliases
    },
    Var {
        name: SmolStr,
    },
    /// For TypeParams only - corresponds to [TypeParam::SimpleType]
    ST,
    /// For TypeParams only - corresponds to [TypeParam::ClassicType]
    CT,
    /// For TypeParams only - corresponds to [TypeParam::HashableType]
    HT,
}

trait SerializableType: TypeRowElem {
    const TAG: SerializableTag;
}

impl SerializableType for ClassicType {
    const TAG: SerializableTag = SerializableTag::Classic;
}

impl SerializableType for SimpleType {
    const TAG: SerializableTag = SerializableTag::Simple;
}

impl SerializableType for HashableType {
    const TAG: SerializableTag = SerializableTag::Hashable;
}

impl SerializableType for TypeParam {
    const TAG: SerializableTag = SerializableTag::TypeParam;
}

enum Deserialized {
    Simple(SimpleType),
    Classic(ClassicType),
    Hashable(HashableType),
    TypeParam(TypeParam),
}

impl<T: SerializableType> From<Container<T>> for SerSimpleType
where
    SerSimpleType: From<T>,
{
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Sum(inner) => SerSimpleType::Sum {
                row: Box::new(inner.map_into()),
                c: T::TAG, // We could inspect inner.containing_tag(), but this should have been done already
            },
            Container::List(inner) => SerSimpleType::List {
                inner: Box::new((*inner).into()),
                c: T::TAG, // We could inspect inner.tag(), but this should have been done already
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: Box::new(inner.map_into()),
                c: T::TAG,
            },
            Container::Map(inner) => SerSimpleType::Map {
                k: Box::new(inner.0.into()),
                v: Box::new(inner.1.into()),
                c: T::TAG,
            },
            Container::Array(inner, len) => SerSimpleType::Array {
                inner: box_convert(*inner),
                len,
                c: T::TAG,
            },
            Container::Alias(name) => {
                let c = match T::TAG {
                    SerializableTag::Simple => TypeTag::Simple,
                    SerializableTag::Classic => TypeTag::Classic,
                    SerializableTag::Hashable => TypeTag::Hashable,
                    SerializableTag::TypeParam => panic!("No TypeParam aliases"),
                };
                SerSimpleType::Alias { name, c }
            }
            Container::Opaque(custom) => SerSimpleType::Opaque { custom, c: T::TAG },
        }
    }
}

impl From<HashableType> for SerSimpleType {
    fn from(value: HashableType) -> Self {
        match value {
            HashableType::Variable(s) => SerSimpleType::Var { name: s },
            HashableType::Int(w) => SerSimpleType::I { width: w },
            HashableType::String => SerSimpleType::S,
            HashableType::Container(c) => c.into(),
        }
    }
}

impl From<ClassicType> for SerSimpleType {
    fn from(value: ClassicType) -> Self {
        match value {
            ClassicType::F64 => SerSimpleType::F,
            ClassicType::Graph(inner) => SerSimpleType::G {
                signature: Box::new(*inner),
            },
            ClassicType::Container(c) => c.into(),
            ClassicType::Hashable(h) => h.into(),
        }
    }
}

impl From<SimpleType> for SerSimpleType {
    fn from(value: SimpleType) -> Self {
        match value {
            SimpleType::Classic(c) => c.into(),
            SimpleType::Qubit => SerSimpleType::Q,
            SimpleType::Qontainer(c) => c.into(),
        }
    }
}

impl From<TypeParam> for SerSimpleType {
    fn from(value: TypeParam) -> Self {
        match value {
            TypeParam::Type => Self::ST,
            TypeParam::ClassicType => Self::CT,
            TypeParam::HashableType => Self::HT,
            TypeParam::Container(c) => c.into(),
            TypeParam::Value(h) => h.into(),
        }
    }
}

pub(crate) fn box_convert_try<T, F>(value: T) -> Box<F>
where
    T: TryInto<F>,
    <T as TryInto<F>>::Error: std::fmt::Debug,
{
    Box::new((value).try_into().unwrap())
}

pub(crate) fn box_convert<T, F>(value: T) -> Box<F>
where
    T: Into<F>,
{
    Box::new((value).into())
}

macro_rules! handle_container {
   ($tag:ident, $variant:ident($($r:expr),*)) => {
        match $tag {
            SerializableTag::Simple => Deserialized::Simple(Container::<SimpleType>::$variant($($r),*).into()),
            SerializableTag::Classic => Deserialized::Classic(ClassicType::Container(Container::<ClassicType>::$variant($($r),*))),
            SerializableTag::Hashable => Deserialized::Hashable(HashableType::Container(Container::<HashableType>::$variant($($r),*))),
            SerializableTag::TypeParam => Deserialized::TypeParam(TypeParam::Container(Container::<TypeParam>::$variant($($r),*)))
        }
    }
}

impl From<SerSimpleType> for Deserialized {
    fn from(value: SerSimpleType) -> Self {
        match value {
            SerSimpleType::Q => Deserialized::Simple(SimpleType::Qubit),
            SerSimpleType::I { width } => Deserialized::Hashable(HashableType::Int(width)),
            SerSimpleType::F => Deserialized::Classic(ClassicType::F64),
            SerSimpleType::S => Deserialized::Hashable(HashableType::String),
            SerSimpleType::G { signature } => {
                Deserialized::Classic(ClassicType::Graph(Box::new(*signature)))
            }
            SerSimpleType::Tuple { row: inner, c } => {
                handle_container!(c, Tuple(Box::new(inner.try_convert_elems().unwrap())))
            }
            SerSimpleType::Sum { row: inner, c } => {
                handle_container!(c, Sum(Box::new(inner.try_convert_elems().unwrap())))
            }
            SerSimpleType::List { inner, c } => handle_container!(c, List(box_convert_try(*inner))),
            SerSimpleType::Map { k, v, c } => handle_container!(
                c,
                Map(Box::new((
                    (*k).try_into().unwrap(),
                    (*v).try_into().unwrap(),
                )))
            ),
            SerSimpleType::Array { inner, len, c } => {
                handle_container!(c, Array(box_convert_try(*inner), len))
            }
            SerSimpleType::Alias { name, c } => match c {
                TypeTag::Simple => {
                    Deserialized::Simple(SimpleType::Qontainer(Container::Alias(name)))
                }
                TypeTag::Classic => {
                    Deserialized::Classic(ClassicType::Container(Container::Alias(name)))
                }
                TypeTag::Hashable => {
                    Deserialized::Hashable(HashableType::Container(Container::Alias(name)))
                }
            },
            SerSimpleType::Opaque { custom, c } => {
                handle_container!(c, Opaque(custom))
            }
            SerSimpleType::Var { name: s } => Deserialized::Hashable(HashableType::Variable(s)),
            SerSimpleType::ST => Deserialized::TypeParam(TypeParam::Type),
            SerSimpleType::CT => Deserialized::TypeParam(TypeParam::ClassicType),
            SerSimpleType::HT => Deserialized::TypeParam(TypeParam::HashableType),
        }
    }
}

impl TryFrom<SerSimpleType> for SimpleType {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let d: Deserialized = value.into();
        Ok(match d {
            Deserialized::Simple(s) => s,
            Deserialized::Classic(c) => c.into(),
            Deserialized::Hashable(h) => h.into(),
            Deserialized::TypeParam(p) => return Err(format!("Not a SimpleType: {:?}", p)),
        })
    }
}

impl TryFrom<SerSimpleType> for TypeParam {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let d: Deserialized = value.into();
        Ok(match d {
            Deserialized::Hashable(h) => TypeParam::Value(h),
            Deserialized::TypeParam(p) => p,
            Deserialized::Classic(c) => return Err(format!("Not a valid TypeParam: {:?}", c)),
            Deserialized::Simple(s) => return Err(format!("Not a valid TypeParam: {:?}", s)),
        })
    }
}

impl TryFrom<SerSimpleType> for ClassicType {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let s: SimpleType = value.try_into()?;
        if let SimpleType::Classic(c) = s {
            Ok(c)
        } else {
            Err(format!("Not a ClassicType: {}", s))
        }
    }
}

impl TryFrom<SerSimpleType> for HashableType {
    type Error = String;
    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        match value.try_into()? {
            ClassicType::Hashable(h) => Ok(h),
            ty => Err(format!("Classic type is not hashable: {}", ty)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hugr::serialize::test::ser_roundtrip;
    use crate::types::{ClassicType, Container, HashableType, SimpleType};

    #[test]
    fn serialize_types_roundtrip() {
        // A Simple tuple
        let t = SimpleType::new_tuple(vec![
            SimpleType::Qubit,
            SimpleType::Classic(ClassicType::F64),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Classic sum
        let t = SimpleType::new_sum(vec![
            SimpleType::Classic(ClassicType::Hashable(HashableType::Int(4))),
            SimpleType::Classic(ClassicType::F64),
        ]);
        assert_eq!(ser_roundtrip(&t), t);

        // A Hashable list
        let t = SimpleType::Classic(ClassicType::Hashable(HashableType::Container(
            Container::List(Box::new(HashableType::Int(8))),
        )));
        assert_eq!(ser_roundtrip(&t), t);
    }

    #[test]
    fn serialize_types_current_behaviour() {
        // This list should be represented as a HashableType::Container.
        let malformed = SimpleType::Qontainer(Container::List(Box::new(SimpleType::Classic(
            ClassicType::Hashable(HashableType::Int(8)),
        ))));
        // If this behaviour changes, i.e. to return the well-formed version, that'd be fine.
        // Just to document current serialization behaviour that we leave it untouched.
        assert_eq!(ser_roundtrip(&malformed), malformed);
    }
}
