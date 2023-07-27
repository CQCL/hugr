use super::ClassicType;

use super::Container;

use super::HashableType;
use super::PrimType;
use super::TypeTag;

use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::SimpleType;

use super::super::AbstractSignature;

use crate::ops::constant::HugrIntWidthStore;

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
        inner: Box<SimpleType>,
        c: TypeTag,
    },
    Map {
        k: Box<SerSimpleType>,
        v: Box<SerSimpleType>,
        c: TypeTag,
    },
    Tuple {
        row: Box<TypeRow<SerSimpleType>>,
        c: TypeTag,
    },
    Sum {
        row: Box<TypeRow<SerSimpleType>>,
        c: TypeTag,
    },
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
        c: TypeTag,
    },
    Opaque {
        custom: CustomType,
        c: TypeTag,
    },
    Alias {
        name: SmolStr,
        c: TypeTag,
    },
    Var {
        name: SmolStr,
    },
}

impl super::sealed::Sealed for SerSimpleType {}
impl PrimType for SerSimpleType {
    fn tag(&self) -> TypeTag {
        unimplemented!()
    }
}

trait SerializableType: PrimType {
    const TAG: TypeTag;
}

impl SerializableType for ClassicType {
    const TAG: TypeTag = TypeTag::Classic;
}

impl SerializableType for SimpleType {
    const TAG: TypeTag = TypeTag::Simple;
}

impl SerializableType for HashableType {
    const TAG: TypeTag = TypeTag::Hashable;
}

impl<T: SerializableType> From<Container<T>> for SerSimpleType
where
    SerSimpleType: From<T>,
    SimpleType: From<T>,
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
            Container::Alias(name) => SerSimpleType::Alias { name, c: T::TAG },
        }
    }
}

impl From<HashableType> for SerSimpleType {
    fn from(value: HashableType) -> Self {
        match value {
            HashableType::Variable(s) => SerSimpleType::Var { name: s },
            HashableType::Int(w) => SerSimpleType::I { width: w },
            HashableType::Opaque(c) => SerSimpleType::Opaque {
                custom: c,
                c: TypeTag::Hashable,
            },
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
            ClassicType::Opaque(inner) => SerSimpleType::Opaque {
                custom: inner,
                c: TypeTag::Classic,
            },
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
            SimpleType::Qpaque(inner) => SerSimpleType::Opaque {
                custom: inner,
                c: TypeTag::Simple,
            },
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
            TypeTag::Simple => (Container::<SimpleType>::$variant($($r),*)).into(),
            TypeTag::Classic => (Container::<ClassicType>::$variant($($r),*)).into(),
            TypeTag::Hashable => (Container::<HashableType>::$variant($($r),*)).into()
        }
    }
}

impl From<SerSimpleType> for SimpleType {
    fn from(value: SerSimpleType) -> Self {
        match value {
            SerSimpleType::Q => SimpleType::Qubit,
            SerSimpleType::I { width } => HashableType::Int(width).into(),
            SerSimpleType::F => ClassicType::F64.into(),
            SerSimpleType::S => HashableType::String.into(),
            SerSimpleType::G { signature } => ClassicType::Graph(Box::new(*signature)).into(),
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
            SerSimpleType::Alias { name: s, c } => handle_container!(c, Alias(s)),
            SerSimpleType::Opaque { custom, c } => match c {
                TypeTag::Simple => SimpleType::Qpaque(custom),
                TypeTag::Classic => ClassicType::Opaque(custom).into(),
                TypeTag::Hashable => HashableType::Opaque(custom).into(),
            },
            SerSimpleType::Var { name: s } => {
                ClassicType::Hashable(HashableType::Variable(s)).into()
            }
        }
    }
}

impl TryFrom<SerSimpleType> for ClassicType {
    type Error = String;

    fn try_from(value: SerSimpleType) -> Result<Self, Self::Error> {
        let s: SimpleType = value.into();
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
