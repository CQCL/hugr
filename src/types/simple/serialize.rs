use super::ClassicType;

use super::Container;

use super::HashableType;
use super::PrimType;
use super::TypeTag;

use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::SimpleType;

use super::super::Signature;

use crate::ops::constant::HugrIntWidthStore;
use crate::resource::ResourceSet;

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
        resources: Box<ResourceSet>,
        signature: Box<Signature>,
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

impl PrimType for SerSimpleType {
    fn tag(&self) -> TypeTag {
        unimplemented!()
    }
}

trait SerializableType: PrimType {
    const CLASS: TypeTag;
}

impl SerializableType for ClassicType {
    const CLASS: TypeTag = TypeTag::Classic;
}

impl SerializableType for SimpleType {
    const CLASS: TypeTag = TypeTag::Any;
}

impl SerializableType for HashableType {
    const CLASS: TypeTag = TypeTag::Hashable;
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
                c: T::CLASS, // We could inspect inner.common_class(), but this should have been done already
            },
            Container::List(inner) => SerSimpleType::List {
                inner: Box::new((*inner).into()),
                c: T::CLASS, // We could inspect inner.tag(), but this should have been done already
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: Box::new(inner.map_into()),
                c: T::CLASS,
            },
            Container::Map(inner) => SerSimpleType::Map {
                k: Box::new(inner.0.into()),
                v: Box::new(inner.1.into()),
                c: T::CLASS,
            },
            Container::Array(inner, len) => SerSimpleType::Array {
                inner: box_convert(*inner),
                len,
                c: T::CLASS,
            },
            Container::Alias(name) => SerSimpleType::Alias { name, c: T::CLASS },
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
                resources: Box::new(inner.0),
                signature: Box::new(inner.1),
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
                c: TypeTag::Any,
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

impl From<SerSimpleType> for SimpleType {
    fn from(value: SerSimpleType) -> Self {
        match value {
            SerSimpleType::Q => SimpleType::Qubit,
            SerSimpleType::I { width } => HashableType::Int(width).into(),
            SerSimpleType::F => ClassicType::F64.into(),
            SerSimpleType::S => HashableType::String.into(),
            SerSimpleType::G {
                resources,
                signature,
            } => ClassicType::Graph(Box::new((*resources, *signature))).into(),
            SerSimpleType::Tuple {
                row: inner,
                c: TypeTag::Any,
            } => Container::<SimpleType>::Tuple(Box::new(inner.map_into())).into(),
            SerSimpleType::Tuple {
                row: inner,
                c: TypeTag::Classic,
            } => {
                Container::<ClassicType>::Tuple(Box::new(inner.try_convert_elems().unwrap())).into()
            }
            SerSimpleType::Tuple {
                row: inner,
                c: TypeTag::Hashable,
            } => Container::<HashableType>::Tuple(Box::new(inner.try_convert_elems().unwrap()))
                .into(),
            SerSimpleType::Sum {
                row: inner,
                c: TypeTag::Any,
            } => Container::<SimpleType>::Sum(Box::new(inner.map_into())).into(),
            SerSimpleType::Sum {
                row: inner,
                c: TypeTag::Classic,
            } => Container::<ClassicType>::Sum(Box::new(inner.try_convert_elems().unwrap())).into(),
            SerSimpleType::Sum {
                row: inner,
                c: TypeTag::Hashable,
            } => {
                Container::<HashableType>::Sum(Box::new(inner.try_convert_elems().unwrap())).into()
            }
            SerSimpleType::List {
                inner,
                c: TypeTag::Any,
            } => Container::<SimpleType>::List(box_convert_try(*inner)).into(),
            SerSimpleType::List {
                inner,
                c: TypeTag::Classic,
            } => Container::<ClassicType>::List(box_convert_try(*inner)).into(),
            SerSimpleType::List {
                inner,
                c: TypeTag::Hashable,
            } => Container::<HashableType>::List(box_convert_try(*inner)).into(),
            SerSimpleType::Map {
                k,
                v,
                c: TypeTag::Any,
            } => Container::<SimpleType>::Map(Box::new(((*k).try_into().unwrap(), (*v).into())))
                .into(),
            SerSimpleType::Map {
                k,
                v,
                c: TypeTag::Classic,
            } => Container::<ClassicType>::Map(Box::new((
                (*k).try_into().unwrap(),
                (*v).try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Map {
                k,
                v,
                c: TypeTag::Hashable,
            } => Container::<HashableType>::Map(Box::new((
                (*k).try_into().unwrap(),
                (*v).try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Array {
                inner,
                len,
                c: TypeTag::Any,
            } => Container::<SimpleType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Array {
                inner,
                len,
                c: TypeTag::Classic,
            } => Container::<ClassicType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Array {
                inner,
                len,
                c: TypeTag::Hashable,
            } => Container::<HashableType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Alias {
                name: s,
                c: TypeTag::Any,
            } => Container::<SimpleType>::Alias(s).into(),
            SerSimpleType::Alias {
                name: s,
                c: TypeTag::Classic,
            } => Container::<ClassicType>::Alias(s).into(),
            SerSimpleType::Alias {
                name: s,
                c: TypeTag::Hashable,
            } => Container::<HashableType>::Alias(s).into(),
            SerSimpleType::Opaque {
                custom: c,
                c: TypeTag::Any,
            } => SimpleType::Qpaque(c),
            SerSimpleType::Opaque {
                custom: c,
                c: TypeTag::Classic,
            } => ClassicType::Opaque(c).into(),
            SerSimpleType::Opaque {
                custom: c,
                c: TypeTag::Hashable,
            } => HashableType::Opaque(c).into(),
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
