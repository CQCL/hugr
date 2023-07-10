use super::ClassicType;

use super::Container;

use super::PrimType;

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
        l: bool,
    },
    Map {
        k: Box<SerSimpleType>,
        v: Box<SerSimpleType>,
        l: bool,
    },
    Tuple {
        row: Box<TypeRow<SerSimpleType>>,
        l: bool,
    },
    Sum {
        row: Box<TypeRow<SerSimpleType>>,
        l: bool,
    },
    Array {
        inner: Box<SerSimpleType>,
        len: usize,
        l: bool,
    },
    Opaque {
        custom: CustomType,
        l: bool,
    },
    Alias {
        name: SmolStr,
        l: bool,
    },
    Var {
        name: SmolStr,
    },
}

impl PrimType for SerSimpleType {
    const LINEAR: bool = true;
}

impl<T: PrimType> From<Container<T>> for SerSimpleType
where
    SerSimpleType: From<T>,
    SimpleType: From<T>,
{
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Sum(inner) => SerSimpleType::Sum {
                row: Box::new(inner.map_into()),
                l: T::LINEAR,
            },
            Container::List(inner) => SerSimpleType::List {
                inner: Box::new((*inner).into()),
                l: T::LINEAR,
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: Box::new(inner.map_into()),
                l: T::LINEAR,
            },
            Container::Map(inner) => SerSimpleType::Map {
                k: Box::new(inner.0.into()),
                v: Box::new(inner.1.into()),
                l: T::LINEAR,
            },
            Container::Array(inner, len) => SerSimpleType::Array {
                inner: box_convert(*inner),
                len,
                l: T::LINEAR,
            },
            Container::Alias(name) => SerSimpleType::Alias { name, l: T::LINEAR },
        }
    }
}

impl From<ClassicType> for SerSimpleType {
    fn from(value: ClassicType) -> Self {
        match value {
            ClassicType::Int(w) => SerSimpleType::I { width: w },
            ClassicType::F64 => SerSimpleType::F,
            ClassicType::Graph(inner) => SerSimpleType::G {
                resources: Box::new(inner.0),
                signature: Box::new(inner.1),
            },
            ClassicType::String => SerSimpleType::S,
            ClassicType::Container(c) => c.into(),
            ClassicType::Opaque(inner) => SerSimpleType::Opaque {
                custom: inner,
                l: false,
            },
            ClassicType::Variable(s) => SerSimpleType::Var { name: s },
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
                l: true,
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
            SerSimpleType::I { width } => ClassicType::Int(width).into(),
            SerSimpleType::F => ClassicType::F64.into(),
            SerSimpleType::S => ClassicType::String.into(),
            SerSimpleType::G {
                resources,
                signature,
            } => ClassicType::Graph(Box::new((*resources, *signature))).into(),
            SerSimpleType::Tuple {
                row: inner,
                l: true,
            } => Container::<SimpleType>::Tuple(Box::new(inner.map_into())).into(),
            SerSimpleType::Tuple {
                row: inner,
                l: false,
            } => {
                Container::<ClassicType>::Tuple(Box::new(inner.try_convert_elems().unwrap())).into()
            }
            SerSimpleType::Sum {
                row: inner,
                l: true,
            } => Container::<SimpleType>::Sum(Box::new(inner.map_into())).into(),
            SerSimpleType::Sum {
                row: inner,
                l: false,
            } => Container::<ClassicType>::Sum(Box::new(inner.try_convert_elems().unwrap())).into(),
            SerSimpleType::List { inner, l: true } => {
                Container::<SimpleType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::List { inner, l: false } => {
                Container::<ClassicType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::Map { k, v, l: true } => {
                Container::<SimpleType>::Map(Box::new(((*k).try_into().unwrap(), (*v).into())))
                    .into()
            }
            SerSimpleType::Map { k, v, l: false } => Container::<ClassicType>::Map(Box::new((
                (*k).try_into().unwrap(),
                (*v).try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Array {
                inner,
                len,
                l: true,
            } => Container::<SimpleType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Array {
                inner,
                len,
                l: false,
            } => Container::<ClassicType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Alias { name: s, l: true } => Container::<SimpleType>::Alias(s).into(),
            SerSimpleType::Alias { name: s, l: false } => Container::<ClassicType>::Alias(s).into(),
            SerSimpleType::Opaque { custom: c, l: true } => SimpleType::Qpaque(c),
            SerSimpleType::Opaque {
                custom: c,
                l: false,
            } => ClassicType::Opaque(c).into(),
            SerSimpleType::Var { name: s } => ClassicType::Variable(s).into(),
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
