use super::ClassicType;

use super::Container;

use super::LinearType;
use super::PrimType;

use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::SimpleType;

use super::super::AbstractSignature;

use crate::ops::constant::HugrIntWidthStore;

#[derive(serde::Serialize, serde::Deserialize)]
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
        l: bool,
    },
    Map {
        k: Box<SimpleType>,
        v: Box<SimpleType>,
        l: bool,
    },
    Tuple {
        row: Box<TypeRow>,
        l: bool,
    },
    Sum {
        row: Box<TypeRow>,
        l: bool,
    },
    Array {
        inner: Box<SimpleType>,
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

impl<T: PrimType + Into<SimpleType>> From<Container<T>> for SerSimpleType {
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Sum(inner) => SerSimpleType::Sum {
                row: inner,
                l: T::LINEAR,
            },
            Container::List(inner) => SerSimpleType::List {
                inner: box_convert(*inner),
                l: T::LINEAR,
            },
            Container::Tuple(inner) => SerSimpleType::Tuple {
                row: inner,
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
                signature: Box::new(*inner),
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

impl From<LinearType> for SerSimpleType {
    fn from(value: LinearType) -> Self {
        match value {
            LinearType::Qubit => SerSimpleType::Q,
            LinearType::Container(c) => c.into(),
            LinearType::Qpaque(inner) => SerSimpleType::Opaque {
                custom: inner,
                l: true,
            },
        }
    }
}

impl From<SimpleType> for SerSimpleType {
    fn from(value: SimpleType) -> Self {
        match value {
            SimpleType::Linear(l) => l.into(),
            SimpleType::Classic(c) => c.into(),
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
            SerSimpleType::Q => LinearType::Qubit.into(),
            SerSimpleType::I { width } => ClassicType::Int(width).into(),
            SerSimpleType::F => ClassicType::F64.into(),
            SerSimpleType::S => ClassicType::String.into(),
            SerSimpleType::G { signature } => ClassicType::Graph(Box::new(*signature)).into(),
            SerSimpleType::Tuple {
                row: inner,
                l: true,
            } => Container::<LinearType>::Tuple(box_convert_try(*inner)).into(),
            SerSimpleType::Tuple {
                row: inner,
                l: false,
            } => Container::<ClassicType>::Tuple(box_convert_try(*inner)).into(),
            SerSimpleType::Sum {
                row: inner,
                l: true,
            } => Container::<LinearType>::Sum(box_convert_try(*inner)).into(),
            SerSimpleType::Sum {
                row: inner,
                l: false,
            } => Container::<ClassicType>::Sum(box_convert_try(*inner)).into(),
            SerSimpleType::List { inner, l: true } => {
                Container::<LinearType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::List { inner, l: false } => {
                Container::<ClassicType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::Map { k, v, l: true } => Container::<LinearType>::Map(Box::new((
                (*k).try_into().unwrap(),
                (*v).try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Map { k, v, l: false } => Container::<ClassicType>::Map(Box::new((
                (*k).try_into().unwrap(),
                (*v).try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Array {
                inner,
                len,
                l: true,
            } => Container::<LinearType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Array {
                inner,
                len,
                l: false,
            } => Container::<ClassicType>::Array(box_convert_try(*inner), len).into(),
            SerSimpleType::Alias { name: s, l: true } => Container::<LinearType>::Alias(s).into(),
            SerSimpleType::Alias { name: s, l: false } => Container::<ClassicType>::Alias(s).into(),
            SerSimpleType::Opaque { custom: c, l: true } => LinearType::Qpaque(c).into(),
            SerSimpleType::Opaque {
                custom: c,
                l: false,
            } => ClassicType::Opaque(c).into(),
            SerSimpleType::Var { name: s } => ClassicType::Variable(s).into(),
        }
    }
}
