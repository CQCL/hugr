use super::ClassicType;

use super::Container;

use super::LinearType;
use super::PrimType;

use smol_str::SmolStr;

use super::super::custom::CustomType;

use super::TypeRow;

use super::SimpleType;

use super::super::Signature;

use crate::resource::ResourceSet;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) enum SerSimpleType {
    Q,
    I(u8),
    F,
    S,
    G(Box<(ResourceSet, Signature)>),
    List(Box<SimpleType>, bool),
    Map(Box<(SimpleType, SimpleType)>, bool),
    Tuple(Box<TypeRow>, bool),
    Sum(Box<TypeRow>, bool),
    Array(Box<SimpleType>, usize, bool),
    Opaque(CustomType, bool),
    Alias(SmolStr, bool),
    Var(SmolStr),
}

impl<T: PrimType + Into<SimpleType>> From<Container<T>> for SerSimpleType {
    fn from(value: Container<T>) -> Self {
        match value {
            Container::Sum(inner) => SerSimpleType::Sum(inner, T::LINEAR),
            Container::List(inner) => SerSimpleType::List(box_convert(*inner), T::LINEAR),
            Container::Tuple(inner) => SerSimpleType::Tuple(inner, T::LINEAR),
            Container::Map(inner) => {
                SerSimpleType::Map(Box::new((inner.0.into(), inner.1.into())), T::LINEAR)
            }
            Container::Array(inner, len) => {
                SerSimpleType::Array(box_convert(*inner), len, T::LINEAR)
            }
            Container::Alias(name) => SerSimpleType::Alias(name, T::LINEAR),
        }
    }
}

impl From<ClassicType> for SerSimpleType {
    fn from(value: ClassicType) -> Self {
        match value {
            ClassicType::Int(w) => SerSimpleType::I(w),
            ClassicType::F64 => SerSimpleType::F,
            ClassicType::Graph(inner) => SerSimpleType::G(inner),
            ClassicType::String => SerSimpleType::S,
            ClassicType::Container(c) => c.into(),
            ClassicType::Opaque(inner) => SerSimpleType::Opaque(inner, false),
            ClassicType::Variable(s) => SerSimpleType::Var(s),
        }
    }
}

impl From<LinearType> for SerSimpleType {
    fn from(value: LinearType) -> Self {
        match value {
            LinearType::Qubit => SerSimpleType::Q,
            LinearType::Container(c) => c.into(),
            LinearType::Qpaque(inner) => SerSimpleType::Opaque(inner, true),
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
            SerSimpleType::I(width) => ClassicType::Int(width).into(),
            SerSimpleType::F => ClassicType::F64.into(),
            SerSimpleType::S => ClassicType::String.into(),
            SerSimpleType::G(contents) => ClassicType::Graph(contents).into(),
            SerSimpleType::Tuple(inner, true) => {
                Container::<LinearType>::Tuple(box_convert_try(*inner)).into()
            }
            SerSimpleType::Tuple(inner, false) => {
                Container::<ClassicType>::Tuple(box_convert_try(*inner)).into()
            }
            SerSimpleType::Sum(inner, true) => {
                Container::<LinearType>::Sum(box_convert_try(*inner)).into()
            }
            SerSimpleType::Sum(inner, false) => {
                Container::<ClassicType>::Sum(box_convert_try(*inner)).into()
            }
            SerSimpleType::List(inner, true) => {
                Container::<LinearType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::List(inner, false) => {
                Container::<ClassicType>::List(box_convert_try(*inner)).into()
            }
            SerSimpleType::Map(inner, true) => Container::<LinearType>::Map(Box::new((
                inner.0.try_into().unwrap(),
                inner.1.try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Map(inner, false) => Container::<ClassicType>::Map(Box::new((
                inner.0.try_into().unwrap(),
                inner.1.try_into().unwrap(),
            )))
            .into(),
            SerSimpleType::Array(inner, len, true) => {
                Container::<LinearType>::Array(box_convert_try(*inner), len).into()
            }
            SerSimpleType::Array(inner, len, false) => {
                Container::<ClassicType>::Array(box_convert_try(*inner), len).into()
            }
            SerSimpleType::Alias(s, true) => Container::<LinearType>::Alias(s).into(),
            SerSimpleType::Alias(s, false) => Container::<ClassicType>::Alias(s).into(),
            SerSimpleType::Opaque(c, true) => LinearType::Qpaque(c).into(),
            SerSimpleType::Opaque(c, false) => ClassicType::Opaque(c).into(),
            SerSimpleType::Var(s) => ClassicType::Variable(s).into(),
        }
    }
}
