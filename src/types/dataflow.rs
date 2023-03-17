//! Dataflow types

use downcast_rs::{impl_downcast, Downcast};
use std::any::Any;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

use super::Resource;
use crate::macros::impl_box_clone;

/// A type that represents concrete data.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DataType {
    Variable(String), // TODO: How are variables represented?
    Int,
    Bool,
    F64,
    Quat64,
    Angle,
    Graph {
        resources: Resource,
        input: RowType,
        output: RowType,
    },
    Pair(Box<DataType>, Box<DataType>),
    List(Box<DataType>),
    // TODO: Complete this list

    // Linear types
    Qubit,
    Money,
    //
    Resource(Resource),
    /// An opaque operation that can be downcasted by the extensions that define it.
    Opaque(Box<dyn CustomType>),
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Variable(l0), Self::Variable(r0)) => l0 == r0,
            (
                Self::Graph {
                    resources: l_resources,
                    input: l_input,
                    output: l_output,
                },
                Self::Graph {
                    resources: r_resources,
                    input: r_input,
                    output: r_output,
                },
            ) => l_resources == r_resources && l_input == r_input && l_output == r_output,
            (Self::Pair(l0, l1), Self::Pair(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::List(l0), Self::List(r0)) => l0 == r0,
            (Self::Resource(l0), Self::Resource(r0)) => l0 == r0,
            (Self::Opaque(l0), Self::Opaque(r0)) => l0.eq(&**r0),
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for DataType {}

impl DataType {
    pub fn is_linear(&self) -> bool {
        match self {
            Self::Qubit | Self::Money => true,
            Self::Opaque(op) => op.is_linear(),
            _ => false,
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        Self::Qubit
    }
}

pub trait CustomType: Send + Sync + std::fmt::Debug + CustomTypeBoxClone + Any + Downcast {
    fn name(&self) -> &str;

    /// Check if two custom ops are equal, by downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomType) -> bool {
        let _ = other;
        false
    }

    fn is_linear(&self) -> bool {
        false
    }
}

impl_downcast!(CustomType);
impl_box_clone!(CustomType, CustomTypeBoxClone);

/// List of types, used for function signatures
#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub struct RowType(pub Vec<DataType>);

#[cfg_attr(feature = "pyo3", pymethods)]
impl RowType {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.0.iter().all(|typ| typ.is_linear())
    }

    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        !self
            .0
            .iter()
            .any(|typ| matches!(typ, DataType::Qubit | DataType::Money))
    }
}

impl RowType {
    pub fn new(types: Vec<DataType>) -> Self {
        Self(types)
    }
}

impl From<Vec<DataType>> for RowType {
    fn from(types: Vec<DataType>) -> Self {
        Self::new(types)
    }
}
