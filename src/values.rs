//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use crate::macros::impl_box_clone;
use crate::types::{CustomCheckFailure, CustomType};

/// A constant value of a primitive (or leaf) type.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PrimValue {
    /// An extension constant value, that can check it is of a given [CustomType].
    ///
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Extension((Box<dyn CustomConst>,)),
    /// A higher-order function value.
    // TODO add  HUGR<DFG> payload
    Graph,
}

impl PrimValue {
    fn name(&self) -> String {
        match self {
            PrimValue::Extension(e) => format!("const:custom:{}", e.0.name()),
            PrimValue::Graph => todo!(),
        }
    }
}

/// A value that can be stored as a static constant. Representing core types and
/// extension types.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Value {
    /// A primitive (non-container) value.
    Prim(PrimValue),
    /// A tuple
    Tuple(Vec<Value>),
    /// A Sum variant -- for any Sum type where this value meets
    /// the type of the variant indicated by the tag
    Sum(usize, Box<Value>), // Tag and value
}

impl Value {
    /// Returns the name of this [`Value`].
    pub fn name(&self) -> String {
        match self {
            Value::Prim(p) => p.name(),
            Value::Tuple(vals) => {
                let names: Vec<_> = vals.iter().map(Value::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            Value::Sum(tag, val) => format!("const:sum:{{tag:{tag}, val:{}}}", val.name()),
        }
    }

    /// Description of the value.
    pub fn description(&self) -> &str {
        "Constant value"
    }

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> Self {
        Self::Tuple(vec![])
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize) -> Self {
        Self::sum(tag, Self::unit())
    }

    /// Constant Sum over Tuples with just one variant of unit type
    pub fn simple_unary_predicate() -> Self {
        Self::simple_predicate(0)
    }

    /// Tuple of values.
    pub fn tuple(items: impl IntoIterator<Item = Value>) -> Self {
        Self::Tuple(items.into_iter().collect())
    }

    /// Sum value (could be of any compatible type, e.g. a predicate)
    pub fn sum(tag: usize, value: Value) -> Self {
        Self::Sum(tag, Box::new(value))
    }

    /// New custom value (of type that implements [`CustomConst`]).
    pub fn custom<C: CustomConst>(c: C) -> Self {
        Self::Prim(PrimValue::Extension((Box::new(c),)))
    }
}

impl<T: CustomConst> From<T> for Value {
    fn from(v: T) -> Self {
        Self::custom(v)
    }
}

/// Constant value for opaque [`CustomType`]s.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
#[typetag::serde]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> SmolStr;

    /// Check the value is a valid instance of the provided type.
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure>;

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A value stored as a serialized blob that can report its own type.
pub struct CustomSerialized {
    typ: CustomType,
    value: serde_yaml::Value,
}

impl CustomSerialized {
    /// Creates a new [`CustomSerialized`].
    pub fn new(typ: CustomType, value: serde_yaml::Value) -> Self {
        Self { typ, value }
    }
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> SmolStr {
        format!("yaml:{:?}", self.value).into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if &self.typ == typ {
            Ok(())
        } else {
            Err(CustomCheckFailure::TypeMismatch {
                expected: typ.clone(),
                found: self.typ.clone(),
            })
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::types::{custom::test::COPYABLE_CUST, TypeBound};

    use super::*;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]

    /// A custom constant value used in testing that purports to be an instance
    /// of a custom type with a specific type bound.
    pub(crate) struct CustomTestValue(pub TypeBound);
    #[typetag::serde]
    impl CustomConst for CustomTestValue {
        fn name(&self) -> SmolStr {
            format!("CustomTestValue({:?})", self.0).into()
        }

        fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
            if self.0 == typ.bound() {
                Ok(())
            } else {
                Err(CustomCheckFailure::Message(
                    "CustomTestValue check fail.".into(),
                ))
            }
        }
    }

    pub(crate) fn serialized_float(f: f64) -> Value {
        Value::custom(CustomSerialized {
            typ: COPYABLE_CUST,
            value: serde_yaml::Value::Number(f.into()),
        })
    }
}
