//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use crate::macros::impl_box_clone;
use crate::{Hugr, HugrView};

use crate::types::{CustomCheckFailure, CustomType};

/// A constant value of a primitive (or leaf) type.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "pv")]
pub enum PrimValue {
    /// An extension constant value, that can check it is of a given [CustomType].
    ///
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Extension {
        #[allow(missing_docs)]
        c: (Box<dyn CustomConst>,),
    },
    /// A higher-order function value.
    // TODO use a root parametrised hugr, e.g. Hugr<DFG>.
    Function {
        #[allow(missing_docs)]
        hugr: Box<Hugr>,
    },
}

impl PrimValue {
    fn name(&self) -> String {
        match self {
            PrimValue::Extension { c: e } => format!("const:custom:{}", e.0.name()),
            PrimValue::Function { hugr: h } => {
                let Some(t) = h.get_function_type() else {
                    panic!("HUGR root node isn't a valid function parent.");
                };
                format!("const:function:[{}]", t)
            }
        }
    }
}

/// A value that can be stored as a static constant. Representing core types and
/// extension types.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "v")]
pub enum Value {
    /// A primitive (non-container) value.
    Prim {
        #[allow(missing_docs)]
        val: PrimValue,
    },
    /// A tuple
    Tuple {
        #[allow(missing_docs)]
        vs: Vec<Value>,
    },
    /// A Sum variant -- for any Sum type where this value meets
    /// the type of the variant indicated by the tag
    Sum {
        /// The tag index of the variant
        tag: usize,
        /// The value of the variant
        value: Box<Value>,
    },
}

impl Value {
    /// Returns the name of this [`Value`].
    pub fn name(&self) -> String {
        match self {
            Value::Prim { val: p } => p.name(),
            Value::Tuple { vs: vals } => {
                let names: Vec<_> = vals.iter().map(Value::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            Value::Sum { tag, value: val } => {
                format!("const:sum:{{tag:{tag}, val:{}}}", val.name())
            }
        }
    }

    /// Description of the value.
    pub fn description(&self) -> &str {
        "Constant value"
    }

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> Self {
        Self::Tuple { vs: vec![] }
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
        Self::Tuple {
            vs: items.into_iter().collect(),
        }
    }

    /// Sum value (could be of any compatible type, e.g. a predicate)
    pub fn sum(tag: usize, value: Value) -> Self {
        Self::Sum {
            tag,
            value: Box::new(value),
        }
    }

    /// New custom value (of type that implements [`CustomConst`]).
    pub fn custom<C: CustomConst>(c: C) -> Self {
        Self::Prim {
            val: PrimValue::Extension { c: (Box::new(c),) },
        }
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
#[typetag::serde(tag = "c")]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> SmolStr;

    /// Check the value is a valid instance of the provided type.
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure>;

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    // Can't derive PartialEq for trait objects
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overloaded
        false
    }
}

/// Const equality for types that have PartialEq
pub fn downcast_equal_consts<T: CustomConst + PartialEq>(
    value: &T,
    other: &dyn CustomConst,
) -> bool {
    if let Some(other) = other.as_any().downcast_ref::<T>() {
        value == other
    } else {
        false
    }
}

/// Simpler trait for constant structs that have a known custom type to check against.
pub trait KnownTypeConst {
    /// The type of the constants.
    const TYPE: CustomType;

    /// Fixed implementation of [CustomConst::check_custom_type] that checks
    /// against known correct type.
    fn check_known_type(&self, typ: &CustomType) -> Result<(), CustomCheckFailure> {
        if typ == &Self::TYPE {
            Ok(())
        } else {
            Err(CustomCheckFailure::TypeMismatch {
                expected: Self::TYPE,
                found: typ.clone(),
            })
        }
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
    use rstest::rstest;

    use super::*;
    use crate::builder::test::simple_dfg_hugr;
    use crate::std_extensions::arithmetic::float_types::FLOAT64_CUSTOM_TYPE;
    use crate::type_row;
    use crate::types::{FunctionType, Type, TypeBound};

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
            typ: FLOAT64_CUSTOM_TYPE,
            value: serde_yaml::Value::Number(f.into()),
        })
    }

    #[rstest]
    fn function_value(simple_dfg_hugr: Hugr) {
        let v = Value::Prim {
            val: PrimValue::Function {
                hugr: Box::new(simple_dfg_hugr),
            },
        };

        let correct_type = Type::new_function(FunctionType::new_linear(type_row![
            crate::extension::prelude::BOOL_T
        ]));

        assert!(correct_type.check_type(&v).is_ok());
    }
}
