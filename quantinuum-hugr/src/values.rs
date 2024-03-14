//! Representation of values (shared between [Const] and in future [TypeArg])
//!
//! [Const]: crate::ops::Const
//! [TypeArg]: crate::types::type_param::TypeArg

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};
use itertools::Itertools;
use smol_str::SmolStr;

use crate::extension::ExtensionSet;
use crate::macros::impl_box_clone;

use crate::{Hugr, HugrView};

use crate::types::{CustomCheckFailure, CustomType};

/// A value that can be stored as a static constant. Representing core types and
/// extension types.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "v")]
pub enum Value {
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
        values: Vec<Box<Value>>,
    },
}

impl Value {
    /// Returns the name of this [`Value`].
    pub fn name(&self) -> String {
        match self {
            Value::Extension { c: e } => format!("const:custom:{}", e.0.name()),
            Value::Function { hugr: h } => {
                let Some(t) = h.get_function_type() else {
                    panic!("HUGR root node isn't a valid function parent.");
                };
                format!("const:function:[{}]", t)
            }
            Value::Tuple { vs: vals } => {
                let names: Vec<_> = vals.iter().map(Value::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            Value::Sum { tag, values } => {
                format!("const:sum:{{tag:{tag}, vals:{values:?}}}")
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

    /// Constant Sum of a unit value, used to control branches.
    pub fn unit_sum(tag: usize) -> Self {
        Self::sum(tag, [])
    }

    /// Constant Sum with just one variant of unit type
    pub fn unary_unit_sum() -> Self {
        Self::unit_sum(0)
    }

    /// Tuple of values.
    pub fn tuple(items: impl IntoIterator<Item = Value>) -> Self {
        Self::Tuple {
            vs: items.into_iter().collect(),
        }
    }

    /// Sum value (could be of any compatible type - i.e. a Sum type where the
    /// `tag`th row is equal in length and compatible elementwise with `values`)
    pub fn sum(tag: usize, values: impl IntoIterator<Item = Value>) -> Self {
        Self::Sum {
            tag,
            values: values.into_iter().map(Box::new).collect_vec(),
        }
    }

    /// New custom value (of type that implements [`CustomConst`]).
    pub fn custom<C: CustomConst>(c: C) -> Self {
        Self::Extension { c: (Box::new(c),) }
    }

    /// For a Const holding a CustomConst, extract the CustomConst by downcasting.
    pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T> {
        if let Value::Extension { c: (custom,) } = self {
            custom.downcast_ref()
        } else {
            None
        }
    }

    /// The Extensions that must be supported to handle the value at runtime
    pub fn extension_reqs(&self) -> ExtensionSet {
        match self {
            Value::Extension { c } => c.0.extension_reqs().clone(),
            Value::Function { .. } => ExtensionSet::new(), // no extensions reqd to load Hugr (only to run)
            Value::Tuple { vs } => ExtensionSet::union_over(vs.iter().map(Value::extension_reqs)),
            Value::Sum { values, .. } => {
                ExtensionSet::union_over(values.iter().map(|x| x.extension_reqs()))
            }
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

    /// The extension(s) defining the custom value
    /// (a set to allow, say, a [List] of [USize])
    ///
    /// [List]: crate::std_extensions::collections::LIST_TYPENAME
    /// [USize]: crate::extension::prelude::USIZE_T
    fn extension_reqs(&self) -> ExtensionSet;

    /// Check the value is a valid instance of the provided type.
    fn validate(&self) -> Result<(), CustomCheckFailure> {
        Ok(())
    }

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    // Can't derive PartialEq for trait objects
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overloaded
        false
    }

    /// report the type
    fn custom_type(&self) -> CustomType;
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

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A value stored as a serialized blob that can report its own type.
pub struct CustomSerialized {
    typ: CustomType,
    value: serde_yaml::Value,
    extensions: ExtensionSet,
}

impl CustomSerialized {
    /// Creates a new [`CustomSerialized`].
    pub fn new(typ: CustomType, value: serde_yaml::Value, exts: impl Into<ExtensionSet>) -> Self {
        let extensions = exts.into();
        Self {
            typ,
            value,
            extensions,
        }
    }

    /// Returns the inner value.
    pub fn value(&self) -> &serde_yaml::Value {
        &self.value
    }
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> SmolStr {
        format!("yaml:{:?}", self.value).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }

    fn extension_reqs(&self) -> ExtensionSet {
        self.extensions.clone()
    }
    fn custom_type(&self) -> CustomType {
        self.typ.clone()
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
    use crate::ops::Const;
    use crate::std_extensions::arithmetic::float_types::{self, FLOAT64_CUSTOM_TYPE};
    use crate::type_row;
    use crate::types::{FunctionType, Type};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]

    /// A custom constant value used in testing
    pub(crate) struct CustomTestValue(pub CustomType);
    #[typetag::serde]
    impl CustomConst for CustomTestValue {
        fn name(&self) -> SmolStr {
            format!("CustomTestValue({:?})", self.0).into()
        }

        fn extension_reqs(&self) -> ExtensionSet {
            ExtensionSet::singleton(self.0.extension())
        }

        fn custom_type(&self) -> CustomType {
            self.0.clone()
        }
    }

    pub(crate) fn serialized_float(f: f64) -> Const {
        CustomSerialized {
            typ: FLOAT64_CUSTOM_TYPE,
            value: serde_yaml::Value::Number(f.into()),
            extensions: float_types::EXTENSION_ID.into(),
        }
        .into()
    }

    #[rstest]
    fn function_value(simple_dfg_hugr: Hugr) {
        let v = Value::Function {
            hugr: Box::new(simple_dfg_hugr),
        };

        let correct_type = Type::new_function(FunctionType::new_endo(type_row![
            crate::extension::prelude::BOOL_T
        ]));

        assert!(correct_type.check_type(&v).is_ok());
    }
}
