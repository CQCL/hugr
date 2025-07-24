//! Constant value definitions.

mod custom;
mod serialize;

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher; // Moves into std::hash in Rust 1.76.
use std::hash::{Hash, Hasher};

use super::{NamedOp, OpName, OpTrait, StaticTag};
use super::{OpTag, OpType};
use crate::envelope::serde_with::AsStringEnvelope;
use crate::types::{CustomType, EdgeKind, Signature, SumType, SumTypeError, Type, TypeRow};
use crate::{Hugr, HugrView};
use serialize::SerialSum;

use delegate::delegate;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use smol_str::SmolStr;
use thiserror::Error;

pub use custom::{
    CustomConst, CustomSerialized, TryHash, downcast_equal_consts, get_pair_of_input_values,
    get_single_input_value,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// An operation returning a constant value.
///
/// Represents core types and extension types.
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Const {
    /// The [Value] of the constant.
    #[serde(rename = "v")]
    pub value: Value,
}

impl Const {
    /// Create a new [`Const`] operation.
    #[must_use]
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    /// The inner value of the [`Const`]
    #[must_use]
    pub fn value(&self) -> &Value {
        &self.value
    }

    delegate! {
        to self.value {
            /// Returns the type of this constant.
            #[must_use] pub fn get_type(&self) -> Type;
            /// For a Const holding a CustomConst, extract the CustomConst by
            /// downcasting.
            #[must_use] pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T>;

            /// Check the value.
            pub fn validate(&self) -> Result<(), ConstTypeError>;
        }
    }
}

impl From<Value> for Const {
    fn from(value: Value) -> Self {
        Self::new(value)
    }
}

impl NamedOp for Const {
    fn name(&self) -> OpName {
        self.value().name()
    }
}

impl StaticTag for Const {
    const TAG: OpTag = OpTag::Const;
}

impl OpTrait for Const {
    fn description(&self) -> &'static str {
        "Constant value"
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Const(self.get_type()))
    }

    // Constants cannot refer to TypeArgs of the enclosing Hugr, so no substitute().
}

impl From<Const> for Value {
    fn from(konst: Const) -> Self {
        konst.value
    }
}

impl AsRef<Value> for Const {
    fn as_ref(&self) -> &Value {
        self.value()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SerialSum")]
#[serde(into = "SerialSum")]
/// A Sum variant, with a tag indicating the index of the variant and its
/// value.
pub struct Sum {
    /// The tag index of the variant.
    pub tag: usize,
    /// The value of the variant.
    ///
    /// Sum variants are always a row of values, hence the Vec.
    pub values: Vec<Value>,
    /// The full type of the Sum, including the other variants.
    pub sum_type: SumType,
}

impl Sum {
    /// If value is a sum with a single row variant, return the row.
    #[must_use]
    pub fn as_tuple(&self) -> Option<&[Value]> {
        // For valid instances, the type row will not have any row variables.
        self.sum_type.as_tuple().map(|_| self.values.as_ref())
    }

    fn try_hash<H: Hasher>(&self, st: &mut H) -> bool {
        maybe_hash_values(&self.values, st) && {
            st.write_usize(self.tag);
            self.sum_type.hash(st);
            true
        }
    }
}

pub(crate) fn maybe_hash_values<H: Hasher>(vals: &[Value], st: &mut H) -> bool {
    // We can't mutate the Hasher with the first element
    // if any element, even the last, fails.
    let mut hasher = DefaultHasher::new();
    vals.iter().all(|e| e.try_hash(&mut hasher)) && {
        st.write_u64(hasher.finish());
        true
    }
}

#[serde_as]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "v")]
/// A value that can be stored as a static constant. Representing core types and
/// extension types.
pub enum Value {
    /// An extension constant value, that can check it is of a given [CustomType].
    Extension {
        #[serde(flatten)]
        /// The custom constant value.
        e: OpaqueValue,
    },
    /// A higher-order function value.
    Function {
        /// A Hugr defining the function.
        #[serde_as(as = "Box<AsStringEnvelope>")]
        hugr: Box<Hugr>,
    },
    /// A Sum variant, with a tag indicating the index of the variant and its
    /// value.
    #[serde(alias = "Tuple")]
    Sum(Sum),
}

/// An opaque newtype around a [`Box<dyn CustomConst>`](CustomConst).
///
/// This type has special serialization behaviour in order to support
/// serialization and deserialization of unknown impls of [CustomConst].
///
/// During serialization we first serialize the internal [`dyn` CustomConst](CustomConst)
/// into a [serde_json::Value]. We then create a [CustomSerialized] wrapping
/// that value.  That [CustomSerialized] is then serialized in place of the
/// [OpaqueValue].
///
/// During deserialization, first we deserialize a [CustomSerialized]. We
/// attempt to deserialize the internal [serde_json::Value] using the [`Box<dyn
/// CustomConst>`](CustomConst) impl. This will fail if the appropriate `impl CustomConst`
/// is not linked into the running program, in which case we coerce the
/// [CustomSerialized] into a [`Box<dyn CustomConst>`](CustomConst). The [OpaqueValue] is
/// then produced from the [`Box<dyn [CustomConst]>`](CustomConst).
///
/// In the case where the internal serialized value of a `CustomSerialized`
/// is another `CustomSerialized` we do not attempt to recurse. This behaviour
/// may change in future.
///
#[cfg_attr(not(miri), doc = "```")] // this doctest depends on typetag, so fails with miri
#[cfg_attr(miri, doc = "```ignore")]
/// use serde::{Serialize,Deserialize};
/// use hugr::{
///   types::Type,ops::constant::{OpaqueValue, ValueName, CustomConst, CustomSerialized},
///   extension::{ExtensionSet, prelude::{usize_t, ConstUsize}},
///   std_extensions::arithmetic::int_types};
/// use serde_json::json;
///
/// let expected_json = json!({
///     "typ": usize_t(),
///     "value": {'c': "ConstUsize", 'v': 1}
/// });
/// let ev = OpaqueValue::new(ConstUsize::new(1));
/// assert_eq!(&serde_json::to_value(&ev).unwrap(), &expected_json);
/// assert_eq!(ev, serde_json::from_value(expected_json).unwrap());
///
/// let ev = OpaqueValue::new(CustomSerialized::new(usize_t().clone(), serde_json::Value::Null));
/// let expected_json = json!({
///     "typ": usize_t(),
///     "value": null
/// });
///
/// assert_eq!(&serde_json::to_value(ev.clone()).unwrap(), &expected_json);
/// assert_eq!(ev, serde_json::from_value(expected_json).unwrap());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpaqueValue {
    #[serde(flatten, with = "self::custom::serde_extension_value")]
    v: Box<dyn CustomConst>,
}

impl OpaqueValue {
    /// Create a new [`OpaqueValue`] from any [`CustomConst`].
    pub fn new(cc: impl CustomConst) -> Self {
        Self { v: Box::new(cc) }
    }

    /// Returns a reference to the internal [`CustomConst`].
    #[must_use]
    pub fn value(&self) -> &dyn CustomConst {
        self.v.as_ref()
    }

    /// Returns a reference to the internal [`CustomConst`].
    pub(crate) fn value_mut(&mut self) -> &mut dyn CustomConst {
        self.v.as_mut()
    }

    delegate! {
        to self.value() {
            /// Returns the type of the internal [`CustomConst`].
            #[must_use] pub fn get_type(&self) -> Type;
            /// An identifier of the internal [`CustomConst`].
            #[must_use] pub fn name(&self) -> ValueName;
        }
    }
}

impl<CC: CustomConst> From<CC> for OpaqueValue {
    fn from(x: CC) -> Self {
        Self::new(x)
    }
}

impl From<Box<dyn CustomConst>> for OpaqueValue {
    fn from(value: Box<dyn CustomConst>) -> Self {
        Self { v: value }
    }
}

impl PartialEq for OpaqueValue {
    fn eq(&self, other: &Self) -> bool {
        self.value().equal_consts(other.value())
    }
}

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum CustomCheckFailure {
    /// The value had a specific type that was not what was expected
    #[error("Expected type: {expected} but value was of type: {found}")]
    TypeMismatch {
        /// The expected custom type.
        expected: Box<CustomType>,
        /// The custom type found when checking.
        found: Box<Type>,
    },
    /// Any other message
    #[error("{0}")]
    Message(String),
}

/// Errors that arise from typechecking constants
#[derive(Clone, Debug, PartialEq, Error)]
#[non_exhaustive]
pub enum ConstTypeError {
    /// Invalid sum type definition.
    #[error("{0}")]
    SumType(#[from] SumTypeError),
    /// Function constant missing a function type.
    #[error(
        "A function constant cannot be defined using a Hugr with root of type {hugr_root_type}. Must be a monomorphic function."
    )]
    NotMonomorphicFunction {
        /// The root node type of the Hugr that (claims to) define the function constant.
        hugr_root_type: Box<OpType>,
    },
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ConstCheckFail(Box<Type>, Value),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0}")]
    CustomCheckFail(#[from] CustomCheckFailure),
}

/// Hugrs (even functions) inside Consts must be monomorphic
fn mono_fn_type(h: &Hugr) -> Result<Cow<'_, Signature>, ConstTypeError> {
    let err = || ConstTypeError::NotMonomorphicFunction {
        hugr_root_type: Box::new(h.entrypoint_optype().clone()),
    };
    if let Some(pf) = h.poly_func_type() {
        match pf.try_into() {
            Ok(sig) => return Ok(Cow::Owned(sig)),
            Err(_) => return Err(err()),
        };
    }

    h.inner_function_type().ok_or_else(err)
}

impl Value {
    /// Returns the type of this [`Value`].
    #[must_use]
    pub fn get_type(&self) -> Type {
        match self {
            Self::Extension { e } => e.get_type(),
            Self::Sum(Sum { sum_type, .. }) => sum_type.clone().into(),
            Self::Function { hugr } => {
                let func_type = mono_fn_type(hugr).unwrap_or_else(|e| panic!("{}", e));
                Type::new_function(func_type.into_owned())
            }
        }
    }

    /// Returns a Sum constant. The value is determined by `items` and is
    /// type-checked `typ`. The `tag`th variant of `typ` should match the types
    /// of `items`.
    pub fn sum(
        tag: usize,
        items: impl IntoIterator<Item = Value>,
        typ: SumType,
    ) -> Result<Self, ConstTypeError> {
        let values: Vec<Value> = items.into_iter().collect();
        typ.check_type(tag, &values)?;
        Ok(Self::Sum(Sum {
            tag,
            values,
            sum_type: typ,
        }))
    }

    /// Returns a tuple constant of constant values.
    pub fn tuple(items: impl IntoIterator<Item = Value>) -> Self {
        let vs = items.into_iter().collect_vec();
        let tys = vs.iter().map(Self::get_type).collect_vec();

        Self::sum(0, vs, SumType::new_tuple(tys)).expect("Tuple type is valid")
    }

    /// Returns a constant function defined by a Hugr.
    ///
    /// # Errors
    ///
    /// Returns an error if the Hugr root node does not define a function.
    pub fn function(hugr: impl Into<Hugr>) -> Result<Self, ConstTypeError> {
        let hugr = hugr.into();
        mono_fn_type(&hugr)?;
        Ok(Self::Function {
            hugr: Box::new(hugr),
        })
    }

    /// Returns a constant unit type (empty Tuple).
    #[must_use]
    pub const fn unit() -> Self {
        Self::Sum(Sum {
            tag: 0,
            values: vec![],
            sum_type: SumType::Unit { size: 1 },
        })
    }

    /// Returns a constant Sum over units. Used as branching values.
    pub fn unit_sum(tag: usize, size: u8) -> Result<Self, ConstTypeError> {
        Self::sum(tag, [], SumType::Unit { size })
    }

    /// Returns a constant Sum over units, with only one variant.
    #[must_use]
    pub fn unary_unit_sum() -> Self {
        Self::unit_sum(0, 1).expect("0 < 1")
    }

    /// Returns a constant "true" value, i.e. the second variant of Sum((), ()).
    #[must_use]
    pub fn true_val() -> Self {
        Self::unit_sum(1, 2).expect("1 < 2")
    }

    /// Returns a constant "false" value, i.e. the first variant of Sum((), ()).
    #[must_use]
    pub fn false_val() -> Self {
        Self::unit_sum(0, 2).expect("0 < 2")
    }

    /// Returns an optional with some values. This is a Sum with two variants, the
    /// first being empty and the second being the values.
    pub fn some<V: Into<Value>>(values: impl IntoIterator<Item = V>) -> Self {
        let values: Vec<Value> = values.into_iter().map(Into::into).collect_vec();
        let value_types: Vec<Type> = values.iter().map(Value::get_type).collect_vec();
        let sum_type = SumType::new_option(value_types);
        Self::sum(1, values, sum_type).unwrap()
    }

    /// Returns an optional with no value. This is a Sum with two variants, the
    /// first being empty and the second being the value.
    pub fn none(value_types: impl Into<TypeRow>) -> Self {
        Self::sum(0, [], SumType::new_option(value_types)).unwrap()
    }

    /// Returns a constant `bool` value.
    ///
    /// see [`Value::true_val`] and [`Value::false_val`].
    #[must_use]
    pub fn from_bool(b: bool) -> Self {
        if b {
            Self::true_val()
        } else {
            Self::false_val()
        }
    }

    /// Returns a [`Value::Extension`] holding `custom_const`.
    pub fn extension(custom_const: impl CustomConst) -> Self {
        Self::Extension {
            e: OpaqueValue::new(custom_const),
        }
    }

    /// For a [Value] holding a [`CustomConst`], extract the `CustomConst` by downcasting.
    #[must_use]
    pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T> {
        if let Self::Extension { e } = self {
            e.v.downcast_ref()
        } else {
            None
        }
    }

    fn name(&self) -> OpName {
        match self {
            Self::Extension { e } => format!("const:custom:{}", e.name()),
            Self::Function { hugr: h } => {
                let Ok(t) = mono_fn_type(h) else {
                    panic!("HUGR root node isn't a valid function parent.");
                };
                format!("const:function:[{t}]")
            }
            Self::Sum(Sum {
                tag,
                values,
                sum_type,
            }) => {
                if sum_type.as_tuple().is_some() {
                    let names: Vec<_> = values.iter().map(Value::name).collect();
                    format!("const:seq:{{{}}}", names.iter().join(", "))
                } else {
                    format!("const:sum:{{tag:{tag}, vals:{values:?}}}")
                }
            }
        }
        .into()
    }

    /// Check the value.
    pub fn validate(&self) -> Result<(), ConstTypeError> {
        match self {
            Self::Extension { e } => Ok(e.value().validate()?),
            Self::Function { hugr } => {
                mono_fn_type(hugr)?;
                Ok(())
            }
            Self::Sum(Sum {
                tag,
                values,
                sum_type,
            }) => {
                sum_type.check_type(*tag, values)?;
                Ok(())
            }
        }
    }

    /// If value is a sum with a single row variant, return the row.
    #[must_use]
    pub fn as_tuple(&self) -> Option<&[Value]> {
        if let Self::Sum(sum) = self {
            sum.as_tuple()
        } else {
            None
        }
    }

    /// Hashes this value, if possible. [`Value::Extension`]s are hashable according
    /// to their implementation of [`TryHash`]; [`Value::Function`]s never are;
    /// [`Value::Sum`]s are if their contents are.
    pub fn try_hash<H: Hasher>(&self, st: &mut H) -> bool {
        match self {
            Value::Extension { e } => e.value().try_hash(&mut *st),
            Value::Function { .. } => false,
            Value::Sum(s) => s.try_hash(st),
        }
    }
}

impl<T> From<T> for Value
where
    T: CustomConst,
{
    fn from(value: T) -> Self {
        Self::extension(value)
    }
}

/// A unique identifier for a constant value.
pub type ValueName = SmolStr;

/// Slice of a [`ValueName`] constant value identifier.
pub type ValueNameRef = str;

#[cfg(test)]
pub(crate) mod test {
    use std::collections::HashSet;
    use std::sync::{Arc, Weak};

    use super::Value;
    use crate::builder::inout_sig;
    use crate::builder::test::simple_dfg_hugr;
    use crate::extension::PRELUDE;
    use crate::extension::prelude::{bool_t, usize_custom_t};
    use crate::extension::resolution::{
        ExtensionResolutionError, WeakExtensionRegistry, resolve_custom_type_extensions,
        resolve_typearg_extensions,
    };
    use crate::std_extensions::arithmetic::int_types::ConstInt;
    use crate::std_extensions::collections::array::{ArrayValue, array_type};
    use crate::std_extensions::collections::value_array::{VArrayValue, value_array_type};
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            ExtensionId,
            prelude::{ConstUsize, usize_t},
        },
        std_extensions::arithmetic::float_types::{ConstF64, float64_type},
        type_row,
        types::type_param::TypeArg,
        types::{Type, TypeBound, TypeRow},
    };
    use cool_asserts::assert_matches;
    use rstest::{fixture, rstest};

    use super::*;

    #[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
    /// A custom constant value used in testing
    pub(crate) struct CustomTestValue(pub CustomType);

    #[typetag::serde]
    impl CustomConst for CustomTestValue {
        fn name(&self) -> ValueName {
            format!("CustomTestValue({:?})", self.0).into()
        }

        fn update_extensions(
            &mut self,
            extensions: &WeakExtensionRegistry,
        ) -> Result<(), ExtensionResolutionError> {
            resolve_custom_type_extensions(&mut self.0, extensions)?;
            // This loop is redundant, but we use it to test the public
            // function.
            for arg in self.0.args_mut() {
                resolve_typearg_extensions(arg, extensions)?;
            }
            Ok(())
        }

        fn get_type(&self) -> Type {
            self.0.clone().into()
        }

        fn equal_consts(&self, other: &dyn CustomConst) -> bool {
            crate::ops::constant::downcast_equal_consts(self, other)
        }
    }

    /// A [`CustomSerialized`] encoding a [`float64_type()`] float constant used in testing.
    pub(crate) fn serialized_float(f: f64) -> Value {
        CustomSerialized::try_from_custom_const(ConstF64::new(f))
            .unwrap()
            .into()
    }

    /// Constructs a DFG hugr defining a sum constant, and returning the loaded value.
    #[test]
    fn test_sum() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![vec![usize_t(), float64_type()].into(), Type::EMPTY_TYPEROW];
        let pred_ty = SumType::new(pred_rows.clone());

        let mut b = DFGBuilder::new(inout_sig(
            type_row![],
            TypeRow::from(vec![pred_ty.clone().into()]),
        ))?;
        let usize_custom_t = usize_custom_t(&Arc::downgrade(&PRELUDE));
        let c = b.add_constant(Value::sum(
            0,
            [
                CustomTestValue(usize_custom_t.clone()).into(),
                ConstF64::new(5.1).into(),
            ],
            pred_ty.clone(),
        )?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w]).unwrap();

        let mut b = DFGBuilder::new(Signature::new(
            type_row![],
            TypeRow::from(vec![pred_ty.clone().into()]),
        ))?;
        let c = b.add_constant(Value::sum(1, [], pred_ty.clone())?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w]).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_sum() {
        let pred_ty = SumType::new([vec![usize_t(), float64_type()].into(), type_row![]]);

        let good_sum = const_usize();
        println!("{}", serde_json::to_string_pretty(&good_sum).unwrap());

        let good_sum =
            Value::sum(0, [const_usize(), serialized_float(5.1)], pred_ty.clone()).unwrap();
        println!("{}", serde_json::to_string_pretty(&good_sum).unwrap());

        let res = Value::sum(0, [], pred_ty.clone());
        assert_matches!(
            res,
            Err(ConstTypeError::SumType(SumTypeError::WrongVariantLength {
                tag: 0,
                expected: 2,
                found: 0
            }))
        );

        let res = Value::sum(4, [], pred_ty.clone());
        assert_matches!(
            res,
            Err(ConstTypeError::SumType(SumTypeError::InvalidTag {
                tag: 4,
                num_variants: 2
            }))
        );

        let res = Value::sum(0, [const_usize(), const_usize()], pred_ty);
        assert_matches!(
            res,
            Err(ConstTypeError::SumType(SumTypeError::InvalidValueType {
                tag: 0,
                index: 1,
                expected,
                found,
            })) if *expected == float64_type() && *found == const_usize()
        );
    }

    #[rstest]
    fn function_value(simple_dfg_hugr: Hugr) {
        let v = Value::function(simple_dfg_hugr).unwrap();

        let correct_type = Type::new_function(Signature::new_endo(vec![bool_t()]));

        assert_eq!(v.get_type(), correct_type);
        assert!(v.name().starts_with("const:function:"));
    }

    #[fixture]
    fn const_usize() -> Value {
        ConstUsize::new(257).into()
    }

    #[fixture]
    fn const_serialized_usize() -> Value {
        CustomSerialized::try_from_custom_const(ConstUsize::new(257))
            .unwrap()
            .into()
    }

    #[fixture]
    fn const_tuple() -> Value {
        Value::tuple([const_usize(), Value::true_val()])
    }

    /// Equivalent to [`const_tuple`], but uses a non-resolved opaque op for the usize element.
    #[fixture]
    fn const_tuple_serialized() -> Value {
        Value::tuple([const_serialized_usize(), Value::true_val()])
    }

    #[fixture]
    fn const_array_bool() -> Value {
        ArrayValue::new(bool_t(), [Value::true_val(), Value::false_val()]).into()
    }

    #[fixture]
    fn const_value_array_bool() -> Value {
        VArrayValue::new(bool_t(), [Value::true_val(), Value::false_val()]).into()
    }

    #[fixture]
    fn const_array_options() -> Value {
        let some_true = Value::some([Value::true_val()]);
        let none = Value::none(vec![bool_t()]);
        let elem_ty = SumType::new_option(vec![bool_t()]);
        ArrayValue::new(elem_ty.into(), [some_true, none]).into()
    }

    #[fixture]
    fn const_value_array_options() -> Value {
        let some_true = Value::some([Value::true_val()]);
        let none = Value::none(vec![bool_t()]);
        let elem_ty = SumType::new_option(vec![bool_t()]);
        VArrayValue::new(elem_ty.into(), [some_true, none]).into()
    }

    #[rstest]
    #[case(Value::unit(), Type::UNIT, "const:seq:{}")]
    #[case(const_usize(), usize_t(), "const:custom:ConstUsize(")]
    #[case(serialized_float(17.4), float64_type(), "const:custom:json:Object")]
    #[case(const_tuple(), Type::new_tuple(vec![usize_t(), bool_t()]), "const:seq:{")]
    #[case(const_array_bool(), array_type(2, bool_t()), "const:custom:array")]
    #[case(
        const_value_array_bool(),
        value_array_type(2, bool_t()),
        "const:custom:value_array"
    )]
    #[case(
        const_array_options(),
        array_type(2, SumType::new_option(vec![bool_t()]).into()),
        "const:custom:array"
    )]
    #[case(
        const_value_array_options(),
        value_array_type(2, SumType::new_option(vec![bool_t()]).into()),
        "const:custom:value_array"
    )]
    fn const_type(
        #[case] const_value: Value,
        #[case] expected_type: Type,
        #[case] name_prefix: &str,
    ) {
        assert_eq!(const_value.get_type(), expected_type);
        let name = const_value.name();
        assert!(
            name.starts_with(name_prefix),
            "{name} does not start with {name_prefix}"
        );
    }

    #[rstest]
    #[case(Value::unit(), Value::unit())]
    #[case(const_usize(), const_usize())]
    #[case(const_serialized_usize(), const_usize())]
    #[case(const_tuple_serialized(), const_tuple())]
    #[case(const_array_bool(), const_array_bool())]
    #[case(const_value_array_bool(), const_value_array_bool())]
    #[case(const_array_options(), const_array_options())]
    #[case(const_value_array_options(), const_value_array_options())]
    // Opaque constants don't get resolved into concrete types when running miri,
    // as the `typetag` machinery is not available.
    #[cfg_attr(miri, ignore)]
    fn const_serde_roundtrip(#[case] const_value: Value, #[case] expected_value: Value) {
        let serialized = serde_json::to_string(&const_value).unwrap();
        let deserialized: Value = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, expected_value);
    }

    #[rstest]
    fn const_custom_value(const_usize: Value, const_tuple: Value) {
        assert_eq!(
            const_usize.get_custom_value::<ConstUsize>(),
            Some(&ConstUsize::new(257))
        );
        assert_eq!(const_usize.get_custom_value::<ConstInt>(), None);
        assert_eq!(const_tuple.get_custom_value::<ConstUsize>(), None);
        assert_eq!(const_tuple.get_custom_value::<ConstInt>(), None);
    }

    #[test]
    fn test_json_const() {
        let ex_id: ExtensionId = "my_extension".try_into().unwrap();
        let typ_int = CustomType::new(
            "my_type",
            vec![TypeArg::BoundedNat(8)],
            ex_id.clone(),
            TypeBound::Copyable,
            // Dummy extension reference.
            &Weak::default(),
        );
        let json_const: Value = CustomSerialized::new(typ_int.clone(), 6.into()).into();
        let classic_t = Type::new_extension(typ_int.clone());
        assert_matches!(classic_t.least_upper_bound(), TypeBound::Copyable);
        assert_eq!(json_const.get_type(), classic_t);

        let typ_qb = CustomType::new(
            "my_type",
            vec![],
            ex_id,
            TypeBound::Copyable,
            &Weak::default(),
        );
        let t = Type::new_extension(typ_qb.clone());
        assert_ne!(json_const.get_type(), t);
    }

    #[rstest]
    fn hash_tuple(const_tuple: Value) {
        let vals = [
            Value::unit(),
            Value::true_val(),
            Value::false_val(),
            ConstUsize::new(13).into(),
            Value::tuple([ConstUsize::new(13).into()]),
            Value::tuple([ConstUsize::new(13).into(), ConstUsize::new(14).into()]),
            Value::tuple([ConstUsize::new(13).into(), ConstUsize::new(15).into()]),
            const_tuple,
        ];

        let num_vals = vals.len();
        let hashes = vals.map(|v| {
            let mut h = DefaultHasher::new();
            v.try_hash(&mut h).then_some(()).unwrap();
            h.finish()
        });
        assert_eq!(HashSet::from(hashes).len(), num_vals); // all distinct
    }

    #[test]
    fn unhashable_tuple() {
        let tup = Value::tuple([ConstUsize::new(5).into(), ConstF64::new(4.97).into()]);
        let mut h1 = DefaultHasher::new();
        let r = tup.try_hash(&mut h1);
        assert!(!r);

        // Check that didn't do anything, by checking the hasher behaves
        // just like one which never saw the tuple
        h1.write_usize(5);
        let mut h2 = DefaultHasher::new();
        h2.write_usize(5);
        assert_eq!(h1.finish(), h2.finish());
    }

    mod proptest {
        use super::super::{OpaqueValue, Sum};
        use crate::{
            ops::{Value, constant::CustomSerialized},
            std_extensions::arithmetic::int_types::ConstInt,
            std_extensions::collections::list::ListValue,
            types::{SumType, Type},
        };
        use ::proptest::{collection::vec, prelude::*};
        impl Arbitrary for OpaqueValue {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                // We intentionally do not include `ConstF64` because it does not
                // roundtrip serialize
                prop_oneof![
                    any::<ConstInt>().prop_map_into(),
                    any::<CustomSerialized>().prop_map_into()
                ]
                .prop_recursive(
                    3,  // No more than 3 branch levels deep
                    32, // Target around 32 total elements
                    3,  // Each collection is up to 3 elements long
                    |child_strat| {
                        (any::<Type>(), vec(child_strat, 0..3)).prop_map(|(typ, children)| {
                            Self::new(ListValue::new(
                                typ,
                                children.into_iter().map(|e| Value::Extension { e }),
                            ))
                        })
                    },
                )
                .boxed()
            }
        }

        impl Arbitrary for Value {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                use ::proptest::collection::vec;
                let leaf_strat = prop_oneof![
                    any::<OpaqueValue>().prop_map(|e| Self::Extension { e }),
                    crate::proptest::any_hugr().prop_map(|x| Value::function(x).unwrap())
                ];
                leaf_strat
                    .prop_recursive(
                        3,  // No more than 3 branch levels deep
                        32, // Target around 32 total elements
                        3,  // Each collection is up to 3 elements long
                        |element| {
                            prop_oneof![
                                vec(element.clone(), 0..3).prop_map(Self::tuple),
                                (
                                    any::<usize>(),
                                    vec(element.clone(), 0..3),
                                    any_with::<SumType>(1.into()) // for speed: don't generate large sum types for now
                                )
                                    .prop_map(
                                        |(tag, values, sum_type)| {
                                            Self::Sum(Sum {
                                                tag,
                                                values,
                                                sum_type,
                                            })
                                        }
                                    ),
                            ]
                        },
                    )
                    .boxed()
            }
        }
    }

    #[test]
    fn test_tuple_deserialize() {
        let json = r#"
        {
    "v": "Tuple",
    "vs": [
        {
            "v": "Sum",
            "tag": 0,
            "typ": {
                "t": "Sum",
                "s": "Unit",
                "size": 1
            },
            "vs": []
        },
        {
            "v": "Sum",
            "tag": 1,
            "typ": {
                "t": "Sum",
                "s": "General",
                "rows": [
                    [
                        {
                            "t": "Sum",
                            "s": "Unit",
                            "size": 1
                        }
                    ],
                    [
                        {
                            "t": "Sum",
                            "s": "Unit",
                            "size": 2
                        }
                    ]
                ]
            },
            "vs": [
                {
                    "v": "Sum",
                    "tag": 1,
                    "typ": {
                        "t": "Sum",
                        "s": "Unit",
                        "size": 2
                    },
                    "vs": []
                }
            ]
        }
    ]
}
        "#;

        let v: Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            v,
            Value::tuple([
                Value::unit(),
                Value::sum(
                    1,
                    [Value::true_val()],
                    SumType::new([
                        type_row![Type::UNIT],
                        vec![Value::true_val().get_type()].into()
                    ]),
                )
                .unwrap()
            ])
        );
    }
}
