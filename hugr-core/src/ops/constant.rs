//! Constant value definitions.

mod custom;

use super::{NamedOp, OpName, OpTrait, StaticTag};
use super::{OpTag, OpType};
use crate::extension::ExtensionSet;
use crate::types::{CustomType, EdgeKind, FunctionType, SumType, SumTypeError, Type};
use crate::{Hugr, HugrView};

use delegate::delegate;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;
use thiserror::Error;

pub use custom::{
    downcast_equal_consts, get_pair_of_input_values, get_single_input_value, CustomConst,
    CustomSerialized,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// An operation returning a constant value.
///
/// Represents core types and extension types.
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Const {
    #[serde(rename = "v")]
    value: Value,
}

impl Const {
    /// Create a new [`Const`] operation.
    pub fn new(value: Value) -> Self {
        Self { value }
    }

    /// The inner value of the [`Const`]
    pub fn value(&self) -> &Value {
        &self.value
    }

    delegate! {
        to self.value {
            /// Returns the type of this constant.
            pub fn get_type(&self) -> Type;
            /// For a Const holding a CustomConst, extract the CustomConst by
            /// downcasting.
            pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T>;
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
    fn description(&self) -> &str {
        "Constant value"
    }

    fn extension_delta(&self) -> ExtensionSet {
        self.value().extension_reqs()
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Const(self.get_type()))
    }
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    // TODO use a root parametrised hugr, e.g. Hugr<DFG>.
    Function {
        /// A Hugr defining the function.
        hugr: Box<Hugr>,
    },
    /// A tuple
    Tuple {
        /// Constant values in the tuple.
        vs: Vec<Value>,
    },
    /// A Sum variant, with a tag indicating the index of the variant and its
    /// value.
    Sum {
        /// The tag index of the variant.
        tag: usize,
        /// The value of the variant.
        ///
        /// Sum variants are always a row of values, hence the Vec.
        #[serde(rename = "vs")]
        values: Vec<Value>,
        /// The full type of the Sum, including the other variants.
        #[serde(rename = "typ")]
        sum_type: SumType,
    },
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        let cmp_tuple_sum =
            |tuple_values: &[Value], tag: usize, sum_values: &[Value], sum_type: &SumType| {
                tag == 0 && sum_type.num_variants() == 1 && tuple_values == sum_values
            };
        match (self, other) {
            (Self::Extension { e: e1 }, Self::Extension { e: e2 }) => e1 == e2,
            (Self::Function { hugr: h1 }, Self::Function { hugr: h2 }) => h1 == h2,
            (Self::Tuple { vs: v1 }, Self::Tuple { vs: v2 }) => v1 == v2,
            (
                Self::Sum {
                    tag: t1,
                    values: v1,
                    sum_type: s1,
                },
                Self::Sum {
                    tag: t2,
                    values: v2,
                    sum_type: s2,
                },
            ) => t1 == t2 && v1 == v2 && s1 == s2,
            (
                Self::Tuple { vs },
                Self::Sum {
                    tag,
                    values,
                    sum_type,
                },
            )
            | (
                Self::Sum {
                    tag,
                    values,
                    sum_type,
                },
                Self::Tuple { vs },
            ) => cmp_tuple_sum(vs, *tag, values, sum_type),
            _ => false,
        }
    }
}

/// An opaque newtype around a [`Box<dyn CustomConst>`](CustomConst).
///
/// This type has special serialization behaviour in order to support
/// serialisation and deserialisation of unknown impls of [CustomConst].
///
/// During serialization we first serialize the internal [`dyn` CustomConst](CustomConst)
/// into a [serde_yaml::Value]. We then create a [CustomSerialized] wrapping
/// that value.  That [CustomSerialized] is then serialized in place of the
/// [OpaqueValue].
///
/// During deserialization, first we deserialize a [CustomSerialized]. We
/// attempt to deserialize the internal [serde_yaml::Value] using the [`Box<dyn
/// CustomConst>`](CustomConst) impl. This will fail if the appropriate `impl CustomConst`
/// is not linked into the running program, in which case we coerce the
/// [CustomSerialized] into a [`Box<dyn CustomConst>`](CustomConst). The [OpaqueValue] is
/// then produced from the [`Box<dyn [CustomConst]>`](CustomConst).
///
/// In the case where the internal serialised value of a `CustomSerialized`
/// is another `CustomSerialized` we do not attempt to recurse. This behaviour
/// may change in future.
///
#[cfg_attr(not(miri), doc = "```")] // this doctest depends on typetag, so fails with miri
#[cfg_attr(miri, doc = "```ignore")]
/// use serde::{Serialize,Deserialize};
/// use hugr::{
///   types::Type,ops::constant::{OpaqueValue, ValueName, CustomConst, CustomSerialized},
///   extension::{ExtensionSet, prelude::{USIZE_T, ConstUsize}},
///   std_extensions::arithmetic::int_types};
/// use serde_json::json;
///
/// let expected_json = json!({
///     "extensions": ["prelude"],
///     "typ": USIZE_T,
///     "value": {'c': "ConstUsize", 'v': 1}
/// });
/// let ev = OpaqueValue::new(ConstUsize::new(1));
/// assert_eq!(&serde_json::to_value(&ev).unwrap(), &expected_json);
/// assert_eq!(ev, serde_json::from_value(expected_json).unwrap());
///
/// let ev = OpaqueValue::new(CustomSerialized::new(USIZE_T.clone(), serde_yaml::Value::Null, ExtensionSet::default()));
/// let expected_json = json!({
///     "extensions": [],
///     "typ": USIZE_T,
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
    pub fn value(&self) -> &dyn CustomConst {
        self.v.as_ref()
    }

    delegate! {
        to self.value() {
            /// Returns the type of the internal [`CustomConst`].
            pub fn get_type(&self) -> Type;
            /// An identifier of the internal [`CustomConst`].
            pub fn name(&self) -> ValueName;
            /// The extension(s) defining the internal [`CustomConst`].
            pub fn extension_reqs(&self) -> ExtensionSet;
        }
    }
}

impl<CC: CustomConst> From<CC> for OpaqueValue {
    fn from(x: CC) -> Self {
        Self::new(x)
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
        expected: CustomType,
        /// The custom type found when checking.
        found: Type,
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
        "A function constant cannot be defined using a Hugr with root of type {hugr_root_type:?}. Must be a monomorphic function.",
    )]
    NotMonomorphicFunction {
        /// The root node type of the Hugr that (claims to) define the function constant.
        hugr_root_type: OpType,
    },
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ConstCheckFail(Type, Value),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFailure),
}

/// Hugrs (even functions) inside Consts must be monomorphic
fn mono_fn_type(h: &Hugr) -> Result<FunctionType, ConstTypeError> {
    if let Some(pf) = h.get_function_type() {
        if let Ok(ft) = pf.try_into() {
            return Ok(ft);
        }
    }
    Err(ConstTypeError::NotMonomorphicFunction {
        hugr_root_type: h.root_type().op().clone(),
    })
}

impl Value {
    /// Returns the type of this [`Value`].
    pub fn get_type(&self) -> Type {
        match self {
            Self::Extension { e } => e.get_type(),
            Self::Tuple { vs } => Type::new_tuple(vs.iter().map(Self::get_type).collect_vec()),
            Self::Sum { sum_type, .. } => sum_type.clone().into(),
            Self::Function { hugr } => {
                let func_type = mono_fn_type(hugr).unwrap_or_else(|e| panic!("{}", e));
                Type::new_function(func_type)
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
        Ok(Self::Sum {
            tag,
            values,
            sum_type: typ,
        })
    }

    /// Returns a tuple constant of constant values.
    pub fn tuple(items: impl IntoIterator<Item = Value>) -> Self {
        Self::Tuple {
            vs: items.into_iter().collect(),
        }
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
    pub const fn unit() -> Self {
        Self::Tuple { vs: vec![] }
    }

    /// Returns a constant Sum over units. Used as branching values.
    pub fn unit_sum(tag: usize, size: u8) -> Result<Self, ConstTypeError> {
        Self::sum(tag, [], SumType::Unit { size })
    }

    /// Returns a constant Sum over units, with only one variant.
    pub fn unary_unit_sum() -> Self {
        Self::unit_sum(0, 1).expect("0 < 1")
    }

    /// Returns a constant "true" value, i.e. the second variant of Sum((), ()).
    pub fn true_val() -> Self {
        Self::unit_sum(1, 2).expect("1 < 2")
    }

    /// Returns a constant "false" value, i.e. the first variant of Sum((), ()).
    pub fn false_val() -> Self {
        Self::unit_sum(0, 2).expect("0 < 2")
    }

    /// Returns a constant `bool` value.
    ///
    /// see [`Value::true_val`] and [`Value::false_val`].
    pub fn from_bool(b: bool) -> Self {
        if b {
            Self::true_val()
        } else {
            Self::false_val()
        }
    }

    /// Returns a [Value::Extension] holding `custom_const`.
    pub fn extension(custom_const: impl CustomConst) -> Self {
        Self::Extension {
            e: OpaqueValue::new(custom_const),
        }
    }

    /// For a [Value] holding a [CustomConst], extract the CustomConst by downcasting.
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
                let Some(t) = h.get_function_type() else {
                    panic!("HUGR root node isn't a valid function parent.");
                };
                format!("const:function:[{}]", t)
            }
            Self::Tuple { vs: vals } => {
                let names: Vec<_> = vals.iter().map(Value::name).collect();
                format!("const:seq:{{{}}}", names.iter().join(", "))
            }
            Self::Sum { tag, values, .. } => {
                format!("const:sum:{{tag:{tag}, vals:{values:?}}}")
            }
        }
        .into()
    }

    /// The extensions required by a [`Value`]
    pub fn extension_reqs(&self) -> ExtensionSet {
        match self {
            Self::Extension { e } => e.extension_reqs().clone(),
            Self::Function { .. } => ExtensionSet::new(), // no extensions required to load Hugr (only to run)
            Self::Tuple { vs } => ExtensionSet::union_over(vs.iter().map(Value::extension_reqs)),
            Self::Sum { values, .. } => {
                ExtensionSet::union_over(values.iter().map(|x| x.extension_reqs()))
            }
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
mod test {
    use super::Value;
    use crate::builder::test::simple_dfg_hugr;
    use crate::std_extensions::arithmetic::int_types::ConstInt;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            prelude::{ConstUsize, USIZE_CUSTOM_T, USIZE_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        type_row,
        types::type_param::TypeArg,
        types::{Type, TypeBound, TypeRow},
    };
    use cool_asserts::assert_matches;
    use rstest::{fixture, rstest};
    use serde_yaml::Value as YamlValue;

    use super::*;

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    /// A custom constant value used in testing
    pub(crate) struct CustomTestValue(pub CustomType);

    #[typetag::serde]
    impl CustomConst for CustomTestValue {
        fn name(&self) -> ValueName {
            format!("CustomTestValue({:?})", self.0).into()
        }

        fn extension_reqs(&self) -> ExtensionSet {
            ExtensionSet::singleton(self.0.extension())
        }

        fn get_type(&self) -> Type {
            self.0.clone().into()
        }

        fn equal_consts(&self, other: &dyn CustomConst) -> bool {
            crate::ops::constant::downcast_equal_consts(self, other)
        }
    }

    /// A [`CustomSerialized`] encoding a [`FLOAT64_TYPE`] float constant used in testing.
    pub(crate) fn serialized_float(f: f64) -> Value {
        CustomSerialized::try_from_custom_const(ConstF64::new(f))
            .unwrap()
            .into()
    }

    fn test_registry() -> ExtensionRegistry {
        ExtensionRegistry::try_new([PRELUDE.to_owned(), float_types::EXTENSION.to_owned()]).unwrap()
    }

    /// Constructs a DFG hugr defining a sum constant, and returning the loaded value.
    #[test]
    fn test_sum() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![type_row![USIZE_T, FLOAT64_TYPE], Type::EMPTY_TYPEROW];
        let pred_ty = SumType::new(pred_rows.clone());

        let mut b = DFGBuilder::new(FunctionType::new(
            type_row![],
            TypeRow::from(vec![pred_ty.clone().into()]),
        ))?;
        let c = b.add_constant(Value::sum(
            0,
            [
                CustomTestValue(USIZE_CUSTOM_T).into(),
                ConstF64::new(5.1).into(),
            ],
            pred_ty.clone(),
        )?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w], &test_registry()).unwrap();

        let mut b = DFGBuilder::new(FunctionType::new(
            type_row![],
            TypeRow::from(vec![pred_ty.clone().into()]),
        ))?;
        let c = b.add_constant(Value::sum(1, [], pred_ty.clone())?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w], &test_registry()).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_sum() {
        let pred_ty = SumType::new([type_row![USIZE_T, FLOAT64_TYPE], type_row![]]);

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
            })) if expected == FLOAT64_TYPE && found == const_usize()
        );
    }

    #[rstest]
    fn function_value(simple_dfg_hugr: Hugr) {
        let v = Value::function(simple_dfg_hugr).unwrap();

        let correct_type = Type::new_function(FunctionType::new_endo(type_row![
            crate::extension::prelude::BOOL_T
        ]));

        assert_eq!(v.get_type(), correct_type);
        assert!(v.name().starts_with("const:function:"))
    }

    #[fixture]
    fn const_usize() -> Value {
        ConstUsize::new(257).into()
    }

    #[fixture]
    fn const_tuple() -> Value {
        Value::tuple([ConstUsize::new(257).into(), serialized_float(5.1)])
    }

    #[rstest]
    #[case(Value::unit(), Type::UNIT, "const:seq:{}")]
    #[case(const_usize(), USIZE_T, "const:custom:ConstUsize(")]
    #[case(serialized_float(17.4), FLOAT64_TYPE, "const:custom:yaml:Mapping")]
    #[case(const_tuple(), Type::new_tuple(type_row![USIZE_T, FLOAT64_TYPE]), "const:seq:{")]
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
    fn test_yaml_const() {
        let ex_id: ExtensionId = "my_extension".try_into().unwrap();
        let typ_int = CustomType::new(
            "my_type",
            vec![TypeArg::BoundedNat { n: 8 }],
            ex_id.clone(),
            TypeBound::Eq,
        );
        let yaml_const: Value =
            CustomSerialized::new(typ_int.clone(), YamlValue::Number(6.into()), ex_id.clone())
                .into();
        let classic_t = Type::new_extension(typ_int.clone());
        assert_matches!(classic_t.least_upper_bound(), TypeBound::Eq);
        assert_eq!(yaml_const.get_type(), classic_t);

        let typ_qb = CustomType::new("my_type", vec![], ex_id, TypeBound::Eq);
        let t = Type::new_extension(typ_qb.clone());
        assert_ne!(yaml_const.get_type(), t);
    }

    mod proptest {
        use super::super::OpaqueValue;
        use crate::{
            ops::{constant::CustomSerialized, Value},
            std_extensions::arithmetic::int_types::ConstInt,
            std_extensions::collections::ListValue,
            types::{SumType, Type},
        };
        use ::proptest::{collection::vec, prelude::*};
        impl Arbitrary for OpaqueValue {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
                // We intentionally do not include `ConstF64` because it does not
                // roundtrip serialise
                prop_oneof![
                    any::<ConstInt>().prop_map_into(),
                    any::<CustomSerialized>().prop_map_into()
                ]
                .prop_recursive(
                    3,  // No more than 3 branch levels deep
                    32, // Target around 32 total elements
                    3,  // Each collection is up to 3 elements long
                    |child_strat| {
                        (Type::any_non_row_var(), vec(child_strat, 0..3)).prop_map(
                            |(typ, children)| {
                                Self::new(ListValue::new(
                                    typ,
                                    children.into_iter().map(|e| Value::Extension { e }),
                                ))
                            },
                        )
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
                                vec(element.clone(), 0..3).prop_map(|vs| Self::Tuple { vs }),
                                (
                                    any::<usize>(),
                                    vec(element.clone(), 0..3),
                                    any_with::<SumType>(1.into()) // for speed: don't generate large sum types for now
                                )
                                    .prop_map(
                                        |(tag, values, sum_type)| {
                                            Self::Sum {
                                                tag,
                                                values,
                                                sum_type,
                                            }
                                        }
                                    ),
                            ]
                        },
                    )
                    .boxed()
            }
        }
    }
}
