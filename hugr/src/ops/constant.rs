//! Constant value definitions.

mod custom;

use super::{NamedOp, OpName, OpTrait, StaticTag};
use super::{OpTag, OpType};
use crate::extension::ExtensionSet;
use crate::types::{CustomType, EdgeKind, FunctionType, SumType, SumTypeError, Type};
use crate::{Hugr, HugrView};

use itertools::Itertools;
use smol_str::SmolStr;
use thiserror::Error;

pub use custom::{downcast_equal_consts, CustomConst, CustomSerialized};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// An operation returning a constant value.
///
/// Represents core types and extension types.
#[non_exhaustive]
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

    /// Returns a reference to the type of this constant.
    pub fn const_type(&self) -> Type {
        self.value.const_type()
    }
}

impl From<Value> for Const {
    fn from(value: Value) -> Self {
        Self::new(value)
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

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "c")]
/// A value that can be stored as a static constant. Representing core types and
/// extension types.
pub enum Value {
    /// An extension constant value, that can check it is of a given [CustomType].
    Extension {
        /// The custom constant value.
        e: ExtensionValue,
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

/// Boxed [`CustomConst`] trait object.
///
/// Use [`Value::extension`] to create a new variant of this type.
///
/// This is required to avoid <https://github.com/rust-lang/rust/issues/78808> in
/// [`Value::Extension`], while implementing a transparent encoding into a
/// `CustomConst`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct ExtensionValue(pub(super) Box<dyn CustomConst>);

impl PartialEq for ExtensionValue {
    fn eq(&self, other: &Self) -> bool {
        self.0.equal_consts(other.0.as_ref())
    }
}

impl ExtensionValue {
    /// TODO
    pub fn new(konst: impl CustomConst) -> Self {
        Self(Box::new(konst))
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
    /// Returns a reference to the type of this [`Value`].
    pub fn const_type(&self) -> Type {
        match self {
            Self::Extension { e } => e.0.get_type(),
            Self::Tuple { vs } => Type::new_tuple(vs.iter().map(Self::const_type).collect_vec()),
            Self::Sum { sum_type, .. } => sum_type.clone().into(),
            Self::Function { hugr } => {
                let func_type = mono_fn_type(hugr).unwrap_or_else(|e| panic!("{}", e));
                Type::new_function(func_type)
            }
        }
    }

    /// Creates a new Const Sum.  The value is determined by `items` and is
    /// type-checked `typ`
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

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> Self {
        Self::Tuple { vs: vec![] }
    }

    /// Constant Sum over units, used as branching values.
    pub fn unit_sum(tag: usize, size: u8) -> Result<Self, ConstTypeError> {
        Self::sum(tag, [], SumType::Unit { size })
    }

    /// Constant Sum over units, with only one variant.
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

    /// Generate a constant equivalent of a boolean,
    /// see [`Value::true_val`] and [`Value::false_val`].
    pub fn from_bool(b: bool) -> Self {
        if b {
            Self::true_val()
        } else {
            Self::false_val()
        }
    }

    /// Returns a tuple constant of constant values.
    pub fn extension(custom_const: impl CustomConst) -> Self {
        Self::Extension {
            e: ExtensionValue(Box::new(custom_const)),
        }
    }

    /// For a Const holding a CustomConst, extract the CustomConst by downcasting.
    pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T> {
        if let Self::Extension { e } = self {
            e.0.downcast_ref()
        } else {
            None
        }
    }

    fn name(&self) -> OpName {
        match self {
            Self::Extension { e } => format!("const:custom:{}", e.0.name()),
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
            Self::Extension { e } => e.0.extension_reqs().clone(),
            Self::Function { .. } => ExtensionSet::new(), // no extensions required to load Hugr (only to run)
            Self::Tuple { vs } => ExtensionSet::union_over(vs.iter().map(Value::extension_reqs)),
            Self::Sum { values, .. } => {
                ExtensionSet::union_over(values.iter().map(|x| x.extension_reqs()))
            }
        }
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
        Some(EdgeKind::Const(self.const_type()))
    }
}

// [KnownTypeConst] is guaranteed to be the right type, so can be constructed
// without initial type check.
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
    use super::{Value, ValueName};
    use crate::builder::test::simple_dfg_hugr;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            prelude::{ConstUsize, USIZE_CUSTOM_T, USIZE_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        std_extensions::arithmetic::int_types::{ConstInt, LOG_WIDTH_MAX},
        std_extensions::collections::ListValue,
        type_row,
        types::type_param::TypeArg,
        types::{Type, TypeBound, TypeRow},
    };
    use cool_asserts::assert_matches;
    use rstest::{fixture, rstest};
    use serde_yaml::Value as YamlValue;

    use super::*;
    use proptest::prelude::*;

    impl Arbitrary for ExtensionValue {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            use proptest::collection::vec;
            let signed_strat = (..=LOG_WIDTH_MAX).prop_flat_map(|log_width| {
                use std::i64;
                let max_val = (2u64.pow(log_width as u32) / 2) as i64;
                let min_val = -max_val - 1;
                (min_val..=max_val).prop_map(move |v| {
                    ExtensionValue::new(
                        ConstInt::new_s(log_width, v).expect("guaranteed to be in bounds"),
                    )
                })
            });
            let unsigned_strat = (..=LOG_WIDTH_MAX).prop_flat_map(|log_width| {
                (0..2u64.pow(log_width as u32)).prop_map(move |v| {
                    ExtensionValue::new(
                        ConstInt::new_u(log_width, v).expect("guaranteed to be in bounds"),
                    )
                })
            });
            prop_oneof![unsigned_strat, signed_strat]
                .prop_recursive(
                    3,  // No more than 3 branch levels deep
                    32, // Target around 32 total elements
                    3,  // Each collection is up to 3 elements long
                    |element| {
                        (any::<Type>(), vec(element.clone(), 0..3)).prop_map(|(typ, contents)| {
                            ExtensionValue::new(ListValue::new(
                                typ,
                                contents.into_iter().map(|e| Value::Extension { e }),
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
            use proptest::collection::vec;
            let leaf_strat =
                prop_oneof![
                    any::<ExtensionValue>().prop_map(|e| Self::Extension { e }),
                    prop_oneof![
                        // TODO we need an example of each legal root, in particular FuncDe{fn,cl}
                        Just(crate::builder::test::simple_dfg_hugr()),
                    ].prop_map(|x| Value::function(x).unwrap())
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
                                any_with::<SumType>(1.into())
                            )
                                .prop_map(|(tag, values, sum_type)| {
                                    Self::Sum {
                                        tag,
                                        values,
                                        sum_type,
                                    }
                                }),
                        ]
                    },
                )
                .boxed()
        }
    }

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
        CustomSerialized::new(
            FLOAT64_TYPE,
            serde_yaml::Value::Number(f.into()),
            float_types::EXTENSION_ID,
        )
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
                serialized_float(5.1),
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

        assert_eq!(v.const_type(), correct_type);
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
    #[case(serialized_float(17.4), FLOAT64_TYPE, "const:custom:yaml:Number(17.4)")]
    #[case(const_tuple(), Type::new_tuple(type_row![USIZE_T, FLOAT64_TYPE]), "const:seq:{")]
    fn const_type(
        #[case] const_value: Value,
        #[case] expected_type: Type,
        #[case] name_prefix: &str,
    ) {
        assert_eq!(const_value.const_type(), expected_type);
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
        assert_eq!(const_usize.get_custom_value::<ConstF64>(), None);
        assert_eq!(const_tuple.get_custom_value::<ConstUsize>(), None);
        assert_eq!(const_tuple.get_custom_value::<ConstF64>(), None);
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
        assert_eq!(yaml_const.const_type(), classic_t);

        let typ_qb = CustomType::new("my_type", vec![], ex_id, TypeBound::Eq);
        let t = Type::new_extension(typ_qb.clone());
        assert_ne!(yaml_const.const_type(), t);
    }
}
