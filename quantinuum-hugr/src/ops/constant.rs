//! Constant value definitions.

use super::{OpName, OpTrait, StaticTag};
use super::{OpTag, OpType};
use crate::extension::ExtensionSet;
use crate::types::{CustomType, EdgeKind, SumType, SumTypeError, Type};
use crate::values::CustomConst;
use crate::{Hugr, HugrView};

use itertools::Itertools;
use smol_str::SmolStr;
use thiserror::Error;

/// An operation returning a constant value.
///
/// Represents core types and extension types.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "v")]
pub enum Const {
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
        vs: Vec<Const>,
    },
    /// A Sum variant, with a tag indicating the index of the variant and its
    /// value.
    Sum {
        /// The tag index of the variant.
        tag: usize,
        /// The value of the variant.
        ///
        /// Sum variants are always a row of values, hence the Vec.
        values: Vec<Const>,
        /// The full type of the Sum, including the other variants.
        #[serde(rename = "typ")]
        sum_type: SumType,
    },
}

/// Struct for custom type check fails.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
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
pub enum ConstTypeError {
    /// Invalid sum type definition.
    #[error("{0}")]
    SumType(#[from] SumTypeError),
    /// Function constant missing a function type.
    #[error(
        "A function constant cannot be defined using a Hugr with root of type {}.",
        .hugr_root_type.name()
    )]
    FunctionTypeMissing {
        /// The root node type of the Hugr defining the function constant.
        hugr_root_type: OpType,
    },
    /// A mismatch between the type expected and the value.
    #[error("Value {1:?} does not match expected type {0}")]
    ConstCheckFail(Type, Const),
    /// Error when checking a custom value.
    #[error("Error when checking custom type: {0:?}")]
    CustomCheckFail(#[from] CustomCheckFailure),
}

impl Const {
    /// Returns a reference to the type of this [`Const`].
    pub fn const_type(&self) -> Type {
        match self {
            Self::Extension { c: (e,) } => e.get_type(),
            Self::Tuple { vs } => Type::new_tuple(vs.iter().map(Self::const_type).collect_vec()),
            Self::Sum { sum_type, .. } => sum_type.clone().into(),
            Self::Function { hugr } => {
                let func_type = hugr.get_function_type().unwrap_or_else(|| {
                    panic!(
                        "{}",
                        ConstTypeError::FunctionTypeMissing {
                            hugr_root_type: hugr.get_optype(hugr.root()).clone()
                        }
                    )
                });
                Type::new_function(func_type)
            }
        }
    }

    /// Creates a new Const Sum.  The value is determined by `items` and is
    /// type-checked `typ`
    pub fn sum(
        tag: usize,
        items: impl IntoIterator<Item = Const>,
        typ: SumType,
    ) -> Result<Self, ConstTypeError> {
        let values: Vec<Const> = items.into_iter().collect();
        typ.check_type(tag, &values)?;
        Ok(Self::Sum {
            tag,
            values,
            sum_type: typ,
        })
    }

    /// Returns a tuple constant of constant values.
    pub fn tuple(items: impl IntoIterator<Item = Const>) -> Self {
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
        if hugr.get_function_type().is_none() {
            Err(ConstTypeError::FunctionTypeMissing {
                hugr_root_type: hugr.get_optype(hugr.root()).clone(),
            })?;
        }
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
    /// see [`Const::true_val`] and [`Const::false_val`].
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
            c: (Box::new(custom_const),),
        }
    }

    /// For a Const holding a CustomConst, extract the CustomConst by downcasting.
    pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T> {
        if let Self::Extension { c: (custom,) } = self {
            custom.downcast_ref()
        } else {
            None
        }
    }
}

impl OpName for Const {
    fn name(&self) -> SmolStr {
        match self {
            Self::Extension { c: e } => format!("const:custom:{}", e.0.name()),
            Self::Function { hugr: h } => {
                let Some(t) = h.get_function_type() else {
                    panic!("HUGR root node isn't a valid function parent.");
                };
                format!("const:function:[{}]", t)
            }
            Self::Tuple { vs: vals } => {
                let names: Vec<_> = vals.iter().map(Self::name).collect();
                format!("const:seq:{{{}}}", names.join(", "))
            }
            Self::Sum { tag, values, .. } => {
                format!("const:sum:{{tag:{tag}, vals:{values:?}}}")
            }
        }
        .into()
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
        match self {
            Self::Extension { c } => c.0.extension_reqs().clone(),
            Self::Function { .. } => ExtensionSet::new(), // no extensions required to load Hugr (only to run)
            Self::Tuple { vs } => ExtensionSet::union_over(vs.iter().map(Const::extension_delta)),
            Self::Sum { values, .. } => {
                ExtensionSet::union_over(values.iter().map(|x| x.extension_delta()))
            }
        }
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(self.const_type()))
    }
}

// [KnownTypeConst] is guaranteed to be the right type, so can be constructed
// without initial type check.
impl<T> From<T> for Const
where
    T: CustomConst,
{
    fn from(value: T) -> Self {
        Self::extension(value)
    }
}

#[cfg(test)]
mod test {
    use super::Const;
    use crate::builder::test::simple_dfg_hugr;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            prelude::{ConstUsize, USIZE_CUSTOM_T, USIZE_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        type_row,
        types::type_param::TypeArg,
        types::{CustomType, FunctionType, Type, TypeBound, TypeRow},
        values::{
            test::{serialized_float, CustomTestValue},
            CustomSerialized,
        },
    };
    use cool_asserts::assert_matches;
    use rstest::{fixture, rstest};
    use serde_yaml::Value as YamlValue;

    use super::*;

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
        let c = b.add_constant(Const::sum(
            0,
            [
                Into::<Const>::into(CustomTestValue(USIZE_CUSTOM_T)),
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
        let c = b.add_constant(Const::sum(1, [], pred_ty.clone())?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w], &test_registry()).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_sum() {
        let pred_ty = SumType::new([type_row![USIZE_T, FLOAT64_TYPE], type_row![]]);

        let res = Const::sum(0, [], pred_ty.clone());
        assert_matches!(
            res,
            Err(ConstTypeError::SumType(SumTypeError::WrongVariantLength {
                tag: 0,
                expected: 2,
                found: 0
            }))
        );

        let res = Const::sum(4, [], pred_ty.clone());
        assert_matches!(
            res,
            Err(ConstTypeError::SumType(SumTypeError::InvalidTag {
                tag: 4,
                num_variants: 2
            }))
        );

        let res = Const::sum(0, [const_usize(), const_usize()], pred_ty);
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
        let v = Const::Function {
            hugr: Box::new(simple_dfg_hugr),
        };

        let correct_type = Type::new_function(FunctionType::new_endo(type_row![
            crate::extension::prelude::BOOL_T
        ]));

        assert_eq!(v.const_type(), correct_type);
    }

    #[fixture]
    fn const_usize() -> Const {
        ConstUsize::new(257).into()
    }

    #[fixture]
    fn const_tuple() -> Const {
        Const::tuple([ConstUsize::new(257).into(), serialized_float(5.1)])
    }

    #[rstest]
    #[case(const_usize(), USIZE_T)]
    #[case(serialized_float(17.4), FLOAT64_TYPE)]
    #[case(const_tuple(), Type::new_tuple(type_row![USIZE_T, FLOAT64_TYPE]))]
    fn const_type(#[case] const_value: Const, #[case] expected_type: Type) {
        assert_eq!(const_value.const_type(), expected_type);
    }

    #[rstest]
    fn const_custom_value(const_usize: Const, const_tuple: Const) {
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
        let yaml_const: Const =
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
