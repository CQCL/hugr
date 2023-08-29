//! Constant value definitions.

use crate::{
    types::{ConstTypeError, EdgeKind, Type, TypeRow},
    values::{CustomConst, KnownTypeConst, Value},
};

use smol_str::SmolStr;

use super::OpTag;
use super::{OpName, OpTrait, StaticTag};

/// A constant value definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Const {
    value: Value,
    typ: Type,
}

impl Const {
    /// Creates a new Const, type-checking the value.
    pub fn new(value: Value, typ: Type) -> Result<Self, ConstTypeError> {
        typ.check_type(&value)?;
        Ok(Self { value, typ })
    }

    /// Returns a reference to the value of this [`Const`].
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// Returns a reference to the type of this [`Const`].
    pub fn const_type(&self) -> &Type {
        &self.typ
    }

    /// Sum of Tuples, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn predicate(
        tag: usize,
        value: Value,
        variant_rows: impl IntoIterator<Item = TypeRow>,
    ) -> Result<Self, ConstTypeError> {
        let typ = Type::new_predicate(variant_rows);
        Self::new(Value::sum(tag, value), typ)
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize, size: u8) -> Self {
        Self {
            value: Value::simple_predicate(tag),
            typ: Type::new_simple_predicate(size),
        }
    }

    /// Constant Sum over units, with only one variant.
    pub fn simple_unary_predicate() -> Self {
        Self {
            value: Value::simple_unary_predicate(),
            typ: Type::new_simple_predicate(1),
        }
    }

    /// Constant "true" value, i.e. the second variant of Sum((), ()).
    pub fn true_val() -> Self {
        Self::simple_predicate(1, 2)
    }

    /// Constant "false" value, i.e. the first variant of Sum((), ()).
    pub fn false_val() -> Self {
        Self::simple_predicate(0, 2)
    }

    /// Tuple of values
    pub fn new_tuple(items: impl IntoIterator<Item = Const>) -> Self {
        let (values, types): (Vec<Value>, Vec<Type>) = items
            .into_iter()
            .map(|Const { value, typ }| (value, typ))
            .unzip();
        Self::new(Value::tuple(values), Type::new_tuple(types)).unwrap()
    }
}

impl OpName for Const {
    fn name(&self) -> SmolStr {
        self.value.name().into()
    }
}
impl StaticTag for Const {
    const TAG: OpTag = OpTag::Const;
}
impl OpTrait for Const {
    fn description(&self) -> &str {
        self.value.description()
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(self.typ.clone()))
    }
}

// [KnownTypeConst] is guaranteed to be the right type, so can be constructed
// without initial type check.
impl<T> From<T> for Const
where
    T: KnownTypeConst + CustomConst,
{
    fn from(value: T) -> Self {
        Const {
            value: Value::custom(value),
            typ: Type::new_extension(T::TYPE),
        }
    }
}

#[cfg(test)]
mod test {
    use super::Const;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            prelude::{ConstUsize, USIZE_T},
            EMPTY_REG,
        },
        type_row,
        types::test::COPYABLE_T,
        types::type_param::TypeArg,
        types::{CustomCheckFailure, CustomType, FunctionType, Type, TypeBound, TypeRow},
        values::{
            test::{serialized_float, CustomTestValue},
            CustomSerialized, Value,
        },
    };
    use cool_asserts::assert_matches;
    use serde_yaml::Value as YamlValue;

    use super::*;

    #[test]
    fn test_predicate() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![type_row![USIZE_T, COPYABLE_T], type_row![]];
        let pred_ty = Type::new_predicate(pred_rows.clone());

        let mut b = DFGBuilder::new(FunctionType::new(
            type_row![],
            TypeRow::from(vec![pred_ty.clone()]),
        ))?;
        let c = b.add_constant(Const::predicate(
            0,
            Value::tuple([CustomTestValue(TypeBound::Eq).into(), serialized_float(5.1)]),
            pred_rows.clone(),
        )?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w], &EMPTY_REG).unwrap();

        let mut b = DFGBuilder::new(FunctionType::new(type_row![], TypeRow::from(vec![pred_ty])))?;
        let c = b.add_constant(Const::predicate(1, Value::unit(), pred_rows)?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w], &EMPTY_REG).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_predicate() {
        let pred_rows = [type_row![USIZE_T, COPYABLE_T], type_row![]];

        let res = Const::predicate(0, Value::tuple([]), pred_rows);
        assert_matches!(res, Err(ConstTypeError::TupleWrongLength));
    }

    #[test]
    fn test_constant_values() {
        let int_value: Value = ConstUsize::new(257).into();
        USIZE_T.check_type(&int_value).unwrap();
        COPYABLE_T.check_type(&serialized_float(17.4)).unwrap();
        assert_matches!(
            COPYABLE_T.check_type(&int_value),
            Err(ConstTypeError::CustomCheckFail(
                CustomCheckFailure::TypeMismatch { .. }
            ))
        );
        let tuple_ty = Type::new_tuple(vec![USIZE_T, COPYABLE_T]);
        let tuple_val = Value::tuple([int_value.clone(), serialized_float(5.1)]);
        tuple_ty.check_type(&tuple_val).unwrap();
        let tuple_val2 = Value::tuple(vec![serialized_float(6.1), int_value.clone()]);
        assert_matches!(
            tuple_ty.check_type(&tuple_val2),
            Err(ConstTypeError::ValueCheckFail(ty, tv2)) => ty == tuple_ty && tv2 == tuple_val2
        );
        let tuple_val3 = Value::tuple([int_value, serialized_float(3.3), serialized_float(2.0)]);
        assert_eq!(
            tuple_ty.check_type(&tuple_val3),
            Err(ConstTypeError::TupleWrongLength)
        );
    }

    #[test]
    fn test_yaml_const() {
        let typ_int = CustomType::new(
            "mytype",
            vec![TypeArg::BoundedNat(8)],
            "myrsrc",
            TypeBound::Eq,
        );
        let val: Value = CustomSerialized::new(typ_int.clone(), YamlValue::Number(6.into())).into();
        let classic_t = Type::new_extension(typ_int.clone());
        assert_matches!(classic_t.least_upper_bound(), TypeBound::Eq);
        classic_t.check_type(&val).unwrap();

        let typ_qb = CustomType::new("mytype", vec![], "myrsrc", TypeBound::Eq);
        let t = Type::new_extension(typ_qb.clone());
        assert_matches!(t.check_type(&val),
            Err(ConstTypeError::CustomCheckFail(CustomCheckFailure::TypeMismatch{expected, found})) => expected == typ_int && found == typ_qb);

        assert_eq!(val, val);
    }
}
