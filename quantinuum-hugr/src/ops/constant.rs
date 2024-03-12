//! Constant value definitions.

use crate::{
    extension::ExtensionSet,
    types::{ConstTypeError, EdgeKind, SumType, Type},
    values::{CustomConst, Value},
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

    /// Creates a new Const Sum.  The value determined by `items` and is
    /// type-checked `typ`
    pub fn new_sum(
        tag: usize,
        items: impl IntoIterator<Item = Const>,
        typ: SumType,
    ) -> Result<Self, ConstTypeError> {
        Self::new(
            Value::sum(tag, items.into_iter().map(|x| x.value().to_owned())),
            typ.into(),
        )
    }

    /// Constant Sum over units, used as branching values.
    pub fn unit_sum(tag: usize, size: u8) -> Self {
        Self {
            value: Value::unit_sum(tag),
            typ: Type::new_unit_sum(size),
        }
    }

    /// Constant Sum over units, with only one variant.
    pub fn unary_unit_sum() -> Self {
        Self::unit_sum(0, 1)
    }

    /// Constant "true" value, i.e. the second variant of Sum((), ()).
    pub fn true_val() -> Self {
        Self::unit_sum(1, 2)
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

    /// Constant "false" value, i.e. the first variant of Sum((), ()).
    pub fn false_val() -> Self {
        Self::unit_sum(0, 2)
    }

    /// Tuple of values
    pub fn new_tuple(items: impl IntoIterator<Item = Const>) -> Self {
        let (values, types): (Vec<Value>, Vec<Type>) = items
            .into_iter()
            .map(|Const { value, typ }| (value, typ))
            .unzip();
        Self::new(Value::tuple(values), Type::new_tuple(types)).unwrap()
    }

    /// For a Const holding a CustomConst, extract the CustomConst by downcasting.
    pub fn get_custom_value<T: CustomConst>(&self) -> Option<&T> {
        self.value().get_custom_value()
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

    fn extension_delta(&self) -> ExtensionSet {
        self.value.extension_reqs()
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn static_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(self.typ.clone()))
    }
}

// [KnownTypeConst] is guaranteed to be the right type, so can be constructed
// without initial type check.
impl<T> From<T> for Const
where
    T: CustomConst,
{
    fn from(value: T) -> Self {
        let typ = Type::new_extension(value.custom_type());
        Const {
            value: Value::custom(value),
            typ,
        }
    }
}

#[cfg(test)]
mod test {
    use super::Const;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        extension::{
            prelude::{ConstUsize, USIZE_CUSTOM_T, USIZE_T},
            ExtensionId, ExtensionRegistry, PRELUDE,
        },
        std_extensions::arithmetic::float_types::{self, ConstF64, FLOAT64_TYPE},
        type_row,
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

    fn test_registry() -> ExtensionRegistry {
        ExtensionRegistry::try_new([PRELUDE.to_owned(), float_types::EXTENSION.to_owned()]).unwrap()
    }

    #[test]
    fn test_sum() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![type_row![USIZE_T, FLOAT64_TYPE], Type::EMPTY_TYPEROW];
        let pred_ty = SumType::new(pred_rows.clone());

        let mut b = DFGBuilder::new(FunctionType::new(
            type_row![],
            TypeRow::from(vec![pred_ty.clone().into()]),
        ))?;
        let c = b.add_constant(Const::new_sum(
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
        let c = b.add_constant(Const::new_sum(1, [], pred_ty.clone())?);
        let w = b.load_const(&c);
        b.finish_hugr_with_outputs([w], &test_registry()).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_sum() {
        let pred_ty = SumType::new([type_row![USIZE_T, FLOAT64_TYPE], type_row![]]);

        let res = Const::new_sum(0, [Const::new_tuple(std::iter::empty())], pred_ty);
        assert_matches!(res, Err(ConstTypeError::SumWrongLength));
    }

    #[test]
    fn test_constant_values() {
        let int_value: Value = ConstUsize::new(257).into();
        USIZE_T.check_type(&int_value).unwrap();
        FLOAT64_TYPE
            .check_type(serialized_float(17.4).value())
            .unwrap();
        assert_matches!(
            FLOAT64_TYPE.check_type(&int_value),
            Err(ConstTypeError::CustomCheckFail(
                CustomCheckFailure::TypeMismatch { .. }
            ))
        );
        let tuple_ty = Type::new_tuple(vec![USIZE_T, FLOAT64_TYPE]);
        let tuple_val = Value::tuple([int_value.clone(), serialized_float(5.1).value().to_owned()]);
        tuple_ty.check_type(&tuple_val).unwrap();
        let tuple_val2 = Value::tuple(vec![
            serialized_float(6.1).value().to_owned(),
            int_value.clone(),
        ]);
        assert_matches!(
            tuple_ty.check_type(&tuple_val2),
            Err(ConstTypeError::ValueCheckFail(ty, tv2)) => ty == tuple_ty && tv2 == tuple_val2
        );
        let tuple_val3 = Value::tuple([
            int_value.clone(),
            serialized_float(3.3).value().clone(),
            serialized_float(2.0).value().clone(),
        ]);
        assert_eq!(
            tuple_ty.check_type(&tuple_val3),
            Err(ConstTypeError::TupleWrongLength)
        );

        let op = Const::new(int_value, USIZE_T).unwrap();

        assert_eq!(op.get_custom_value(), Some(&ConstUsize::new(257)));
        let try_float: Option<&ConstF64> = op.get_custom_value();
        assert!(try_float.is_none());
        let try_usize: Option<&ConstUsize> = tuple_val.get_custom_value();
        assert!(try_usize.is_none());
    }

    #[test]
    fn test_yaml_const() {
        let ex_id: ExtensionId = "myrsrc".try_into().unwrap();
        let typ_int = CustomType::new(
            "mytype",
            vec![TypeArg::BoundedNat { n: 8 }],
            ex_id.clone(),
            TypeBound::Eq,
        );
        let val: Value =
            CustomSerialized::new(typ_int.clone(), YamlValue::Number(6.into()), ex_id.clone())
                .into();
        let classic_t = Type::new_extension(typ_int.clone());
        assert_matches!(classic_t.least_upper_bound(), TypeBound::Eq);
        classic_t.check_type(&val).unwrap();

        let typ_qb = CustomType::new("mytype", vec![], ex_id, TypeBound::Eq);
        let t = Type::new_extension(typ_qb.clone());
        assert_matches!(t.check_type(&val),
            Err(ConstTypeError::CustomCheckFail(CustomCheckFailure::TypeMismatch{expected, found})) => expected == typ_int && found == typ_qb);

        assert_eq!(val, val);
    }
}
