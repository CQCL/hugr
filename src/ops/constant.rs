//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{ConstTypeError, CustomCheckFail, CustomType, EdgeKind, Type, TypeRow},
    values::{PrimValue, Value},
};

use downcast_rs::{impl_downcast, Downcast};
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
    pub fn simple_predicate(tag: usize, size: usize) -> Self {
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

// impl ValueOfType for Value {
//     type T = Type;

//     fn name(&self) -> String {
//         match self {
//             Value::Hashable(hv) => hv.name(),
//             Value::Container(ctr) => ctr.desc(),
//             Value::Opaque((v,)) => format!("const:custom:{}", v.name()),
//         }
//     }

//     fn check_type(&self, ty: &Type) -> Result<(), ConstTypeError> {
//         todo!();
//         match self {
//             Value::Hashable(hv) => {
//                 match ty {
//                     Type::Hashable(exp) => return hv.check_type(exp),
//                     Type::Container(cty) => {
//                         // A "hashable" value might be an instance of a non-hashable type:
//                         // e.g. an empty list is hashable, yet can be checked against a classic element type!
//                         if let HashableValue::Container(ctr) = hv {
//                             return ctr.map_vals(&Value::Hashable).check_container(cty);
//                         }
//                     }
//                     _ => (),
//                 }
//             }
//             Value::Container(vals) => {
//                 match ty {
//                     Type::Container(cty) => return vals.check_container(cty),
//                     // We might also fail to deduce a container *value* was hashable,
//                     // because it contains opaque values whose tag is unknown.
//                     Type::Hashable(HashableType::Container(cty)) => {
//                         return vals.check_container(&map_container_type(cty, &Type::Hashable))
//                     }
//                     _ => (),
//                 };
//             }
//             Value::Opaque((val,)) => {
//                 let maybe_cty = match ty {
//                     Type::Container(Container::Opaque(t)) => Some(t),
//                     Type::Hashable(HashableType::Container(Container::Opaque(t))) => Some(t),
//                     _ => None,
//                 };
//                 if let Some(cu_ty) = maybe_cty {
//                     return val.check_custom_type(cu_ty).map_err(ConstTypeError::from);
//                 }
//             }
//         };
//         Err(ConstTypeError::ValueCheckFail(ty.clone(), self.clone()))
//     }
// }

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;
    use serde_yaml::Value as YamlValue;

    use super::Const;
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        type_row,
        types::{custom::test::COPYABLE_CUST, test::CLASSIC_T, TypeRow},
        types::{test::EQ_T, type_param::TypeArg},
        types::{AbstractSignature, CustomType, Type, TypeBound},
        values::{PrimValue, Value},
    };

    fn custom_value(f: f64) -> Value {
        Value::Prim(PrimValue::Extension((Box::new(CustomSerialized {
            typ: COPYABLE_CUST,
            value: serde_yaml::Value::Number(f.into()),
        }),)))
    }

    #[test]
    fn test_predicate() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![type_row![EQ_T, CLASSIC_T], type_row![]];
        let pred_ty = Type::new_predicate(pred_rows.clone());

        let mut b = DFGBuilder::new(AbstractSignature::new_df(
            type_row![],
            TypeRow::from(vec![pred_ty.clone()]),
        ))?;
        let c = b.add_constant(Const::predicate(
            0,
            Value::tuple(&[Value::Hashable(HashableValue::Int(3)), custom_value(5.1)]),
            pred_rows.clone(),
        )?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        let mut b = DFGBuilder::new(AbstractSignature::new_df(
            type_row![],
            TypeRow::from(vec![pred_ty]),
        ))?;
        let c = b.add_constant(Const::predicate(1, Value::unit(), pred_rows)?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_predicate() {
        let pred_rows = vec![type_row![Type::i64(), CLASSIC_T], type_row![]];

        let res = Const::predicate(0, Value::sequence(&[]), pred_rows);
        assert_matches!(res, Err(ConstTypeError::TupleWrongLength));
    }

    #[test]
    fn test_constant_values() {
        const T_INT: Type = Type::usize();
        const V_INT: Value = Value::Hashable(HashableValue::Int(257));
        V_INT.check_type(&T_INT).unwrap();
        custom_value(17.4).check_type(&CLASSIC_T).unwrap();
        assert_matches!(
            V_INT.check_type(&CLASSIC_T),
            Err(ConstTypeError::ValueCheckFail(t, v)) => t == CLASSIC_T && v == V_INT
        );
        let tuple_ty = Type::new_tuple(type_row![T_INT, CLASSIC_T]);
        let tuple_val = Value::sequence(&[V_INT, custom_value(5.1)]);
        tuple_val.check_type(&tuple_ty).unwrap();
        let tuple_val2 = Value::sequence(&[custom_value(6.1), V_INT]);
        assert_matches!(
            tuple_val2.check_type(&tuple_ty),
            Err(ConstTypeError::ValueCheckFail(ty, tv2)) => ty == tuple_ty && tv2 == tuple_val2
        );
        let tuple_val3 = Value::sequence(&[V_INT, custom_value(3.3), custom_value(2.0)]);
        assert_eq!(
            tuple_val3.check_type(&tuple_ty),
            Err(ConstTypeError::TupleWrongLength)
        );
    }

    #[test]
    fn test_yaml_const() {
        let typ_int = CustomType::new(
            "mytype",
            vec![TypeArg::USize(8)],
            "myrsrc",
            Some(TypeBound::Eq),
        );
        let val = Value::Opaque((Box::new(CustomSerialized {
            typ: typ_int.clone(),
            value: Value::Number(6.into()),
        }),));
        let classic_t = Type::new_extension(typ_int);
        assert_matches!(classic_t.least_upper_bound(), Some(TypeBound::Eq));
        val.check_type(&classic_t).unwrap();

        let typ_qb = CustomType::new("mytype", vec![], "myrsrc", Some(TypeBound::Eq));
        let t = Type::new_extension(typ_qb);
        assert_matches!(val.check_type(&t.try_into().unwrap()),
            Err(ConstTypeError::CustomCheckFail(CustomCheckFail::TypeMismatch(a, b))) => a == typ_int && b == typ_qb);

        assert_eq!(val, val);
    }
}
