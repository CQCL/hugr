//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{CustomType, EdgeKind, Type, TypeRow},
    values::{
        map_container_type, ConstTypeError, ContainerValue, CustomCheckFail, HashableValue,
        ValueOfType,
    },
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use super::OpTag;
use super::{OpName, OpTrait, StaticTag};

/// A constant value definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Const {
    value: ConstValue,
    typ: Type,
}

impl Const {
    /// Creates a new Const, type-checking the value.
    pub fn new(value: ConstValue, typ: Type) -> Result<Self, ConstTypeError> {
        value.check_type(&typ)?;
        Ok(Self { value, typ })
    }

    /// Returns a reference to the value of this [`Const`].
    pub fn value(&self) -> &ConstValue {
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
        value: ConstValue,
        variant_rows: impl IntoIterator<Item = TypeRow>,
    ) -> Result<Self, ConstTypeError> {
        let typ = Type::new_predicate(variant_rows);
        Self::new(ConstValue::sum(tag, value), typ)
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize, size: usize) -> Self {
        Self {
            value: ConstValue::simple_predicate(tag),
            typ: Type::new_simple_predicate(size),
        }
    }

    /// Constant Sum over units, with only one variant.
    pub fn simple_unary_predicate() -> Self {
        Self {
            value: ConstValue::simple_unary_predicate(),
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

    /// Size
    pub fn usize(value: u64) -> Result<Self, ConstTypeError> {
        Self::new(
            ConstValue::Hashable(HashableValue::Int(value)),
            Type::usize(),
        )
    }

    /// Tuple of values
    pub fn new_tuple(items: impl IntoIterator<Item = Const>) -> Self {
        let (values, types): (Vec<ConstValue>, Vec<Type>) = items
            .into_iter()
            .map(|Const { value, typ }| (value, typ))
            .unzip();
        Self::new(ConstValue::sequence(&values), Type::new_tuple(types)).unwrap()
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

/// Value constants. (This could be "ClassicValue" to parallel [HashableValue])
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ConstValue {
    Hashable(HashableValue),
    /// A collection of constant values (at least some of which are not [ConstValue::Hashable])
    Container(ContainerValue<ConstValue>),
    /// An opaque constant value, that can check it is of a given [CustomType].
    /// This may include values that are [hashable]
    ///
    /// [hashable]: crate::types::simple::TypeTag::Hashable
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Opaque((Box<dyn CustomConst>,)),
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

impl ValueOfType for ConstValue {
    type T = Type;

    fn name(&self) -> String {
        match self {
            ConstValue::Hashable(hv) => hv.name(),
            ConstValue::Container(ctr) => ctr.desc(),
            ConstValue::Opaque((v,)) => format!("const:custom:{}", v.name()),
        }
    }

    fn check_type(&self, ty: &Type) -> Result<(), ConstTypeError> {
        todo!();
        // match self {
        //     ConstValue::Hashable(hv) => {
        //         match ty {
        //             Type::Hashable(exp) => return hv.check_type(exp),
        //             Type::Container(cty) => {
        //                 // A "hashable" value might be an instance of a non-hashable type:
        //                 // e.g. an empty list is hashable, yet can be checked against a classic element type!
        //                 if let HashableValue::Container(ctr) = hv {
        //                     return ctr.map_vals(&ConstValue::Hashable).check_container(cty);
        //                 }
        //             }
        //             _ => (),
        //         }
        //     }
        //     ConstValue::Container(vals) => {
        //         match ty {
        //             Type::Container(cty) => return vals.check_container(cty),
        //             // We might also fail to deduce a container *value* was hashable,
        //             // because it contains opaque values whose tag is unknown.
        //             Type::Hashable(HashableType::Container(cty)) => {
        //                 return vals.check_container(&map_container_type(cty, &Type::Hashable))
        //             }
        //             _ => (),
        //         };
        //     }
        //     ConstValue::Opaque((val,)) => {
        //         let maybe_cty = match ty {
        //             Type::Container(Container::Opaque(t)) => Some(t),
        //             Type::Hashable(HashableType::Container(Container::Opaque(t))) => Some(t),
        //             _ => None,
        //         };
        //         if let Some(cu_ty) = maybe_cty {
        //             return val.check_custom_type(cu_ty).map_err(ConstTypeError::from);
        //         }
        //     }
        // };
        Err(ConstTypeError::ValueCheckFail(ty.clone(), self.clone()))
    }
}

impl ConstValue {
    /// Description of the constant.
    pub fn description(&self) -> &str {
        "Constant value"
    }

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> Self {
        Self::Hashable(HashableValue::Container(ContainerValue::Sequence(vec![])))
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize) -> Self {
        Self::sum(tag, Self::unit())
    }

    /// Constant Sum over Tuples with just one variant of unit type
    pub fn simple_unary_predicate() -> Self {
        Self::simple_predicate(0)
    }

    /// Sequence of values (could be a tuple, list or array)
    pub fn sequence(items: &[ConstValue]) -> Self {
        // Keep Hashable at the outside (if all values are)
        match items
            .iter()
            .map(|item| match item {
                ConstValue::Hashable(h) => Some(h),
                _ => None,
            })
            .collect::<Option<Vec<&HashableValue>>>()
        {
            Some(hashables) => ConstValue::Hashable(HashableValue::Container(
                ContainerValue::Sequence(hashables.into_iter().cloned().collect()),
            )),
            None => ConstValue::Container(ContainerValue::Sequence(items.to_vec())),
        }
    }

    /// Sum value (could be of any compatible type, e.g. a predicate)
    pub fn sum(tag: usize, value: ConstValue) -> Self {
        // Keep Hashable as outermost constructor
        match value {
            ConstValue::Hashable(hv) => {
                HashableValue::Container(ContainerValue::Sum(tag, Box::new(hv))).into()
            }
            _ => ConstValue::Container(ContainerValue::Sum(tag, Box::new(value))),
        }
    }
}

impl From<HashableValue> for ConstValue {
    fn from(hv: HashableValue) -> Self {
        Self::Hashable(hv)
    }
}

impl<T: CustomConst> From<T> for ConstValue {
    fn from(v: T) -> Self {
        Self::Opaque((Box::new(v),))
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
    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFail>;

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
struct CustomSerialized {
    typ: CustomType,
    value: serde_yaml::Value,
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> SmolStr {
        format!("yaml:{:?}", self.value).into()
    }

    fn check_custom_type(&self, typ: &CustomType) -> Result<(), CustomCheckFail> {
        if &self.typ == typ {
            Ok(())
        } else {
            Err(CustomCheckFail::TypeMismatch(typ.clone(), self.typ.clone()))
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;
    use serde_yaml::Value;

    use super::{Const, ConstValue, CustomSerialized};
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        classic_row, type_row,
        types::type_param::TypeArg,
        types::{custom::test::COPYABLE_CUST, test::CLASSIC_T, TypeRow},
        types::{AbstractSignature, CustomType, Type, TypeBound},
        values::{ConstTypeError, CustomCheckFail, HashableValue, ValueOfType},
    };

    fn custom_value(f: f64) -> ConstValue {
        ConstValue::Opaque((Box::new(CustomSerialized {
            typ: COPYABLE_CUST,
            value: serde_yaml::Value::Number(f.into()),
        }),))
    }

    #[test]
    fn test_predicate() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![type_row![Type::i64(), CLASSIC_T], type_row![]];
        let pred_ty = Type::new_predicate(pred_rows.clone());

        let mut b = DFGBuilder::new(AbstractSignature::new_df(
            type_row![],
            TypeRow::from(vec![pred_ty.clone()]),
        ))?;
        let c = b.add_constant(Const::predicate(
            0,
            ConstValue::sequence(&[
                ConstValue::Hashable(HashableValue::Int(3)),
                custom_value(5.1),
            ]),
            pred_rows.clone(),
        )?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        let mut b = DFGBuilder::new(AbstractSignature::new_df(
            type_row![],
            TypeRow::from(vec![pred_ty]),
        ))?;
        let c = b.add_constant(Const::predicate(1, ConstValue::unit(), pred_rows)?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_predicate() {
        let pred_rows = vec![type_row![Type::i64(), CLASSIC_T], type_row![]];

        let res = Const::predicate(0, ConstValue::sequence(&[]), pred_rows);
        assert_matches!(res, Err(ConstTypeError::TupleWrongLength));
    }

    #[test]
    fn test_constant_values() {
        const T_INT: Type = Type::usize();
        const V_INT: ConstValue = ConstValue::Hashable(HashableValue::Int(257));
        V_INT.check_type(&T_INT).unwrap();
        custom_value(17.4).check_type(&CLASSIC_T).unwrap();
        assert_matches!(
            V_INT.check_type(&CLASSIC_T),
            Err(ConstTypeError::ValueCheckFail(t, v)) => t == CLASSIC_T && v == V_INT
        );
        let tuple_ty = Type::new_tuple(type_row![T_INT, CLASSIC_T]);
        let tuple_val = ConstValue::sequence(&[V_INT, custom_value(5.1)]);
        tuple_val.check_type(&tuple_ty).unwrap();
        let tuple_val2 = ConstValue::sequence(&[custom_value(6.1), V_INT]);
        assert_matches!(
            tuple_val2.check_type(&tuple_ty),
            Err(ConstTypeError::ValueCheckFail(ty, tv2)) => ty == tuple_ty && tv2 == tuple_val2
        );
        let tuple_val3 = ConstValue::sequence(&[V_INT, custom_value(3.3), custom_value(2.0)]);
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
        let val = ConstValue::Opaque((Box::new(CustomSerialized {
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
