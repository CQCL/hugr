//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{simple::ClassicElem, ClassicRow, ClassicType, CustomType, EdgeKind},
    values::{
        ContainerValue, CustomCheckFail, HashableLeaf, OpaqueValueOfType, ValueError, ValueOfType,
    },
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use super::OpTag;
use super::{OpName, OpTrait, StaticTag};

pub mod typecheck;
/// A constant value definition.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Const {
    value: ConstValue,
    typ: ClassicType,
}

pub type ConstTypeError = ValueError<ConstValue>;

impl Const {
    /// Creates a new Const, type-checking the value.
    pub fn new(value: ConstValue, typ: ClassicType) -> Result<Self, ConstTypeError> {
        value.check_type(&typ)?;
        Ok(Self { value, typ })
    }

    /// Returns a reference to the value of this [`Const`].
    pub fn value(&self) -> &ConstValue {
        &self.value
    }

    /// Returns a reference to the type of this [`Const`].
    pub fn const_type(&self) -> &ClassicType {
        &self.typ
    }

    /// Sum of Tuples, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn predicate(
        tag: usize,
        value: ConstValue,
        variant_rows: impl IntoIterator<Item = ClassicRow>,
    ) -> Result<Self, ConstTypeError> {
        let typ = ClassicType::new_predicate(variant_rows);
        Self::new(ConstValue::sum(tag, value), typ)
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize, size: usize) -> Self {
        Self {
            value: ConstValue::simple_predicate(tag),
            typ: ClassicType::new_simple_predicate(size),
        }
    }

    /// Constant Sum over units, with only one variant.
    pub fn simple_unary_predicate() -> Self {
        Self {
            value: ConstValue::simple_unary_predicate(),
            typ: ClassicType::new_simple_predicate(1),
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

    /// Fixed width integer
    pub fn int<const N: u8>(value: HugrIntValueStore) -> Result<Self, ConstTypeError> {
        Self::new(
            ConstValue::Single(ConstLeaf::Hashable(HashableLeaf::Int(value))),
            ClassicType::int::<N>(),
        )
    }

    /// 64-bit integer
    pub fn i64(value: i64) -> Result<Self, ConstTypeError> {
        Self::int::<64>(value as HugrIntValueStore)
    }

    /// Tuple of values
    pub fn new_tuple(items: impl IntoIterator<Item = Const>) -> Self {
        let (values, types): (Vec<ConstValue>, Vec<ClassicType>) = items
            .into_iter()
            .map(|Const { value, typ }| (value, typ))
            .unzip();
        Self::new(ConstValue::sequence(&values), ClassicType::new_tuple(types)).unwrap()
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

pub(crate) type HugrIntValueStore = u128;
pub(crate) type HugrIntWidthStore = u8;
pub(crate) const HUGR_MAX_INT_WIDTH: HugrIntWidthStore =
    HugrIntValueStore::BITS as HugrIntWidthStore;

/// Value constants. (This could be "ClassicLeaf" to parallel [HashableLeaf])
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ConstLeaf {
    Hashable(HashableLeaf), // ALAN TODO need to standardize naming
    /// Double precision float
    F64(f64),
    /// An opaque constant value, that can check it is of a given [CustomType].
    /// This may include values that are [hashable]
    ///
    /// [hashable]: crate::types::simple::TypeTag::Hashable
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Opaque((Box<dyn CustomConst>,)),
}
impl From<HashableLeaf> for ConstLeaf {
    fn from(value: HashableLeaf) -> Self {
        Self::Hashable(value)
    }
}

pub type ConstValue = ContainerValue<ConstLeaf>;

impl ValueOfType for ConstLeaf {
    type T = ClassicElem;

    fn name(&self) -> String {
        match self {
            ConstLeaf::F64(f) => format!("const:float:{}", f),
            ConstLeaf::Hashable(hv) => hv.name(),
            ConstLeaf::Opaque((v,)) => format!("const:custom:{}", v.name()),
        }
    }

    fn check_type(&self, ty: &ClassicElem) -> Result<(), ValueError<ConstLeaf>> {
        match self {
            ConstLeaf::F64(_) => {
                if let ClassicElem::F64 = ty {
                    return Ok(());
                }
            }
            ConstLeaf::Hashable(hv) => {
                if let ClassicElem::Hashable(ht) = ty {
                    return hv.check_type(ht).map_err(|e| e.map_into());
                }
            }
            ConstLeaf::Opaque((val,)) => {}
        };
        Err(ValueError::ValueCheckFail(ty.clone(), self.clone()))
    }
}

impl OpaqueValueOfType for ConstLeaf {
    fn check_custom_type(&self, ty: &CustomType) -> Result<(), Option<CustomCheckFail>> {
        match self {
            ConstLeaf::Opaque((v,)) => v.check_custom_type(ty).map_err(Option::Some),
            _ => Err(None),
        }
    }
}

impl ConstValue {
    /// Description of the constant.
    pub fn description(&self) -> &str {
        "Constant value"
    }

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> Self {
        Self::sequence(&[])
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
        Self::Sequence(items.to_vec())
    }

    /// Sum value (could be of any compatible type, e.g. a predicate)
    pub fn sum(tag: usize, value: ConstValue) -> Self {
        Self::Sum(tag, Box::new(value))
    }
}

/*impl From<HashableValue> for ConstValue {
    fn from(hv: HashableValue) -> Self {
        Self::Hashable(hv)
    }
}*/

impl<T: CustomConst> From<T> for ConstValue {
    fn from(v: T) -> Self {
        Self::Single(ConstLeaf::Opaque((Box::new(v),)))
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

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

// Don't derive Eq here - the yaml could contain floats etc.
// (Perhaps we could derive Eq if-and-only-if "typ.tag() == TypeTag::Hashable"!)
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    use super::{typecheck::ConstIntError, Const, ConstTypeError, ConstValue, CustomSerialized};
    use crate::{
        builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
        classic_row, type_row,
        types::simple::Container,
        types::type_param::TypeArg,
        types::{ClassicType, CustomType, HashableType, SimpleRow, SimpleType, TypeTag},
        values::{CustomCheckFail, HashableValue, ValueOfType},
    };

    #[test]
    fn test_predicate() -> Result<(), BuildError> {
        use crate::builder::Container;
        let pred_rows = vec![
            classic_row![ClassicType::i64(), ClassicType::F64],
            type_row![],
        ];
        let pred_ty = SimpleType::new_predicate(pred_rows.clone());

        let mut b = DFGBuilder::new(type_row![], SimpleRow::from(vec![pred_ty.clone()]))?;
        let c = b.add_constant(Const::predicate(
            0,
            ConstValue::sequence(&[
                ConstValue::Hashable(HashableValue::Int(3)),
                ConstValue::F64(3.15),
            ]),
            pred_rows.clone(),
        )?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        let mut b = DFGBuilder::new(type_row![], SimpleRow::from(vec![pred_ty]))?;
        let c = b.add_constant(Const::predicate(1, ConstValue::unit(), pred_rows)?)?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        Ok(())
    }

    #[test]
    fn test_bad_predicate() {
        let pred_rows = vec![
            classic_row![ClassicType::i64(), ClassicType::F64],
            type_row![],
        ];

        let res = Const::predicate(0, ConstValue::sequence(&[]), pred_rows);
        assert_matches!(res, Err(ConstTypeError::WrongNumber(_, _, _)));
    }

    #[test]
    fn test_constant_values() {
        const T_INT: ClassicType = ClassicType::int::<64>();
        const V_INT: ConstValue = ConstValue::Hashable(HashableValue::Int(257));
        V_INT.check_type(&T_INT).unwrap();
        assert_eq!(
            V_INT.check_type(&ClassicType::int::<8>()),
            Err(ConstTypeError::Int(ConstIntError::IntTooLarge(8, 257)))
        );
        ConstValue::F64(17.4).check_type(&ClassicType::F64).unwrap();
        assert_matches!(
            V_INT.check_type(&ClassicType::F64),
            Err(ConstTypeError::ValueCheckFail(ClassicType::F64, v)) => v == V_INT
        );
        let tuple_ty = ClassicType::new_tuple(classic_row![T_INT, ClassicType::F64]);
        let tuple_val = ConstValue::sequence(&[V_INT, ConstValue::F64(5.1)]);
        tuple_val.check_type(&tuple_ty).unwrap();
        let tuple_val2 = ConstValue::sequence(&[ConstValue::F64(5.1), V_INT]);
        assert_matches!(
            tuple_val2.check_type(&tuple_ty),
            Err(ConstTypeError::ValueCheckFail(ty, tv2)) => ty == tuple_ty && tv2 == tuple_val2
        );
        let tuple_val3 = ConstValue::sequence(&[V_INT, ConstValue::F64(3.3), ConstValue::F64(2.0)]);
        assert_eq!(
            tuple_val3.check_type(&tuple_ty),
            Err(ConstTypeError::WrongNumber(_, _, _))
        );
    }

    #[test]
    fn test_yaml_const() {
        let typ_int = CustomType::new(
            "mytype",
            vec![TypeArg::ClassicType(ClassicType::Hashable(
                HashableType::Int(8),
            ))],
            "myrsrc",
            TypeTag::Hashable,
        );
        let val = ConstValue::Opaque((Box::new(CustomSerialized {
            typ: typ_int.clone(),
            value: Value::Number(6.into()),
        }),));
        let SimpleType::Classic(classic_t) = typ_int.clone().into()
            else {panic!("Hashable CustomType returned as non-Classic");};
        assert_matches!(classic_t, ClassicType::Hashable(_));
        val.check_type(&classic_t).unwrap();

        // This misrepresents the CustomType, so doesn't really "have to work".
        // But just as documentation of current behaviour:
        val.check_type(&ClassicType::Container(Container::Opaque(typ_int.clone())))
            .unwrap();

        let typ_float = CustomType::new(
            "mytype",
            vec![TypeArg::ClassicType(ClassicType::F64)],
            "myrsrc",
            TypeTag::Hashable,
        );
        let t: SimpleType = typ_float.clone().into();
        assert_matches!(val.check_type(&t.try_into().unwrap()),
            Err(ConstTypeError::CustomCheckFail(CustomCheckFail::TypeMismatch(a, b))) => a == typ_int && b == typ_float);

        assert_eq!(val, val);
    }
}
