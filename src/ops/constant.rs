//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{simple::Container, ClassicRow, ClassicType, CustomType, EdgeKind, HashableType},
    values::{
        map_container_type, ConstTypeError, ContainerValue, CustomCheckFail, HashableValue,
        ValueOfType,
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
            ConstValue::Hashable(HashableValue::Int(value)),
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

/// Value constants. (This could be "ClassicValue" to parallel [HashableValue])
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ConstValue {
    Hashable(HashableValue),
    /// A collection of constant values (at least some of which are not [ConstValue::Hashable])
    Container(ContainerValue<ConstValue>),
    /// Double precision float
    F64(f64),
    /// An opaque constant value, with cached type. TODO put this into ContainerValue.
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Opaque((Box<dyn CustomConst>,)),
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

impl ValueOfType for ConstValue {
    type T = ClassicType;

    fn name(&self) -> String {
        match self {
            ConstValue::F64(f) => format!("const:float:{}", f),
            ConstValue::Hashable(hv) => hv.name(),
            ConstValue::Container(ctr) => ctr.desc(),
            ConstValue::Opaque((v,)) => format!("const:custom:{}", v.name()),
        }
    }

    fn check_type(&self, ty: &ClassicType) -> Result<(), ConstTypeError> {
        match self {
            ConstValue::F64(_) => {
                if let ClassicType::F64 = ty {
                    return Ok(());
                }
            }
            ConstValue::Hashable(hv) => {
                match ty {
                    ClassicType::Hashable(exp) => return hv.check_type(exp),
                    ClassicType::Container(cty) => {
                        // A "hashable" value might be an instance of a non-hashable type:
                        // e.g. an empty list is hashable, yet can be checked against a classic element type!
                        if let HashableValue::Container(ctr) = hv {
                            return ctr.map_vals(&ConstValue::Hashable).check_container(cty);
                        }
                    }
                    _ => (),
                }
            }
            ConstValue::Container(vals) => {
                match ty {
                    ClassicType::Container(cty) => return vals.check_container(cty),
                    // We might also fail to deduce a container *value* was hashable,
                    // because it contains opaque values whose tag is unknown.
                    ClassicType::Hashable(HashableType::Container(cty)) => {
                        return vals
                            .check_container(&map_container_type(cty, &ClassicType::Hashable))
                    }
                    _ => (),
                };
            }
            ConstValue::Opaque((val,)) => {
                let maybe_cty = match ty {
                    ClassicType::Container(Container::Opaque(t)) => Some(t),
                    ClassicType::Hashable(HashableType::Container(Container::Opaque(t))) => Some(t),
                    _ => None,
                };
                if let Some(cu_ty) = maybe_cty {
                    return val.check_custom_type(cu_ty).map_err(ConstTypeError::from);
                }
            }
        };
        Err(ConstTypeError::ValueCheckFail(ty.clone(), self.clone()))
    }

    fn container_error(
        typ: Container<ClassicType>,
        vals: ContainerValue<ConstValue>,
    ) -> ConstTypeError {
        ConstTypeError::ValueCheckFail(ClassicType::Container(typ), ConstValue::Container(vals))
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use super::{typecheck::ConstIntError, Const, ConstValue};
    use crate::{
        builder::{BuildError, Container, DFGBuilder, Dataflow, DataflowHugr},
        classic_row, type_row,
        types::{ClassicType, SimpleRow, SimpleType},
        values::{ConstTypeError, HashableValue, ValueOfType},
    };

    #[test]
    fn test_predicate() -> Result<(), BuildError> {
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
        assert_matches!(res, Err(ConstTypeError::TupleWrongLength));
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
        let tuple_val3 =
            ConstValue::sequence(&vec![V_INT, ConstValue::F64(3.3), ConstValue::F64(2.0)]);
        assert_eq!(
            tuple_val3.check_type(&tuple_ty),
            Err(ConstTypeError::TupleWrongLength)
        );
    }
}
