//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{ClassicRow, ClassicType, CustomType, EdgeKind},
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use self::typecheck::{typecheck_const, ConstTypeError};

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

        Self::new(ConstValue::Sum(tag, Box::new(value)), typ)
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
        Self::new(ConstValue::Int(value), ClassicType::int::<N>())
    }

    /// 64-bit integer
    pub fn i64(value: i64) -> Result<Self, ConstTypeError> {
        Self::int::<64>(value as HugrIntValueStore)
    }
}

impl OpName for Const {
    fn name(&self) -> SmolStr {
        self.value.name()
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

/// Value constants
///
/// TODO: Add more constants
/// TODO: bigger/smaller integers.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ConstValue {
    /// An arbitrary length integer constant.
    Int(HugrIntValueStore),
    /// Double precision float
    F64(f64),
    /// A constant specifying a variant of a Sum type.
    Sum(usize, Box<ConstValue>),
    /// A tuple of constant values.
    Tuple(Vec<ConstValue>),
    /// An opaque constant value, with cached type
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Opaque((Box<dyn CustomConst>,)),
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

impl Default for ConstValue {
    fn default() -> Self {
        Self::Int(0)
    }
}

impl ConstValue {
    /// Returns the datatype of the constant.
    pub fn check_type(&self, typ: &ClassicType) -> Result<(), ConstTypeError> {
        typecheck_const(typ, self)
    }
    /// Unique name of the constant.
    pub fn name(&self) -> SmolStr {
        match self {
            Self::Int(value) => format!("const:int{value}"),
            Self::F64(f) => format!("const:float:{f}"),
            Self::Opaque((v,)) => format!("const:{}", v.name()),
            Self::Sum(tag, val) => {
                format!("const:sum:{{tag:{tag}, val:{}}}", val.name())
            }
            Self::Tuple(vals) => {
                let valstr: Vec<_> = vals.iter().map(|v| v.name()).collect();
                let valstr = valstr.join(", ");
                format!("const:tuple:{{{valstr}}}")
            }
        }
        .into()
    }

    /// Description of the constant.
    pub fn description(&self) -> &str {
        "Constant value"
    }

    /// Constant unit type (empty Tuple).
    pub const fn unit() -> ConstValue {
        ConstValue::Tuple(vec![])
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize) -> Self {
        Self::predicate(tag, Self::unit())
    }

    /// Constant Sum over Tuples, used as predicates.
    pub fn predicate(tag: usize, val: ConstValue) -> Self {
        ConstValue::Sum(tag, Box::new(val))
    }

    /// Constant Sum over Tuples with just one variant of unit type
    pub fn simple_unary_predicate() -> Self {
        Self::simple_predicate(0)
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
    fn check_type(&self, typ: &CustomType) -> Result<(), ConstTypeError>;

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;

    use super::ConstValue;
    use super::{typecheck::ConstTypeError, Const};
    use crate::{
        builder::{BuildError, Container, DFGBuilder, Dataflow, DataflowHugr},
        classic_row, type_row,
        types::{ClassicType, SimpleRow, SimpleType},
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
            ConstValue::Tuple(vec![ConstValue::Int(3), ConstValue::F64(3.15)]),
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

        let res = Const::predicate(0, ConstValue::Tuple(vec![]), pred_rows);
        assert_matches!(res, Err(ConstTypeError::TupleWrongLength));
    }
}
