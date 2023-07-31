//! Constant value definitions.

use std::any::Any;

use crate::{
    classic_row,
    macros::impl_box_clone,
    types::{ClassicRow, ClassicType, Container, CustomType, EdgeKind, HashableType},
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use super::OpTag;
use super::{OpName, OpTrait, StaticTag};

/// A constant value definition.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Const(pub ConstValue);
impl OpName for Const {
    fn name(&self) -> SmolStr {
        self.0.name()
    }
}
impl StaticTag for Const {
    const TAG: OpTag = OpTag::Const;
}
impl OpTrait for Const {
    fn description(&self) -> &str {
        self.0.description()
    }

    fn tag(&self) -> OpTag {
        <Self as StaticTag>::TAG
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::Static(self.0.const_type()))
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
    Int {
        value: HugrIntValueStore,
        width: HugrIntWidthStore,
    },
    /// Double precision float
    F64(f64),
    /// A constant specifying a variant of a Sum type.
    Sum {
        tag: usize,
        // We require the type to be entirely Classic (i.e. we don't allow
        // a classic variant of a Sum with other variants that are linear)
        variants: ClassicRow,
        val: Box<ConstValue>,
    },
    /// A tuple of constant values.
    Tuple(Vec<ConstValue>),
    /// An opaque constant value, with cached type
    // Note: the extra level of tupling is to avoid https://github.com/rust-lang/rust/issues/78808
    Opaque((CustomType, Box<dyn CustomConst>)),
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

impl Default for ConstValue {
    fn default() -> Self {
        Self::Int {
            value: 0,
            width: 64,
        }
    }
}

impl ConstValue {
    /// Returns the datatype of the constant.
    pub fn const_type(&self) -> ClassicType {
        match self {
            Self::Int { value: _, width } => HashableType::Int(*width).into(),
            Self::Opaque((_, b)) => Container::Opaque((*b).custom_type()).into(),
            Self::Sum { variants, .. } => ClassicType::new_sum(variants.clone()),
            Self::Tuple(vals) => {
                let row: Vec<_> = vals.iter().map(|val| val.const_type()).collect();
                ClassicType::new_tuple(row)
            }
            Self::F64(_) => ClassicType::F64,
        }
    }
    /// Unique name of the constant.
    pub fn name(&self) -> SmolStr {
        match self {
            Self::Int { value, width } => format!("const:int<{width}>:{value}"),
            Self::F64(f) => format!("const:float:{f}"),
            Self::Opaque((_, v)) => format!("const:{}", v.name()),
            Self::Sum { tag, val, .. } => {
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

    /// Constant "true" value, i.e. the second variant of Sum((), ()).
    pub fn true_val() -> Self {
        Self::simple_predicate(1, 2)
    }

    /// Constant "false" value, i.e. the first variant of Sum((), ()).
    pub fn false_val() -> Self {
        Self::simple_predicate(0, 2)
    }

    /// Constant Sum over units, used as predicates.
    pub fn simple_predicate(tag: usize, size: usize) -> Self {
        Self::predicate(
            tag,
            Self::unit(),
            std::iter::repeat(classic_row![]).take(size),
        )
    }

    /// Constant Sum over Tuples, used as predicates.
    pub fn predicate(
        tag: usize,
        val: ConstValue,
        variant_rows: impl IntoIterator<Item = ClassicRow>,
    ) -> Self {
        ConstValue::Sum {
            tag,
            variants: ClassicRow::predicate_variants_row(variant_rows),
            val: Box::new(val),
        }
    }

    /// Constant Sum over Tuples with just one variant of unit type
    pub fn simple_unary_predicate() -> Self {
        Self::simple_predicate(0, 1)
    }

    /// New 64 bit integer constant
    pub fn i64(value: i64) -> Self {
        Self::Int {
            value: value as HugrIntValueStore,
            width: 64,
        }
    }
}

impl<T: CustomConst> From<T> for ConstValue {
    fn from(v: T) -> Self {
        Self::Opaque((v.custom_type(), Box::new(v)))
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

    /// Returns the type of the constant.
    // TODO it would be good to ensure that this is a *classic* CustomType not a linear one!
    fn custom_type(&self) -> CustomType;

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
    use super::ConstValue;
    use crate::{
        builder::{BuildError, Container, DFGBuilder, Dataflow, DataflowHugr},
        classic_row,
        hugr::{typecheck::ConstTypeError, ValidationError},
        type_row,
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
        let c = b.add_constant(ConstValue::predicate(
            0,
            ConstValue::Tuple(vec![ConstValue::i64(3), ConstValue::F64(3.15)]),
            pred_rows.clone(),
        ))?;
        let w = b.load_const(&c)?;
        b.finish_hugr_with_outputs([w]).unwrap();

        let mut b = DFGBuilder::new(type_row![], SimpleRow::from(vec![pred_ty]))?;
        let c = b.add_constant(ConstValue::predicate(
            1,
            ConstValue::Tuple(vec![]),
            pred_rows,
        ))?;
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
        let pred_ty = SimpleType::new_predicate(pred_rows.clone());

        let mut b = DFGBuilder::new(type_row![], SimpleRow::from(vec![pred_ty])).unwrap();
        let c = b
            .add_constant(ConstValue::predicate(
                0,
                ConstValue::Tuple(vec![]),
                pred_rows,
            ))
            .unwrap();
        let w = b.load_const(&c).unwrap();
        assert_eq!(
            b.finish_hugr_with_outputs([w]),
            Err(BuildError::InvalidHUGR(ValidationError::ConstTypeError(
                ConstTypeError::TupleWrongLength
            )))
        );
    }
}
