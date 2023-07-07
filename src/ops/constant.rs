//! Constant value definitions.

use std::any::Any;

use crate::{
    macros::impl_box_clone,
    type_row,
    types::{ClassicType, Container, EdgeKind, SimpleType, TypeRow},
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

use super::tag::OpTag;
use super::{OpName, OpTrait};

/// A constant value definition.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Const(pub ConstValue);
impl OpName for Const {
    fn name(&self) -> SmolStr {
        self.0.name()
    }
}
impl OpTrait for Const {
    fn description(&self) -> &str {
        self.0.description()
    }

    fn tag(&self) -> OpTag {
        OpTag::Const
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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
        variants: TypeRow,
        val: Box<ConstValue>,
    },
    /// A tuple of constant values.
    Tuple(Vec<ConstValue>),
    /// An opaque constant value.
    Opaque(SimpleType, Box<dyn CustomConst>),
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Int {
                    value: l0,
                    width: l_width,
                },
                Self::Int {
                    value: r0,
                    width: r_width,
                },
            ) => l0 == r0 && l_width == r_width,
            (Self::Opaque(l0, l1), Self::Opaque(r0, r1)) => l0 == r0 && l1.eq(&**r1),
            (
                Self::Sum { tag, variants, val },
                Self::Sum {
                    tag: t1,
                    variants: type1,
                    val: v1,
                },
            ) => tag == t1 && variants == type1 && val == v1,

            (Self::Tuple(v1), Self::Tuple(v2)) => v1.eq(v2),
            (Self::F64(f1), Self::F64(f2)) => f1 == f2,

            _ => false,
        }
    }
}

impl Eq for ConstValue {}

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
            Self::Int { value: _, width } => ClassicType::Int(*width),
            Self::Opaque(_, b) => (*b).const_type(),
            Self::Sum { variants, .. } => {
                ClassicType::Container(Container::Sum(Box::new(variants.clone())))
            }
            Self::Tuple(vals) => {
                let row: Vec<_> = vals
                    .iter()
                    .map(|val| SimpleType::Classic(val.const_type()))
                    .collect();
                ClassicType::Container(Container::Tuple(Box::new(row.into())))
            }
            Self::F64(_) => ClassicType::F64,
        }
    }
    /// Unique name of the constant.
    pub fn name(&self) -> SmolStr {
        match self {
            Self::Int { value, width } => format!("const:int<{width}>:{value}"),
            Self::F64(f) => format!("const:float:{f}"),
            Self::Opaque(_, v) => format!("const:{}", v.name()),
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
        Self::predicate(tag, std::iter::repeat(type_row![]).take(size))
    }

    /// Constant Sum over Tuples, used as predicates.
    pub fn predicate(tag: usize, variant_rows: impl IntoIterator<Item = TypeRow>) -> Self {
        let variants = TypeRow::predicate_variants_row(variant_rows);
        assert!(variants.get(tag) == Some(&SimpleType::new_unit()));
        ConstValue::Sum {
            tag,
            variants,
            val: Box::new(Self::unit()),
        }
    }

    /// Constant Sum over Tuples with just one variant
    pub fn unary_predicate(row: impl Into<TypeRow>) -> Self {
        Self::predicate(0, [row.into()])
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
        Self::Opaque(SimpleType::Classic(v.const_type()), Box::new(v))
    }
}

/// Constant value for opaque [`SimpleType`]s.
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
    fn const_type(&self) -> ClassicType;

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    fn eq(&self, other: &dyn CustomConst) -> bool {
        let _ = other;
        false
    }
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);
