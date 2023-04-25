use std::any::Any;

use crate::{
    macros::impl_box_clone,
    types::{ClassicType, Container, EdgeKind, Signature, SimpleType, TypeRow},
};

use downcast_rs::{impl_downcast, Downcast};
use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ModuleOp {
    #[default]
    /// The root of a module, parent of all other `ModuleOp`s
    Root,
    /// A function definition.
    /// Children nodes are the body of the definition
    Def {
        signature: Signature,
    },
    /// External function declaration, linked at runtime
    Declare {
        signature: Signature,
    },
    /// Top level struct type definition
    #[non_exhaustive] // TODO
    Struct {},
    /// A type alias
    #[non_exhaustive] // TODO
    Alias {},
    // A constant value definition
    Const(ConstValue),
}

impl ModuleOp {
    pub fn name(&self) -> SmolStr {
        match self {
            ModuleOp::Root => "module",
            ModuleOp::Def { .. } => "def",
            ModuleOp::Declare { .. } => "declare",
            ModuleOp::Struct { .. } => "struct",
            ModuleOp::Alias { .. } => "alias",
            ModuleOp::Const(val) => return val.name(),
        }
        .into()
    }

    pub fn description(&self) -> &str {
        match self {
            ModuleOp::Root => "The root of a module, parent of all other `ModuleOp`s",
            ModuleOp::Def { .. } => "A function definition",
            ModuleOp::Declare { .. } => "External function declaration, linked at runtime",
            ModuleOp::Struct { .. } => "Top level struct type definition",
            ModuleOp::Alias { .. } => "A type alias",
            ModuleOp::Const(val) => val.description(),
        }
    }

    pub fn other_inputs(&self) -> Option<EdgeKind> {
        None
    }

    pub fn other_outputs(&self) -> Option<EdgeKind> {
        match self {
            ModuleOp::Root | ModuleOp::Struct { .. } | ModuleOp::Alias { .. } => None,
            ModuleOp::Def { signature } | ModuleOp::Declare { signature } => Some(EdgeKind::Const(
                ClassicType::graph_from_sig(signature.clone()),
            )),
            ModuleOp::Const(v) => Some(EdgeKind::Const(v.const_type())),
        }
    }
}

/// Value constants
///
/// TODO: Add more constants
/// TODO: bigger/smaller integers
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ConstValue {
    Int(i64),
    Sum {
        tag: usize,
        variants: TypeRow,
        val: Box<ConstValue>,
    },
    Tuple(Vec<ConstValue>),
    Opaque(SimpleType, Box<dyn CustomConst>),
}

impl PartialEq for ConstValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
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
            _ => false,
        }
    }
}

impl Eq for ConstValue {}

impl Default for ConstValue {
    fn default() -> Self {
        Self::Int(0)
    }
}

impl ConstValue {
    /// Returns the datatype of the constant
    pub fn const_type(&self) -> ClassicType {
        match self {
            Self::Int(_) => ClassicType::i64(),
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
        }
    }

    /// Unique name of the constant
    pub fn name(&self) -> SmolStr {
        match self {
            Self::Int(v) => format!("const:int:{v}"),
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

    /// Description of the constant
    pub fn description(&self) -> &str {
        "Constant value"
    }
}

impl<T: CustomConst> From<T> for ConstValue {
    fn from(v: T) -> Self {
        Self::Opaque(SimpleType::Classic(v.const_type()), Box::new(v))
    }
}

/// Constant value for opaque [`SimpleType`]s.
///
// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
#[typetag::serde]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
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
