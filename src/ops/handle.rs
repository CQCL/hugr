//! Handles to nodes in HUGR.
//!
use crate::types::{ClassicType, SimpleType};

use derive_more::From as DerFrom;
use portgraph::NodeIndex;
use smol_str::SmolStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(NodeIndex, usize);

impl Wire {
    /// Create a new wire from a node and an offset.
    pub fn new(node: NodeIndex, offset: usize) -> Self {
        Self(node, offset)
    }

    /// The node that this wire is connected to.
    pub fn node(&self) -> NodeIndex {
        self.0
    }

    /// The offset of the output port that this wire is connected to.
    pub fn offset(&self) -> usize {
        self.1
    }
}

/// Common trait for handles to a node.
/// Typically wrappers around [`NodeIndex`].
pub trait NodeHandle {
    /// Index of underlying node.
    fn node(&self) -> NodeIndex;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [DataflowOp](crate::ops::dataflow::DataflowOp).
pub struct DataflowOpID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [DFG](crate::ops::dataflow::DataflowOp::DFG) node.
pub struct DfgID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [CFG](crate::ops::controlflow::ControlFlowOp::CFG) node.
pub struct CfgID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a module [Root](crate::ops::module::ModuleOp::Root) node.
pub struct ModuleRootID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [ModuleOp](crate::ops::module::ModuleOp) node.
pub struct ModuleID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [def](crate::ops::module::ModuleOp::Def)
/// or [declare](crate::ops::module::ModuleOp::Declare) node.
pub struct FuncID(NodeIndex);

#[derive(DerFrom, Debug, Clone, PartialEq, Eq)]
/// Handle to a [NewType](crate::ops::module::ModuleOp::NewType) node.
pub struct NewTypeID {
    node: NodeIndex,
    name: SmolStr,
    core_type: SimpleType,
}

impl NewTypeID {
    /// Retrieve the NewType
    pub fn get_new_type(&self) -> SimpleType {
        self.core_type.clone().into_new_type(self.name.clone())
    }

    /// Retrieve the underlying core type
    pub fn get_core_type(&self) -> &SimpleType {
        &self.core_type
    }

    /// Retrieve the underlying core type
    pub fn get_name(&self) -> &SmolStr {
        &self.name
    }
}

#[derive(DerFrom, Debug, Clone, PartialEq, Eq)]
/// Handle to a [Const](crate::ops::module::ModuleOp::Const) node.
pub struct ConstID(NodeIndex, ClassicType);

impl ConstID {
    /// Return the type of the constant.
    pub fn const_type(&self) -> ClassicType {
        self.1.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [BasicBlock](crate::ops::controlflow::BasicBlockOp) node.
pub struct BasicBlockID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Case](crate::ops::controlflow::CaseOp) node.
pub struct CaseID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [TailLoop](crate::ops::controlflow::ControlFlowOp::TailLoop) node.
pub struct TailLoopID(NodeIndex);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Conditional](crate::ops::controlflow::ControlFlowOp::Conditional) node.
pub struct ConditionalID(NodeIndex);

/// Implements the `NodeHandle` trait for a tuple struct that contains just a
/// NodeIndex.
macro_rules! impl_transparent_nodehandle {
    ($name:ident) => {
        impl NodeHandle for $name {
            #[inline]
            fn node(&self) -> NodeIndex {
                self.0
            }
        }
    };
}
impl_transparent_nodehandle!(DataflowOpID);
impl_transparent_nodehandle!(ConditionalID);
impl_transparent_nodehandle!(DfgID);
impl_transparent_nodehandle!(TailLoopID);
impl_transparent_nodehandle!(CfgID);
impl_transparent_nodehandle!(ModuleRootID);
impl_transparent_nodehandle!(ModuleID);
impl_transparent_nodehandle!(FuncID);
impl_transparent_nodehandle!(BasicBlockID);
impl_transparent_nodehandle!(ConstID);

impl NodeHandle for NewTypeID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.node
    }
}
