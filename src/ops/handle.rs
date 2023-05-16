//! Handles to nodes in HUGR.
//!
use crate::types::{ClassicType, SimpleType};

use derive_more::From as DerFrom;
use portgraph::NodeIndex;
use smol_str::SmolStr;

use super::tag::OpTag;

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
    /// The most specific operation tag associated with the handle.
    const TAG: OpTag;

    /// Index of underlying node.
    fn node(&self) -> NodeIndex;

    /// Operation tag for the handle.
    #[inline]
    fn tag(&self) -> OpTag {
        Self::TAG
    }

    /// Cast the handle to a different more general tag.
    fn try_cast<T: NodeHandle + From<NodeIndex>>(&self) -> Option<T> {
        T::TAG.contains(Self::TAG).then(|| self.node().into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [OpType](crate::ops::OpType).
pub struct OpID(NodeIndex);

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
    /// Create a new NewTypeID
    pub fn new(node: NodeIndex, name: SmolStr, core_type: SimpleType) -> Self {
        Self {
            node,
            name,
            core_type,
        }
    }

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
/// NodeIndex. Takes the name of the struct, and the corresponding OpTag.
///
/// Optionally, the name of the field containing the NodeIndex can be specified
/// as a third argument. Otherwise, it is assumed to be a tuple struct 0th item.
macro_rules! impl_transparent_nodehandle {
    ($name:ident, $tag:expr) => {
        impl_transparent_nodehandle!($name, $tag, 0);
    };
    ($name:ident, $tag:expr, $node_attr:tt) => {
        impl NodeHandle for $name {
            const TAG: OpTag = $tag;

            #[inline]
            fn node(&self) -> NodeIndex {
                self.$node_attr
            }
        }
    };
}

impl_transparent_nodehandle!(OpID, OpTag::Any);

impl_transparent_nodehandle!(DataflowOpID, OpTag::DataflowOp);
impl_transparent_nodehandle!(ConditionalID, OpTag::Conditional);
impl_transparent_nodehandle!(DfgID, OpTag::Dfg);
impl_transparent_nodehandle!(TailLoopID, OpTag::TailLoop);
impl_transparent_nodehandle!(CfgID, OpTag::Cfg);

impl_transparent_nodehandle!(ModuleRootID, OpTag::ModuleRoot);
impl_transparent_nodehandle!(ModuleID, OpTag::ModuleOp);
impl_transparent_nodehandle!(FuncID, OpTag::Function);
impl_transparent_nodehandle!(ConstID, OpTag::Const);

impl_transparent_nodehandle!(BasicBlockID, OpTag::BasicBlock);
impl_transparent_nodehandle!(NewTypeID, OpTag::NewType, node);
