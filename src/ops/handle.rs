//! Handles to nodes in HUGR.
//!
use crate::types::{ClassicType, Container, LinearType, SimpleType};
use crate::Node;

use derive_more::From as DerFrom;
use smol_str::SmolStr;

use super::tag::OpTag;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(Node, usize);

impl Wire {
    /// Create a new wire from a node and an offset.
    pub fn new(node: Node, offset: usize) -> Self {
        Self(node, offset)
    }

    /// The node that this wire is connected to.
    pub fn node(&self) -> Node {
        self.0
    }

    /// The offset of the output port that this wire is connected to.
    pub fn offset(&self) -> usize {
        self.1
    }
}

/// Common trait for handles to a node.
/// Typically wrappers around [`NodeIndex`].
pub trait NodeHandle: Clone {
    /// The most specific operation tag associated with the handle.
    const TAG: OpTag;

    /// Index of underlying node.
    fn node(&self) -> Node;

    /// Operation tag for the handle.
    #[inline]
    fn tag(&self) -> OpTag {
        Self::TAG
    }

    /// Cast the handle to a different more general tag.
    fn try_cast<T: NodeHandle + From<Node>>(&self) -> Option<T> {
        T::TAG.contains(Self::TAG).then(|| self.node().into())
    }
}

/// Trait for handles that contain children.
///
/// The allowed children handles are defined by the associated type.
pub trait ContainerHandle: NodeHandle {
    /// Handle type for the children of this node.
    type ChildrenHandle: NodeHandle;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [OpType](crate::ops::OpType).
pub struct OpID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [DataflowOp](crate::ops::dataflow::DataflowOp).
pub struct DataflowOpID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [DFG](crate::ops::dataflow::DataflowOp::DFG) node.
pub struct DfgID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [CFG](crate::ops::controlflow::ControlFlowOp::CFG) node.
pub struct CfgID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a module [Root](crate::ops::module::ModuleOp::Root) node.
pub struct ModuleRootID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [ModuleOp](crate::ops::module::ModuleOp) node.
pub struct ModuleID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [def](crate::ops::module::ModuleOp::Def)
/// or [declare](crate::ops::module::ModuleOp::Declare) node.
///
/// The `DEF` const generic is used to indicate whether the function is
/// defined or just declared.
pub struct FuncID<const DEF: bool>(Node);

#[derive(Debug, Clone, PartialEq, Eq)]
/// Handle to an [AliasDef](crate::ops::module::ModuleOp::AliasDef)
/// or [AliasDeclare](crate::ops::module::ModuleOp::AliasDeclare) node.
///
/// The `DEF` const generic is used to indicate whether the function is
/// defined or just declared.
pub struct AliasID<const DEF: bool> {
    node: Node,
    name: SmolStr,
    linear: bool,
}

impl<const DEF: bool> AliasID<DEF> {
    /// Construct new AliasID
    pub fn new(node: Node, name: SmolStr, linear: bool) -> Self {
        Self { node, name, linear }
    }

    /// Construct new AliasID
    pub fn get_alias_type(&self) -> SimpleType {
        if self.linear {
            Container::<LinearType>::Alias(self.name.clone()).into()
        } else {
            Container::<ClassicType>::Alias(self.name.clone()).into()
        }
    }
    /// Retrieve the underlying core type
    pub fn get_name(&self) -> &SmolStr {
        &self.name
    }
}

#[derive(DerFrom, Debug, Clone, PartialEq, Eq)]
/// Handle to a [Const](crate::ops::module::ModuleOp::Const) node.
pub struct ConstID(Node, ClassicType);

impl ConstID {
    /// Return the type of the constant.
    pub fn const_type(&self) -> ClassicType {
        self.1.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [BasicBlock](crate::ops::controlflow::BasicBlockOp) node.
pub struct BasicBlockID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Case](crate::ops::controlflow::CaseOp) node.
pub struct CaseID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [TailLoop](crate::ops::controlflow::ControlFlowOp::TailLoop) node.
pub struct TailLoopID(Node);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Conditional](crate::ops::controlflow::ControlFlowOp::Conditional) node.
pub struct ConditionalID(Node);

/// Implements the `NodeHandle` trait for a tuple struct that contains just a
/// NodeIndex. Takes the name of the struct, and the corresponding OpTag.
///
/// Optionally, the name of the field containing the NodeIndex can be specified
/// as a third argument. Otherwise, it is assumed to be a tuple struct 0th item.
macro_rules! impl_nodehandle {
    ($name:ident, $tag:expr) => {
        impl_nodehandle!($name, $tag, 0);
    };
    ($name:ident, $tag:expr, $node_attr:tt) => {
        impl NodeHandle for $name {
            const TAG: OpTag = $tag;

            #[inline]
            fn node(&self) -> Node {
                self.$node_attr
            }
        }
    };
}

impl_nodehandle!(OpID, OpTag::Any);

impl_nodehandle!(DataflowOpID, OpTag::DataflowOp);
impl_nodehandle!(ConditionalID, OpTag::Conditional);
impl_nodehandle!(CaseID, OpTag::Case);
impl_nodehandle!(DfgID, OpTag::Dfg);
impl_nodehandle!(TailLoopID, OpTag::TailLoop);
impl_nodehandle!(CfgID, OpTag::Cfg);

impl_nodehandle!(ModuleRootID, OpTag::ModuleRoot);
impl_nodehandle!(ModuleID, OpTag::ModuleOp);
impl_nodehandle!(ConstID, OpTag::Const);

impl_nodehandle!(BasicBlockID, OpTag::BasicBlock);

impl<const DEF: bool> NodeHandle for FuncID<DEF> {
    const TAG: OpTag = OpTag::Function;
    #[inline]
    fn node(&self) -> Node {
        self.0
    }
}

impl<const DEF: bool> NodeHandle for AliasID<DEF> {
    const TAG: OpTag = OpTag::Alias;
    #[inline]
    fn node(&self) -> Node {
        self.node
    }
}

/// Implements the `ContainerHandle` trait, with the given child handle type.
macro_rules! impl_containerHandle {
    ($name:path, $children:ident) => {
        impl ContainerHandle for $name {
            type ChildrenHandle = $children;
        }
    };
}

impl_containerHandle!(DfgID, DataflowOpID);
impl_containerHandle!(TailLoopID, DataflowOpID);
impl_containerHandle!(ConditionalID, CaseID);
impl_containerHandle!(CaseID, DataflowOpID);
impl_containerHandle!(ModuleRootID, ModuleID);
impl_containerHandle!(CfgID, BasicBlockID);
impl_containerHandle!(BasicBlockID, DataflowOpID);
impl_containerHandle!(FuncID<true>, DataflowOpID);
impl_containerHandle!(AliasID<true>, DataflowOpID);
