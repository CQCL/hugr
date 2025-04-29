//! Handles to nodes in HUGR.
use crate::Node;
use crate::core::HugrNode;
use crate::types::{Type, TypeBound};

use derive_more::From as DerFrom;
use smol_str::SmolStr;

use super::{AliasDecl, OpTag};

/// Common trait for handles to a node.
/// Typically wrappers around [`Node`].
pub trait NodeHandle<N = Node>: Clone {
    /// The most specific operation tag associated with the handle.
    const TAG: OpTag;

    /// Index of underlying node.
    fn node(&self) -> N;

    /// Operation tag for the handle.
    #[inline]
    fn tag(&self) -> OpTag {
        Self::TAG
    }

    /// Cast the handle to a different more general tag.
    fn try_cast<T: NodeHandle<N> + From<N>>(&self) -> Option<T> {
        T::TAG.is_superset(Self::TAG).then(|| self.node().into())
    }

    /// Checks whether the handle can hold an operation with the given tag.
    #[must_use]
    fn can_hold(tag: OpTag) -> bool {
        Self::TAG.is_superset(tag)
    }
}

/// Trait for handles that contain children.
///
/// The allowed children handles are defined by the associated type.
pub trait ContainerHandle<N = Node>: NodeHandle<N> {
    /// Handle type for the children of this node.
    type ChildrenHandle: NodeHandle<N>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [`DataflowOp`](crate::ops::dataflow).
pub struct DataflowOpID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [DFG](crate::ops::DFG) node.
pub struct DfgID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [CFG](crate::ops::CFG) node.
pub struct CfgID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a module [Module](crate::ops::Module) node.
pub struct ModuleRootID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [module op](crate::ops::module) node.
pub struct ModuleID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [def](crate::ops::OpType::FuncDefn)
/// or [declare](crate::ops::OpType::FuncDecl) node.
///
/// The `DEF` const generic is used to indicate whether the function is
/// defined or just declared.
pub struct FuncID<const DEF: bool, N = Node>(N);

#[derive(Debug, Clone, PartialEq, Eq)]
/// Handle to an [`AliasDefn`](crate::ops::OpType::AliasDefn)
/// or [`AliasDecl`](crate::ops::OpType::AliasDecl) node.
///
/// The `DEF` const generic is used to indicate whether the function is
/// defined or just declared.
pub struct AliasID<const DEF: bool, N = Node> {
    node: N,
    name: SmolStr,
    bound: TypeBound,
}

impl<const DEF: bool, N> AliasID<DEF, N> {
    /// Construct new `AliasID`
    pub fn new(node: N, name: SmolStr, bound: TypeBound) -> Self {
        Self { node, name, bound }
    }

    /// Construct new `AliasID`
    pub fn get_alias_type(&self) -> Type {
        Type::new_alias(AliasDecl::new(self.name.clone(), self.bound))
    }
    /// Retrieve the underlying core type
    pub fn get_name(&self) -> &SmolStr {
        &self.name
    }
}

#[derive(DerFrom, Debug, Clone, PartialEq, Eq)]
/// Handle to a [Const](crate::ops::OpType::Const) node.
pub struct ConstID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [`DataflowBlock`](crate::ops::DataflowBlock) or [Exit](crate::ops::ExitBlock) node.
pub struct BasicBlockID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Case](crate::ops::Case) node.
pub struct CaseID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [`TailLoop`](crate::ops::TailLoop) node.
pub struct TailLoopID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a [Conditional](crate::ops::Conditional) node.
pub struct ConditionalID<N = Node>(N);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DerFrom, Debug)]
/// Handle to a dataflow container node.
pub struct DataflowParentID<N = Node>(N);

/// Implements the `NodeHandle` trait for a tuple struct that contains just a
/// `NodeIndex`. Takes the name of the struct, and the corresponding `OpTag`.
///
/// Optionally, the name of the field containing the `NodeIndex` can be specified
/// as a third argument. Otherwise, it is assumed to be a tuple struct 0th item.
macro_rules! impl_nodehandle {
    ($name:ident, $tag:expr) => {
        impl_nodehandle!($name, $tag, 0);
    };
    ($name:ident, $tag:expr, $node_attr:tt) => {
        impl<N: HugrNode> NodeHandle<N> for $name<N> {
            const TAG: OpTag = $tag;

            #[inline]
            fn node(&self) -> N {
                self.$node_attr
            }
        }
    };
}

impl_nodehandle!(DataflowParentID, OpTag::DataflowParent);
impl_nodehandle!(DataflowOpID, OpTag::DataflowChild);
impl_nodehandle!(ConditionalID, OpTag::Conditional);
impl_nodehandle!(CaseID, OpTag::Case);
impl_nodehandle!(DfgID, OpTag::Dfg);
impl_nodehandle!(TailLoopID, OpTag::TailLoop);
impl_nodehandle!(CfgID, OpTag::Cfg);

impl_nodehandle!(ModuleRootID, OpTag::ModuleRoot);
impl_nodehandle!(ModuleID, OpTag::ModuleOp);
impl_nodehandle!(ConstID, OpTag::Const);

impl_nodehandle!(BasicBlockID, OpTag::DataflowBlock);

impl<const DEF: bool, N: HugrNode> NodeHandle<N> for FuncID<DEF, N> {
    const TAG: OpTag = OpTag::Function;
    #[inline]
    fn node(&self) -> N {
        self.0
    }
}

impl<const DEF: bool, N: HugrNode> NodeHandle<N> for AliasID<DEF, N> {
    const TAG: OpTag = OpTag::Alias;
    #[inline]
    fn node(&self) -> N {
        self.node
    }
}

impl<N: HugrNode> NodeHandle<N> for N {
    const TAG: OpTag = OpTag::Any;
    #[inline]
    fn node(&self) -> N {
        *self
    }
}

/// Implements the `ContainerHandle` trait, with the given child handle type.
macro_rules! impl_containerHandle {
    ($name:ident, $children:ident) => {
        impl<N: HugrNode> ContainerHandle<N> for $name<N> {
            type ChildrenHandle = $children<N>;
        }
    };
}

impl_containerHandle!(DataflowParentID, DataflowOpID);
impl_containerHandle!(DfgID, DataflowOpID);
impl_containerHandle!(TailLoopID, DataflowOpID);
impl_containerHandle!(ConditionalID, CaseID);
impl_containerHandle!(CaseID, DataflowOpID);
impl_containerHandle!(ModuleRootID, ModuleID);
impl_containerHandle!(CfgID, BasicBlockID);
impl_containerHandle!(BasicBlockID, DataflowOpID);
impl<N: HugrNode> ContainerHandle<N> for FuncID<true, N> {
    type ChildrenHandle = DataflowOpID<N>;
}
impl<N: HugrNode> ContainerHandle<N> for AliasID<true, N> {
    type ChildrenHandle = DataflowOpID<N>;
}
