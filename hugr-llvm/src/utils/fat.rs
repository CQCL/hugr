//! We define a type [`FatNode`], named for analogy with a "fat pointer".
//!
//! We define a trait [`FatExt`], an extension trait for [`HugrView`]. It provides
//! methods that return [`FatNode`]s rather than [Node]s.
use std::{cmp::Ordering, fmt, hash::Hash, marker::PhantomData, ops::Deref};

use hugr_core::hugr::views::Rerooted;
use hugr_core::{
    Hugr, HugrView, IncomingPort, Node, NodeIndex, OutgoingPort,
    core::HugrNode,
    ops::{CFG, DataflowBlock, ExitBlock, Input, Module, OpType, Output},
    types::Type,
};
use itertools::Itertools as _;

/// A Fat Node is a [Node] along with a reference to the [`HugrView`] whence it
/// originates.
///
/// It carries a type `OT`, the [`OpType`] of that node. `OT` may be
/// general, i.e. exactly [`OpType`], or specifec, e.g. [`FuncDefn`].
///
/// We provide a [Deref<Target=OT>] impl, so it can be used in place of `OT`.
///
/// We provide [`PartialEq`], [Eq], [`PartialOrd`], [Ord], [Hash], so that this type
/// can be used in containers. Note that these implementations use only the
/// stored node, so will silently malfunction if used with [`FatNode`]s from
/// different base [Hugr]s. Note that [Node] has this same behaviour.
///
/// [`FuncDefn`]: [hugr_core::ops::FuncDefn]
#[derive(Debug)]
pub struct FatNode<'hugr, OT = OpType, H = Hugr, N = Node>
where
    H: ?Sized,
{
    hugr: &'hugr H,
    node: N,
    marker: PhantomData<OT>,
}

impl<'hugr, OT, H: HugrView + ?Sized> FatNode<'hugr, OT, H, H::Node>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    /// Create a `FatNode` from a [`HugrView`] and a [Node].
    ///
    /// Panics if the node is not valid in the `Hugr` or if it's `get_optype` is
    /// not an `OT`.
    ///
    /// Note that while we do check the type of the node's `get_optype`, we
    /// do not verify that it is actually equal to `ot`.
    pub fn new(hugr: &'hugr H, node: H::Node, #[allow(unused)] ot: &OT) -> Self {
        assert!(hugr.contains_node(node));
        assert!(TryInto::<&OT>::try_into(hugr.get_optype(node)).is_ok());
        // We don't actually check `ot == hugr.get_optype(node)` so as to not require OT: PartialEq`
        Self {
            hugr,
            node,
            marker: PhantomData,
        }
    }

    /// Tries to create a `FatNode` from a [`HugrView`] and a node (typically a
    /// [Node]).
    ///
    /// If the node is invalid, or if its `get_optype` is not `OT`, returns
    /// `None`.
    pub fn try_new(hugr: &'hugr H, node: H::Node) -> Option<Self> {
        (hugr.contains_node(node)).then_some(())?;
        Some(Self::new(
            hugr,
            node,
            hugr.get_optype(node).try_into().ok()?,
        ))
    }

    /// Create a general `FatNode` from a specific one.
    pub fn generalise(self) -> FatNode<'hugr, OpType, H, H::Node> {
        // guaranteed to be valid because self is valid
        FatNode {
            hugr: self.hugr,
            node: self.node,
            marker: PhantomData,
        }
    }
}

impl<'hugr, OT, H, N: HugrNode> FatNode<'hugr, OT, H, N> {
    /// Gets the [Node] of the `FatNode`.
    pub fn node(&self) -> N {
        self.node
    }

    /// Gets the [`HugrView`] of the `FatNode`.
    pub fn hugr(&self) -> &'hugr H {
        self.hugr
    }
}

impl<'hugr, H: HugrView + ?Sized> FatNode<'hugr, OpType, H, H::Node> {
    /// Creates a new general `FatNode` from a [`HugrView`] and a [Node].
    ///
    /// Panics if the node is not valid in the [Hugr].
    pub fn new_optype(hugr: &'hugr H, node: H::Node) -> Self {
        assert!(hugr.contains_node(node));
        FatNode::new(hugr, node, hugr.get_optype(node))
    }

    /// Tries to downcast a general `FatNode` into a specific `OT`.
    pub fn try_into_ot<OT>(&self) -> Option<FatNode<'hugr, OT, H, H::Node>>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        FatNode::try_new(self.hugr, self.node)
    }

    /// Creates a specific `FatNode` from a general `FatNode`.
    ///
    /// Panics if the node is not valid in the `Hugr` or if its `get_optype` is
    /// not an `OT`.
    ///
    /// Note that while we do check the type of the node's `get_optype`, we
    /// do not verify that it is actually equal to `ot`.
    pub fn into_ot<OT>(self, ot: &OT) -> FatNode<'hugr, OT, H, H::Node>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        FatNode::new(self.hugr, self.node, ot)
    }
}

impl<'hugr, OT, H: HugrView + ?Sized> FatNode<'hugr, OT, H, H::Node> {
    /// If there is exactly one `OutgoingPort` connected to this `IncomingPort`,
    /// return it and its node.
    #[allow(clippy::type_complexity)]
    pub fn single_linked_output(
        &self,
        port: IncomingPort,
    ) -> Option<(FatNode<'hugr, OpType, H, H::Node>, OutgoingPort)> {
        self.hugr
            .single_linked_output(self.node, port)
            .map(|(n, p)| (FatNode::new_optype(self.hugr, n), p))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    pub fn out_value_types(
        &self,
    ) -> impl Iterator<Item = (OutgoingPort, Type)> + 'hugr + use<'hugr, OT, H> {
        self.hugr.out_value_types(self.node)
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    pub fn in_value_types(
        &self,
    ) -> impl Iterator<Item = (IncomingPort, Type)> + 'hugr + use<'hugr, OT, H> {
        self.hugr.in_value_types(self.node)
    }

    /// Return iterator over the direct children of node.
    pub fn children(
        &self,
    ) -> impl Iterator<Item = FatNode<'hugr, OpType, H, H::Node>> + 'hugr + use<'hugr, OT, H> {
        self.hugr
            .children(self.node)
            .map(|n| FatNode::new_optype(self.hugr, n))
    }

    /// Get the input and output child nodes of a dataflow parent.
    /// If the node isn't a dataflow parent, then return None
    #[allow(clippy::type_complexity)]
    pub fn get_io(
        &self,
    ) -> Option<(
        FatNode<'hugr, Input, H, H::Node>,
        FatNode<'hugr, Output, H, H::Node>,
    )> {
        let [i, o] = self.hugr.get_io(self.node)?;
        Some((
            FatNode::try_new(self.hugr, i)?,
            FatNode::try_new(self.hugr, o)?,
        ))
    }

    /// Iterator over output ports of node.
    pub fn node_outputs(&self) -> impl Iterator<Item = OutgoingPort> + 'hugr + use<'hugr, OT, H> {
        self.hugr.node_outputs(self.node)
    }

    /// Iterates over the output neighbours of the `node`.
    pub fn output_neighbours(
        &self,
    ) -> impl Iterator<Item = FatNode<'hugr, OpType, H, H::Node>> + 'hugr + use<'hugr, OT, H> {
        self.hugr
            .output_neighbours(self.node)
            .map(|n| FatNode::new_optype(self.hugr, n))
    }

    /// Returns a view of the internal [`HugrView`] with this [Node] as entrypoint.
    pub fn as_entrypoint(&self) -> Rerooted<&H>
    where
        H: Sized,
    {
        self.hugr.with_entrypoint(self.node)
    }
}

impl<'hugr, H: HugrView> FatNode<'hugr, CFG, H, H::Node> {
    /// Returns the entry and exit nodes of a CFG.
    ///
    /// These are guaranteed to exist the `Hugr` is valid. Panics if they do not
    /// exist.
    #[allow(clippy::type_complexity)]
    pub fn get_entry_exit(
        &self,
    ) -> (
        FatNode<'hugr, DataflowBlock, H, H::Node>,
        FatNode<'hugr, ExitBlock, H, H::Node>,
    ) {
        let [i, o] = self
            .hugr
            .children(self.node)
            .take(2)
            .collect_vec()
            .try_into()
            .unwrap();
        (
            FatNode::try_new(self.hugr, i).unwrap(),
            FatNode::try_new(self.hugr, o).unwrap(),
        )
    }
}

impl<OT, H> PartialEq<Node> for FatNode<'_, OT, H, Node> {
    fn eq(&self, other: &Node) -> bool {
        &self.node == other
    }
}

impl<OT, H> PartialEq<FatNode<'_, OT, H, Node>> for Node {
    fn eq(&self, other: &FatNode<'_, OT, H, Node>) -> bool {
        self == &other.node
    }
}

impl<N: PartialEq, OT1, OT2, H1, H2> PartialEq<FatNode<'_, OT1, H1, N>>
    for FatNode<'_, OT2, H2, N>
{
    fn eq(&self, other: &FatNode<'_, OT1, H1, N>) -> bool {
        self.node == other.node
    }
}

impl<N: Eq, OT, H> Eq for FatNode<'_, OT, H, N> {}

impl<OT, H> PartialOrd<Node> for FatNode<'_, OT, H> {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        self.node.partial_cmp(other)
    }
}

impl<OT, H> PartialOrd<FatNode<'_, OT, H>> for Node {
    fn partial_cmp(&self, other: &FatNode<'_, OT, H>) -> Option<Ordering> {
        self.partial_cmp(&other.node)
    }
}

impl<N: PartialOrd, OT1, OT2, H1, H2> PartialOrd<FatNode<'_, OT1, H1, N>>
    for FatNode<'_, OT2, H2, N>
{
    fn partial_cmp(&self, other: &FatNode<'_, OT1, H1, N>) -> Option<Ordering> {
        self.node.partial_cmp(&other.node)
    }
}

impl<OT, H, N: Ord> Ord for FatNode<'_, OT, H, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.cmp(&other.node)
    }
}

impl<OT, H, N: Hash> Hash for FatNode<'_, OT, H, N> {
    fn hash<HA: std::hash::Hasher>(&self, state: &mut HA) {
        self.node.hash(state);
    }
}

impl<OT, H: HugrView + ?Sized> AsRef<OT> for FatNode<'_, OT, H, H::Node>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    fn as_ref(&self) -> &OT {
        self.hugr.get_optype(self.node).try_into().ok().unwrap()
    }
}

impl<OT, H: HugrView + ?Sized> Deref for FatNode<'_, OT, H, H::Node>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    type Target = OT;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<OT, H> Copy for FatNode<'_, OT, H> {}

impl<OT, H> Clone for FatNode<'_, OT, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<OT: fmt::Display, H: HugrView + ?Sized> fmt::Display for FatNode<'_, OT, H, H::Node>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("N<{}:{}>", self.as_ref(), self.node))
    }
}

impl<OT, H, N: NodeIndex> NodeIndex for FatNode<'_, OT, H, N> {
    fn index(self) -> usize {
        self.node.index()
    }
}

impl<OT, H> NodeIndex for &FatNode<'_, OT, H> {
    fn index(self) -> usize {
        self.node.index()
    }
}

/// An extension trait for [`HugrView`] which provides methods that delegate to
/// [`HugrView`] and then return the result in [`FatNode`] form. See for example
/// [`FatExt::fat_io`].
///
/// TODO: Add the remaining [`HugrView`] equivalents that make sense.
pub trait FatExt: HugrView {
    /// Try to create a specific [`FatNode`] for a given [Node].
    fn try_fat<OT>(&self, node: Self::Node) -> Option<FatNode<'_, OT, Self, Self::Node>>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        FatNode::try_new(self, node)
    }

    /// Create a general [`FatNode`] for a given [Node].
    fn fat_optype(&self, node: Self::Node) -> FatNode<'_, OpType, Self, Self::Node> {
        FatNode::new_optype(self, node)
    }

    /// Try to create [Input] and [Output] [`FatNode`]s for a given [Node]. This
    /// will succeed only for `DataFlow` Parent Nodes.
    #[allow(clippy::type_complexity)]
    fn fat_io(
        &self,
        node: Self::Node,
    ) -> Option<(
        FatNode<'_, Input, Self, Self::Node>,
        FatNode<'_, Output, Self, Self::Node>,
    )> {
        self.fat_optype(node).get_io()
    }

    /// Create general [`FatNode`]s for each of a [Node]'s children.
    fn fat_children(
        &self,
        node: Self::Node,
    ) -> impl Iterator<Item = FatNode<'_, OpType, Self, Self::Node>> {
        self.children(node).map(|x| self.fat_optype(x))
    }

    /// Try to create a specific [`FatNode`] for the root of a [`HugrView`].
    fn fat_root(&self) -> Option<FatNode<'_, Module, Self, Self::Node>> {
        self.try_fat(self.module_root())
    }

    /// Try to create a specific [`FatNode`] for the entrypoint of a [`HugrView`].
    fn fat_entrypoint<OT>(&self) -> Option<FatNode<'_, OT, Self, Self::Node>>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        self.try_fat(self.entrypoint())
    }
}

impl<H: HugrView> FatExt for H {}
