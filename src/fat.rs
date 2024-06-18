//! We define a type [FatNode], named for analogy with a "fat pointer".
//!
//! We define a trait [FatExt], an extension trait for [HugrView]. It provides
//! methods that return [FatNode]s rather than [Node]s.
use std::{cmp::Ordering, hash::Hash, marker::PhantomData, ops::Deref};

use hugr::{
    ops::{DataflowBlock, ExitBlock, Input, NamedOp, OpType, Output, CFG},
    types::Type,
    Hugr, HugrView, IncomingPort, Node, NodeIndex, OutgoingPort,
};
use itertools::Itertools as _;

/// A Fat Node is a [Node] along with a reference to the [HugrView] whence it
/// originates. It carries a type `OT`, the [OpType] of that node. `OT` may be
/// general, i.e. exactly [OpType], or specifec, e.g. [FuncDefn].
///
/// We provide a [Deref<Target=OT>] impl, so it can be used in place of `OT`.
///
/// We provide [PartialEq], [Eq], [PartialOrd], [Ord], [Hash], so that this type
/// can be used in containers. Note that these implementations use only the
/// stored node, so will silently malfunction if used with [FatNode]s from
/// different base [Hugr]s. Note that [Node] has this same behaviour.
///
/// [FuncDefn]: [hugr::ops::FuncDefn]
#[derive(Debug, Copy)]
pub struct FatNode<'c, OT = OpType, H = Hugr>
where
    H: ?Sized,
{
    hugr: &'c H,
    node: Node,
    marker: PhantomData<OT>,
}

impl<'c, OT: 'c, H: HugrView + ?Sized> FatNode<'c, OT, H>
where
    &'c OpType: TryInto<&'c OT>,
{
    /// Create a `FatNode` from a [HugrView] and a [Node].
    ///
    /// Panics if the node is not valid in the `Hugr` or if it's `get_optype` is
    /// not an `OT`.
    ///
    /// Note that while we do check the type of the node's `get_optype`, we
    /// do not verify that it is actually equal to `ot`.
    pub fn new(hugr: &'c H, node: Node, #[allow(unused)] ot: &OT) -> Self {
        assert!(hugr.valid_node(node));
        assert!(TryInto::<&'c OT>::try_into(hugr.get_optype(node)).is_ok());
        // We don't actually check `ot == hugr.get_optype(node)` so as to not require OT: PartialEq`
        Self {
            hugr,
            node,
            marker: PhantomData,
        }
    }

    /// Tries to create a `FatNode` from a [HugrView] and a [Node].
    ///
    /// If the node is invalid, or if its `get_optype` is not `OT`, returns
    /// `None`.
    pub fn try_new(hugr: &'c H, node: Node) -> Option<Self> {
        (hugr.valid_node(node)).then_some(())?;
        Some(Self::new(
            hugr,
            node,
            hugr.get_optype(node).try_into().ok()?,
        ))
    }

    /// Gets the [OpType] of the `FatNode`.
    pub fn get(&self) -> &'c OT {
        self.hugr.get_optype(self.node).try_into().ok().unwrap()
    }
}

impl<'c, OT, H> FatNode<'c, OT, H> {
    /// Gets the [Node] of the `FatNode`.
    pub fn node(&self) -> Node {
        self.node
    }

    /// Gets the [HugrView] of the `FatNode`.
    pub fn hugr(&self) -> &'c H {
        self.hugr
    }
}

impl<'c, H: HugrView + ?Sized> FatNode<'c, OpType, H> {
    /// Creates a new general `FatNode` from a [HugrView] and a [Node].
    ///
    /// Panics if the node is not valid in the [Hugr].
    pub fn new_optype(hugr: &'c H, node: Node) -> Self {
        assert!(hugr.valid_node(node));
        FatNode::new(hugr, node, hugr.get_optype(node))
    }

    /// Tries to downcast a general `FatNode` into a specific `OT`.
    pub fn try_into_ot<OT: 'c>(&self) -> Option<FatNode<'c, OT, H>>
    where
        &'c OpType: TryInto<&'c OT>,
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
    pub fn into_ot<OT: PartialEq + 'c>(self, ot: &OT) -> FatNode<'c, OT, H>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        FatNode::new(self.hugr, self.node, ot)
    }
}

impl<'c, OT, H: HugrView + ?Sized> FatNode<'c, OT, H> {
    /// If there is exactly one OutgoingPort connected to this IncomingPort,
    /// return it and its node.
    pub fn single_linked_output(
        &self,
        port: IncomingPort,
    ) -> Option<(FatNode<'c, OpType, H>, OutgoingPort)> {
        self.hugr
            .single_linked_output(self.node, port)
            .map(|(n, p)| (FatNode::new_optype(self.hugr, n), p))
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    pub fn out_value_types(&self) -> impl Iterator<Item = (OutgoingPort, Type)> + 'c {
        self.hugr.out_value_types(self.node)
    }

    /// Iterator over all incoming ports that have Value type, along
    /// with corresponding types.
    pub fn in_value_types(&self) -> impl Iterator<Item = (IncomingPort, Type)> + 'c {
        self.hugr.in_value_types(self.node)
    }

    /// Return iterator over the direct children of node.
    pub fn children(&self) -> impl Iterator<Item = FatNode<'c, OpType, H>> + 'c {
        self.hugr
            .children(self.node)
            .map(|n| FatNode::new_optype(self.hugr, n))
    }

    /// Get the input and output child nodes of a dataflow parent.
    /// If the node isn't a dataflow parent, then return None
    pub fn get_io(&self) -> Option<(FatNode<'c, Input, H>, FatNode<'c, Output, H>)> {
        let [i, o] = self.hugr.get_io(self.node)?;
        Some((
            FatNode::try_new(self.hugr, i)?,
            FatNode::try_new(self.hugr, o)?,
        ))
    }

    pub fn node_outputs(&self) -> impl Iterator<Item = OutgoingPort> + '_ {
        self.hugr.node_outputs(self.node)
    }

    pub fn output_neighbours(&self) -> impl Iterator<Item = FatNode<'c, OpType, H>> + '_ {
        self.hugr
            .output_neighbours(self.node)
            .map(|n| FatNode::new_optype(self.hugr, n))
    }

    /// Create a general `FatNode` from a specific one.
    pub fn generalise(self) -> FatNode<'c, OpType, H>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
        OT: 'c,
    {
        // guaranteed to be valid becasue self is valid
        FatNode {
            hugr: self.hugr,
            node: self.node,
            marker: PhantomData,
        }
    }
}

impl<'c, H: HugrView> FatNode<'c, CFG, H> {
    /// Returns the entry and exit nodes of a CFG.
    ///
    /// These are guaranteed to exist the `Hugr` is valid. Panics if they do not
    /// exist.
    pub fn get_entry_exit(&self) -> (FatNode<'c, DataflowBlock, H>, FatNode<'c, ExitBlock, H>) {
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

impl<'c, OT, H> PartialEq<Node> for FatNode<'c, OT, H> {
    fn eq(&self, other: &Node) -> bool {
        &self.node == other
    }
}

impl<'c, OT, H> PartialEq<FatNode<'c, OT, H>> for Node {
    fn eq(&self, other: &FatNode<'c, OT, H>) -> bool {
        self == &other.node
    }
}

impl<'c, 'd, OT1, OT2, H1, H2> PartialEq<FatNode<'d, OT1, H1>> for FatNode<'c, OT2, H2> {
    fn eq(&self, other: &FatNode<'d, OT1, H1>) -> bool {
        self.node == other.node
    }
}

impl<'c, OT, H> Eq for FatNode<'c, OT, H> {}

impl<'c, OT, H> PartialOrd<Node> for FatNode<'c, OT, H> {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        self.node.partial_cmp(other)
    }
}

impl<'c, OT, H> PartialOrd<FatNode<'c, OT, H>> for Node {
    fn partial_cmp(&self, other: &FatNode<'c, OT, H>) -> Option<Ordering> {
        self.partial_cmp(&other.node)
    }
}

impl<'c, 'd, OT1, OT2, H1, H2> PartialOrd<FatNode<'d, OT1, H1>> for FatNode<'c, OT2, H2> {
    fn partial_cmp(&self, other: &FatNode<'d, OT1, H1>) -> Option<Ordering> {
        self.partial_cmp(&other.node)
    }
}

impl<'c, OT, H> Ord for FatNode<'c, OT, H> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.cmp(&other.node)
    }
}

impl<'c, OT, H> Hash for FatNode<'c, OT, H> {
    fn hash<HA: std::hash::Hasher>(&self, state: &mut HA) {
        self.node.hash(state);
    }
}

impl<'c, OT: 'c, H: HugrView + ?Sized> AsRef<OT> for FatNode<'c, OT, H>
where
    for<'a> &'a OpType: TryInto<&'a OT>,
{
    fn as_ref(&self) -> &OT {
        self.get()
    }
}

impl<'c, OT, H: HugrView + ?Sized> Deref for FatNode<'c, OT, H>
where
    &'c OpType: TryInto<&'c OT>,
    <&'c OpType as TryInto<&'c OT>>::Error: std::fmt::Debug,
    OT: 'c,
{
    type Target = OT;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'c, OT, H> Clone for FatNode<'c, OT, H> {
    fn clone(&self) -> Self {
        Self {
            hugr: self.hugr,
            node: self.node,
            marker: self.marker,
        }
    }
}

impl<'c, OT: NamedOp, H: HugrView + ?Sized> std::fmt::Display for FatNode<'c, OT, H>
where
    &'c OpType: TryInto<&'c OT>,
    <&'c OpType as TryInto<&'c OT>>::Error: std::fmt::Debug,
    OT: 'c,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("N<{}:{}>", self.get().name(), self.node))
    }
}

impl<'c, OT, H> NodeIndex for FatNode<'c, OT, H> {
    fn index(self) -> usize {
        self.node.index()
    }
}

impl<'c, OT, H> NodeIndex for &FatNode<'c, OT, H> {
    fn index(self) -> usize {
        self.node.index()
    }
}

/// An extension trait for [HugrView] which provides methods that delegate to
/// [HugrView] and then return the result in [FatNode] form. See for example
/// [FatExt::fat_io].
///
/// TODO: Add the remaining [HugrView] equivalents that make sense.
pub trait FatExt: HugrView {
    /// Try to create a specific [FatNode] for a given [Node].
    fn try_fat<'c, OT: 'c>(&'c self, node: Node) -> Option<FatNode<'c, OT, Self>>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        FatNode::try_new(self, node)
    }

    /// Create a general [FatNode] for a given [Node].
    fn fat_optype(&self, node: Node) -> FatNode<OpType, Self> {
        FatNode::new_optype(self, node)
    }

    /// Try to create [Input] and [Output] [FatNode]s for a given [Node]. This
    /// will succeed only for DataFlow Parent Nodes.
    fn fat_io(&self, node: Node) -> Option<(FatNode<Input, Self>, FatNode<Output, Self>)> {
        self.fat_optype(node).get_io()
    }

    /// Create general [FatNode]s for each of a [Node]'s children.
    fn fat_children(&self, node: Node) -> impl Iterator<Item = FatNode<OpType, Self>> {
        self.children(node).map(|x| self.fat_optype(x))
    }

    /// Try to create a specific [FatNode] for the root of a [HugrView].
    fn fat_root<'c, OT: 'c>(&'c self) -> Option<FatNode<'c, OT, Self>>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
        Self: 'c,
    {
        self.try_fat(self.root())
    }
}

impl<H: HugrView> FatExt for H {}
