use std::collections::HashMap;
use std::hash::Hash;

use derive_more::derive::From;

use crate::{HugrView, IncomingPort, OutgoingPort};

/// A map between incoming ports of two HUGRs.
///
/// When querying the image of a port, the [`BoundaryMap`] is passed references
/// to the source and destination HUGRs. This means that the maps do not
/// necessarily need to store the map explicitly, but may rely on traversing the
/// HUGRs instead.
///
/// A [`BoundaryMap`] furthermore defines the set of nodes in the source HUGR
/// that it depends on for the map to be well-defined: it is guaranteed that
/// the source HUGR passed to [`BoundaryMap::map_port`] will contain all the
/// nodes returned by [`BoundaryMap::required_nodes`].
///
/// ## Provided implementations
///
/// - `HashMap<(SrcNode, IncomingPort), (DstNode, IncomingPort)>`: a simple map
///   from incoming ports of the source HUGR to incoming ports of the
///   destination HUGR.
/// - `HashMap<(SrcNode, OutgoingPort), (DstNode, IncomingPort)>`: a map from
///   all incoming ports attached to an outgoing port in the source HUGR to the
///   given incoming port in the destination HUGR. The set of required nodes in
///   the source HUGR is the set of nodes in the keys of the HashMap; this is
///   NOT the same set as the domain of definition of the map!
/// - [`OutputNodeBoundaryMap<SrcNode>`]: a map on incoming ports of the source
///   HUGR to incoming ports of the output node of the destination HUGR. It
///   implements From<`HashMap<(SrcNode, IncomingPort), IncomingPort>>.
pub trait BoundaryMap<SrcNode, DstNode> {
    /// Map an incoming port of the source HUGR to an incoming port of the
    /// destination HUGR.
    fn map_port(
        &self,
        src_node: SrcNode,
        src_port: IncomingPort,
        src_hugr: &impl HugrView<Node = SrcNode>,
        dst_hugr: &impl HugrView<Node = DstNode>,
    ) -> Option<(DstNode, IncomingPort)>;

    /// The set of nodes in the source HUGR that must be present for the map to
    /// be well-defined.
    fn required_nodes(&self) -> impl Iterator<Item = SrcNode>;

    /// All keys in the domain of definition of the map.
    fn all_keys<'a>(
        &'a self,
        src_hugr: &'a impl HugrView<Node = SrcNode>,
    ) -> impl Iterator<Item = (SrcNode, IncomingPort)> + 'a
    where
        SrcNode: 'a;

    /// All values in the image of the map.
    ///
    /// This iterates over all keys and computes the image of each key under
    /// the map. Implementers may want to provide a more efficient
    /// implementation of this method.
    fn all_values<'a>(
        &'a self,
        src_hugr: &'a impl HugrView<Node = SrcNode>,
        dst_hugr: &'a impl HugrView<Node = DstNode>,
    ) -> impl Iterator<Item = (DstNode, IncomingPort)> + 'a
    where
        SrcNode: Copy + 'a,
        DstNode: 'a,
    {
        self.iter(src_hugr, dst_hugr).map(|(_, dst)| dst)
    }

    /// Iterate over all key-value pairs in the map.
    fn iter<'a>(
        &'a self,
        src_hugr: &'a impl HugrView<Node = SrcNode>,
        dst_hugr: &'a impl HugrView<Node = DstNode>,
    ) -> impl Iterator<Item = ((SrcNode, IncomingPort), (DstNode, IncomingPort))> + 'a
    where
        SrcNode: Copy + 'a,
    {
        self.all_keys(src_hugr).filter_map(|(src, src_port)| {
            let (dst, dst_port) = self.map_port(src, src_port, src_hugr, dst_hugr)?;
            Some(((src, src_port), (dst, dst_port)))
        })
    }
}

impl<Src, Dst> BoundaryMap<Src, Dst> for HashMap<(Src, IncomingPort), (Dst, IncomingPort)>
where
    Src: Eq + Hash + Copy,
    Dst: Copy,
{
    fn map_port(
        &self,
        src_node: Src,
        src_port: IncomingPort,
        _src_hugr: &impl HugrView<Node = Src>,
        _dst_hugr: &impl HugrView<Node = Dst>,
    ) -> Option<(Dst, IncomingPort)> {
        self.get(&(src_node, src_port)).copied()
    }

    fn required_nodes(&self) -> impl Iterator<Item = Src> {
        self.keys().map(|(src, _)| *src)
    }

    fn all_keys<'a>(
        &'a self,
        _src_hugr: &'a impl HugrView<Node = Src>,
    ) -> impl Iterator<Item = (Src, IncomingPort)> + 'a
    where
        Src: 'a,
    {
        self.keys().copied()
    }
}

impl<Src, Dst> BoundaryMap<Src, Dst> for HashMap<(Src, OutgoingPort), (Dst, IncomingPort)>
where
    Src: Eq + Hash + Copy,
    Dst: Copy,
{
    fn map_port(
        &self,
        src_node: Src,
        src_port: IncomingPort,
        src_hugr: &impl HugrView<Node = Src>,
        _dst_hugr: &impl HugrView<Node = Dst>,
    ) -> Option<(Dst, IncomingPort)> {
        let (out_node, out_port) = src_hugr
            .single_linked_output(src_node, src_port)
            .expect("BoundaryMap: expected DFG Hugr");
        self.get(&(out_node, out_port)).copied()
    }

    fn required_nodes(&self) -> impl Iterator<Item = Src> {
        self.keys().map(|(src, _)| *src)
    }

    fn all_keys<'a>(
        &'a self,
        src_hugr: &'a impl HugrView<Node = Src>,
    ) -> impl Iterator<Item = (Src, IncomingPort)> + 'a
    where
        Src: 'a,
    {
        self.keys()
            .flat_map(|&(src, port)| src_hugr.linked_inputs(src, port))
    }
}

/// A map on incoming ports of the source HUGR to incoming ports of the output
/// node of the destination HUGR.
///
/// This map is implemented as a `HashMap<(SrcNode, IncomingPort),
/// IncomingPort>`. It implements [`BoundaryMap`] and can be converted from a
/// `HashMap<(SrcNode, IncomingPort), IncomingPort>`.
#[derive(Debug, Clone, From)]
pub struct OutputNodeBoundaryMap<SrcNode>(HashMap<(SrcNode, IncomingPort), IncomingPort>);

impl<Src, Dst> BoundaryMap<Src, Dst> for OutputNodeBoundaryMap<Src>
where
    Src: Eq + Hash + Copy,
    Dst: Copy,
{
    fn map_port(
        &self,
        src_node: Src,
        src_port: IncomingPort,
        _src_hugr: &impl HugrView<Node = Src>,
        dst_hugr: &impl HugrView<Node = Dst>,
    ) -> Option<(Dst, IncomingPort)> {
        let [_, output] = dst_hugr
            .get_io(dst_hugr.root())
            .expect("dst_hugr must be a DFG Hugr");
        let dst_port = *self.0.get(&(src_node, src_port))?;
        Some((output, dst_port))
    }

    fn required_nodes(&self) -> impl Iterator<Item = Src> {
        self.0.keys().map(|(src, _)| *src)
    }

    fn all_keys<'a>(
        &'a self,
        _src_hugr: &'a impl HugrView<Node = Src>,
    ) -> impl Iterator<Item = (Src, IncomingPort)> + 'a
    where
        Src: 'a,
    {
        self.0.keys().copied()
    }
}
