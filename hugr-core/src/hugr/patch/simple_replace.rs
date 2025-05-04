//! Implementation of the `SimpleReplace` operation.

use std::collections::{BTreeSet, HashMap};

use crate::core::HugrNode;
use crate::hugr::hugrmut::InsertionResult;
use crate::hugr::views::SiblingSubgraph;
use crate::hugr::{HugrMut, HugrView};
use crate::ops::{OpTag, OpTrait, OpType};
use crate::{Hugr, IncomingPort, Node, OutgoingPort};

use itertools::{Either, Itertools};

use thiserror::Error;

use super::boundary_map::OutputNodeBoundaryMap;
use super::inline_dfg::InlineDFGError;
use super::{
    BoundaryMap, BoundaryPort, HostPort, PatchHugrMut, PatchVerification, ReplacementPort,
};

/// The default boundary map for the input boundary of [`SimpleReplacement`].
///
/// Maps (target ports of edges from the Input node of `replacement`) to (target
/// ports of edges from nodes not in `subgraph` to nodes in `subgraph`).
pub type DefaultInMap<HostNode> = HashMap<(Node, IncomingPort), (HostNode, IncomingPort)>;

/// The default boundary map for the output boundary of [`SimpleReplacement`].
///
/// Maps (incoming ports of edges from nodes in `subgraph` to nodes not in
/// `subgraph`) to (input ports of the Output node of `replacement`).
///
/// The destination node of the output boundary map is always the output node of
/// the replacement hugr.
pub type DefaultOutMap<Src> = OutputNodeBoundaryMap<Src>;

/// Specification of a simple replacement operation.
///
/// # Type parameters
///
/// - `HostNode`: The type of nodes in the host hugr (defaults to `Node`).
/// - `InMap`: The type of the input boundary map (defaults to
///   [`DefaultInMap`]).
/// - `OutMap`: The type of the output boundary map (defaults to
///   [`DefaultOutMap`]).
#[derive(Debug, Clone)]
pub struct SimpleReplacement<
    HostNode = Node,
    InMap = DefaultInMap<HostNode>,
    OutMap = DefaultOutMap<HostNode>,
> {
    /// The subgraph of the host hugr to be replaced.
    subgraph: SiblingSubgraph<HostNode>,
    /// A hugr with DFG root (consisting of replacement nodes).
    replacement: Hugr,
    /// The input boundary, mapping incoming ports connected to the input node
    /// of the replacement HUGR to incoming ports at the input boundary of the
    /// host subgraph.
    ///
    /// ## Domain and image
    ///
    /// The domain of the input boundary map, as returned by
    /// [`BoundaryMap::all_keys`], must be the set of all incoming ports of
    /// edges connected to the input node of the replacement HUGR.
    ///
    /// The image of the input boundary map, as returned by
    /// [`BoundaryMap::all_values`], must be the incoming boundary of the
    /// subgraph, as returned by [`SiblingSubgraph::incoming_ports`].
    nu_inp: InMap,
    /// The output boundary, mapping incoming ports at the output boundary of
    /// the host subgraph to the incoming ports of the output node of the
    /// replacement HUGR.
    ///
    /// ## Domain and image
    ///
    /// The domain of the output boundary map, as returned by
    /// [`BoundaryMap::all_keys`], must be the set of all incoming ports of
    /// edges connected to the outgoing ports of subgrahp, as returned by
    /// [`SiblingSubgraph::outgoing_ports`].
    ///
    /// The image of the output boundary map, as returned by
    /// [`BoundaryMap::all_values`], must be the set of all incoming ports of
    /// the output node of the replacement HUGR.
    nu_out: OutMap,
}

impl<
        HostNode: HugrNode,
        InMap: BoundaryMap<Node, HostNode>,
        OutMap: BoundaryMap<HostNode, Node>,
    > SimpleReplacement<HostNode, InMap, OutMap>
{
    /// Create a new [`SimpleReplacement`] specification.
    ///
    /// Return an error if the provided boundary is not valid. See below for the
    /// validity conditions.
    ///
    /// ### Suggested boundary map types
    ///
    /// For the input boundary map, a typical choice is a
    /// `HashMap<(Node, P), (HostNode, IncomingPort)>`, where `P` can either be
    /// [`IncomingPort`] or [`OutgoingPort`] (defaults to `IncomingPort`).
    ///
    /// The output boundary map has the property that all nodes in the image
    /// must be the output node of `replacement`. The special type
    /// [`OutputNodeBoundaryMap`] provides an implementation of such maps, and
    /// can be converted from a `HashMap<(HostNode, P), IncomingPort>`.
    ///
    /// ### Input boundary validity
    ///
    /// The domain of the input boundary map, as returned by
    /// [`BoundaryMap::all_keys`], must be the set of all incoming ports of
    /// edges connected to the input node of `replacement`.
    ///
    /// The image of the input boundary map, as returned by
    /// [`BoundaryMap::all_values`], must be the incoming boundary of the
    /// `subgraph`, as returned by [`SiblingSubgraph::incoming_ports`].
    ///
    /// ### Output boundary validity
    ///
    /// The domain of the output boundary map, as returned by
    /// [`BoundaryMap::all_keys`], must be the set of all incoming ports of
    /// edges connected to the outgoing ports of `subgraph`, as returned by
    /// [`SiblingSubgraph::outgoing_ports`].
    ///
    /// The image of the output boundary map, as returned by
    /// [`BoundaryMap::all_values`], must be the set of all incoming ports of
    /// the output node of `replacement`.
    #[inline]
    pub fn try_new(
        subgraph: SiblingSubgraph<HostNode>,
        host: &impl HugrView<Node = HostNode>,
        replacement: Hugr,
        nu_inp: InMap,
        nu_out: OutMap,
    ) -> Result<Self, InvalidBoundaryError<HostNode>> {
        check_valid_boundary(&subgraph, host, &replacement, &nu_inp, &nu_out)?;
        Ok(Self {
            subgraph,
            replacement,
            nu_inp,
            nu_out,
        })
    }

    /// Create a new [`SimpleReplacement`] specification without checking the
    /// validity of the boundary maps.
    ///
    /// Use [`SimpleReplacement::try_new`] to check the validity of the boundary
    /// maps.
    pub fn new_unsafe(
        subgraph: SiblingSubgraph<HostNode>,
        replacement: Hugr,
        nu_inp: InMap,
        nu_out: impl Into<OutMap>,
    ) -> Self {
        Self {
            subgraph,
            replacement,
            nu_inp,
            nu_out: nu_out.into(),
        }
    }

    /// The replacement hugr.
    #[inline]
    pub fn replacement(&self) -> &Hugr {
        &self.replacement
    }

    /// Consume self and return the replacement hugr.
    #[inline]
    pub fn into_replacement(self) -> Hugr {
        self.replacement
    }

    /// Subgraph to be replaced.
    #[inline]
    pub fn subgraph(&self) -> &SiblingSubgraph<HostNode> {
        &self.subgraph
    }

    /// Check if the replacement can be applied to the given hugr.
    pub fn is_valid_rewrite(
        &self,
        h: &impl HugrView<Node = HostNode>,
    ) -> Result<(), SimpleReplacementError> {
        let parent = self.subgraph.get_parent(h);

        // 1. Check the parent node exists and is a DataflowParent.
        if !OpTag::DataflowParent.is_superset(h.get_optype(parent).tag()) {
            return Err(SimpleReplacementError::InvalidParentNode());
        }

        // 2. Check that all the to-be-removed nodes are children of it and are leaves.
        for node in self.subgraph.nodes() {
            if h.get_parent(*node) != Some(parent) || h.children(*node).next().is_some() {
                return Err(SimpleReplacementError::InvalidRemovedNode());
            }
        }

        Ok(())
    }

    /// Get the input and output nodes of the replacement hugr.
    pub fn get_replacement_io(&self) -> Result<[Node; 2], SimpleReplacementError> {
        self.replacement
            .get_io(self.replacement.root())
            .ok_or(SimpleReplacementError::InvalidParentNode())
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `host` to incoming ports in `self.replacement`.
    ///
    /// The incoming ports returned are always connected to outputs of
    /// the [`OpTag::Input`] node of `self.replacement`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `host` and the second is a port in `self.replacement`.
    pub fn incoming_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            HostPort<HostNode, OutgoingPort>,
            ReplacementPort<IncomingPort>,
        ),
    > + 'a {
        // For each p = self.nu_inp[q] such that q is not an Output port,
        // there will be an edge from the predecessor of p to (the new copy of) q.
        self.nu_inp
            .iter(&self.replacement, host)
            .filter(|&((rep_inp_node, _), _)| {
                self.replacement.get_optype(rep_inp_node).tag() != OpTag::Output
            })
            .map(
                |((rep_inp_node, rep_inp_port), (rem_inp_node, rem_inp_port))| {
                    // add edge from predecessor of (s_inp_node, s_inp_port) to (new_inp_node,
                    // n_inp_port)
                    let (rem_inp_pred_node, rem_inp_pred_port) = host
                        .single_linked_output(rem_inp_node, rem_inp_port)
                        .unwrap();
                    (
                        HostPort(rem_inp_pred_node, rem_inp_pred_port),
                        ReplacementPort(rep_inp_node, rep_inp_port),
                    )
                },
            )
    }

    /// Get all edges that the replacement would add from outgoing ports in
    /// `self.replacement` to incoming ports in `host`.
    ///
    /// The outgoing ports returned are always connected to inputs of
    /// the [`OpTag::Output`] node of `self.replacement`.
    ///
    /// For each pair in the returned vector, the first element is a port in
    /// `self.replacement` and the second is a port in `host`.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn outgoing_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            ReplacementPort<OutgoingPort>,
            HostPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        // For each q = self.nu_out[p] such that the predecessor of q is not an Input
        // port, there will be an edge from (the new copy of) the predecessor of
        // q to p.
        self.nu_out.iter(host, &self.replacement).filter_map(
            move |((rem_out_node, rem_out_port), (rep_out_node, rep_out_port))| {
                let (rep_out_pred_node, rep_out_pred_port) = self
                    .replacement
                    .single_linked_output(rep_out_node, rep_out_port)
                    .unwrap();
                (self.replacement.get_optype(rep_out_pred_node).tag() != OpTag::Input).then_some({
                    (
                        // the new output node will be updated after insertion
                        ReplacementPort(rep_out_pred_node, rep_out_pred_port),
                        HostPort(rem_out_node, rem_out_port),
                    )
                })
            },
        )
    }

    /// Get all edges that the replacement would add between ports in `host`.
    ///
    /// These correspond to direct edges between the input and output nodes
    /// in the replacement graph.
    ///
    /// For each pair in the returned vector, the both ports are in `host`.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn host_to_host_boundary<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            HostPort<HostNode, OutgoingPort>,
            HostPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        let [_, replacement_output_node] = self.get_replacement_io().expect("replacement is a DFG");

        // For each q = self.nu_out[p1], p0 = self.nu_inp[q], add an edge from the
        // predecessor of p0 to p1.
        self.nu_out.iter(host, &self.replacement).filter_map(
            move |((rem_out_node, rem_out_port), (rep_out_node, rep_out_port))| {
                assert_eq!(replacement_output_node, rep_out_node);
                self.nu_inp
                    .map_port(rep_out_node, rep_out_port, &self.replacement, host)
                    .map(move |(rem_inp_node, rem_inp_port)| {
                        let (rem_inp_pred_node, rem_inp_pred_port) = host
                            .single_linked_output(rem_inp_node, rem_inp_port)
                            .unwrap();
                        (
                            HostPort(rem_inp_pred_node, rem_inp_pred_port),
                            HostPort(rem_out_node, rem_out_port),
                        )
                    })
            },
        )
    }

    /// Get the incoming port at the output node of `self.replacement` that
    /// corresponds to the given host output port.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_host_output(
        &self,
        port: impl Into<HostPort<HostNode, IncomingPort>>,
        host: &impl HugrView<Node = HostNode>,
    ) -> Option<ReplacementPort<IncomingPort>> {
        let HostPort(node, port) = port.into();
        self.nu_out
            .map_port(node, port, host, &self.replacement)
            .map(|(rep_out_node, rep_out_port)| ReplacementPort(rep_out_node, rep_out_port))
    }

    /// Get the incoming port in `subgraph` that corresponds to the given
    /// replacement input port.
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn map_replacement_input(
        &self,
        port: impl Into<ReplacementPort<IncomingPort>>,
        host: &impl HugrView<Node = HostNode>,
    ) -> Option<HostPort<HostNode, IncomingPort>> {
        let ReplacementPort(node, port) = port.into();
        self.nu_inp
            .map_port(node, port, &self.replacement, host)
            .map(Into::into)
    }

    /// Get all edges that the replacement would add between `host` and
    /// `self.replacement`.
    ///
    /// This is equivalent to chaining the results of
    /// [`Self::incoming_boundary`], [`Self::outgoing_boundary`], and
    /// [`Self::host_to_host_boundary`].
    ///
    /// This panics if self.replacement is not a DFG.
    pub fn all_boundary_edges<'a>(
        &'a self,
        host: &'a impl HugrView<Node = HostNode>,
    ) -> impl Iterator<
        Item = (
            BoundaryPort<HostNode, OutgoingPort>,
            BoundaryPort<HostNode, IncomingPort>,
        ),
    > + 'a {
        let incoming_boundary = self
            .incoming_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));
        let outgoing_boundary = self
            .outgoing_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));
        let host_to_host_boundary = self
            .host_to_host_boundary(host)
            .map(|(src, tgt)| (src.into(), tgt.into()));

        incoming_boundary
            .chain(outgoing_boundary)
            .chain(host_to_host_boundary)
    }
}

fn check_valid_boundary<HostNode, InMap, OutMap>(
    subgraph: &SiblingSubgraph<HostNode>,
    host: &impl HugrView<Node = HostNode>,
    replacement: &Hugr,
    nu_inp: &InMap,
    nu_out: &OutMap,
) -> Result<(), InvalidBoundaryError<HostNode>>
where
    HostNode: HugrNode,
    InMap: BoundaryMap<Node, HostNode>,
    OutMap: BoundaryMap<HostNode, Node>,
{
    check_valid_input_boundary(subgraph, host, replacement, nu_inp)?;
    check_valid_output_boundary(subgraph, host, replacement, nu_out)?;

    Ok(())
}

fn check_valid_input_boundary<HostNode, InMap>(
    subgraph: &SiblingSubgraph<HostNode>,
    host: &impl HugrView<Node = HostNode>,
    replacement: &Hugr,
    nu_inp: &InMap,
) -> Result<(), InvalidBoundaryError<HostNode>>
where
    HostNode: HugrNode,
    InMap: BoundaryMap<Node, HostNode>,
{
    // All expected input ports in replacement
    let exp_ports_repl = {
        let [repl_inp, _] = replacement
            .get_io(replacement.root())
            .expect("replacement is a DFG");
        replacement.all_linked_inputs(repl_inp).collect()
    };

    // Check that ports in nu_inp's domain match one-to-one to input ports in
    // replacement
    match check_one_to_one(nu_inp.all_keys(&replacement), exp_ports_repl) {
        Ok(()) => (),
        Err(Either::Left((node, port))) => {
            return Err(InvalidBoundaryError::InvalidInputPort(
                WhichNode::Replacement(node),
                port,
            ));
        }
        Err(Either::Right(missing)) => {
            return Err(InvalidBoundaryError::MissingInputPorts(
                missing
                    .into_iter()
                    .map(|(node, port)| (WhichNode::Replacement(node), port))
                    .collect(),
            ));
        }
    }

    // All expected input ports in subgraph
    let exp_ports_subgraph: BTreeSet<_> = subgraph
        .incoming_ports()
        .iter()
        .flatten()
        .copied()
        .collect();

    // Check that ports in nu_inp's image are contained within the incoming ports
    // of the subgraph (and linear ports map one-to-one)
    match check_contained(
        nu_inp.all_values(replacement, host),
        exp_ports_subgraph,
        |(node, port)| {
            if let Some(kind) = host.get_optype(node).port_kind(port) {
                kind.is_linear()
            } else {
                false
            }
        },
    ) {
        Ok(()) => (),
        Err(Either::Left((node, port))) => {
            return Err(InvalidBoundaryError::InvalidInputPort(
                WhichNode::Host(node),
                port,
            ));
        }
        Err(Either::Right(missing)) => {
            return Err(InvalidBoundaryError::MissingInputPorts(
                missing
                    .into_iter()
                    .map(|(node, port)| (WhichNode::Host(node), port))
                    .collect(),
            ));
        }
    }

    Ok(())
}

fn check_valid_output_boundary<HostNode, OutMap>(
    subgraph: &SiblingSubgraph<HostNode>,
    host: &impl HugrView<Node = HostNode>,
    replacement: &Hugr,
    nu_out: &OutMap,
) -> Result<(), InvalidBoundaryError<HostNode>>
where
    HostNode: HugrNode,
    OutMap: BoundaryMap<HostNode, Node>,
{
    // All expected output ports in subgraph
    let subgraph_nodes: BTreeSet<_> = subgraph.nodes().iter().copied().collect();
    // The incoming ports at the output boundary of the subgraph are obtained
    // by taking the outgoing ports of the subgraph boundary, finding the linked
    // incoming ports and checking that they are not in the subgraph.
    let exp_ports_subgraph: BTreeSet<_> = subgraph
        .outgoing_ports()
        .iter()
        .flat_map(|&(node, port)| host.linked_inputs(node, port))
        .filter(|&(node, _)| !subgraph_nodes.contains(&node))
        .collect();

    // Check that ports in nu_out's domain match one-to-one to ports linked to
    // output ports of `subgraph`
    match check_one_to_one(nu_out.all_keys(host), exp_ports_subgraph) {
        Ok(()) => (),
        Err(Either::Left((node, port))) => {
            return Err(InvalidBoundaryError::InvalidOutputPort(
                WhichNode::Host(node),
                port,
            ));
        }
        Err(Either::Right(missing)) => {
            return Err(InvalidBoundaryError::MissingOutputPorts(
                missing
                    .into_iter()
                    .map(|(node, port)| (WhichNode::Host(node), port))
                    .collect(),
            ));
        }
    }

    // All expected input ports in replacement
    let exp_ports_repl = {
        let [_, repl_out] = replacement
            .get_io(replacement.root())
            .expect("replacement is a DFG");
        replacement
            .node_inputs(repl_out)
            .map(|port| (repl_out, port))
            .collect()
    };

    // Check that ports in nu_out's image are contained within the incoming ports
    // of the output node of `replacement` (and linear ports map one-to-one)
    match check_contained(
        nu_out.all_values(host, &replacement),
        exp_ports_repl,
        |(node, port)| {
            if let Some(kind) = replacement.get_optype(node).port_kind(port) {
                kind.is_linear()
            } else {
                false
            }
        },
    ) {
        Ok(()) => (),
        Err(Either::Left((node, port))) => {
            return Err(InvalidBoundaryError::InvalidOutputPort(
                WhichNode::Replacement(node),
                port,
            ));
        }
        Err(Either::Right(missing)) => {
            return Err(InvalidBoundaryError::MissingOutputPorts(
                missing
                    .into_iter()
                    .map(|(node, port)| (WhichNode::Replacement(node), port))
                    .collect(),
            ));
        }
    }

    Ok(())
}

fn check_one_to_one<V: Ord>(
    it: impl IntoIterator<Item = V>,
    mut valid_values: BTreeSet<V>,
) -> Result<(), Either<V, BTreeSet<V>>> {
    // 1. Check that all values in it are in valid_values (and appear only once)
    for val in it {
        if !valid_values.remove(&val) {
            return Err(Either::Left(val));
        }
    }

    // 2. Check that all values in valid_values were in it
    if !valid_values.is_empty() {
        return Err(Either::Right(valid_values));
    }

    Ok(())
}

/// Check that all values in `it2` are in `it1` and if they should be unique,
/// that they only appear once in `it2`.
fn check_contained<V: Ord + Copy>(
    it: impl IntoIterator<Item = V>,
    mut valid_values: BTreeSet<V>,
    is_unique: impl Fn(V) -> bool,
) -> Result<(), Either<V, BTreeSet<V>>> {
    // 1. Check that all values in it are in valid_values (and if `is_unique`,
    // appear only once)
    for val in it {
        if is_unique(val) {
            if !valid_values.remove(&val) {
                return Err(Either::Left(val));
            }
        } else if !valid_values.contains(&val) {
            return Err(Either::Left(val));
        }
    }

    // 2. Check that all unique values in valid_values were in it
    valid_values.retain(|&val| is_unique(val));
    if !valid_values.is_empty() {
        return Err(Either::Right(valid_values));
    }

    Ok(())
}

impl<HostNode, InMap, OutMap> PatchVerification for SimpleReplacement<HostNode, InMap, OutMap>
where
    InMap: BoundaryMap<Node, HostNode>,
    OutMap: BoundaryMap<HostNode, Node>,
    HostNode: HugrNode,
{
    type Error = SimpleReplacementError;
    type Node = HostNode;

    fn verify(&self, h: &impl HugrView<Node = HostNode>) -> Result<(), SimpleReplacementError> {
        self.is_valid_rewrite(h)
    }

    #[inline]
    fn invalidation_set(&self) -> impl Iterator<Item = HostNode> {
        let subcirc = self.subgraph.nodes().iter().copied();
        subcirc.chain(self.nu_out.required_nodes()).unique()
    }
}

/// Result of applying a [`SimpleReplacement`].
pub struct Outcome<HostNode = Node> {
    /// Map from Node in replacement to corresponding Node in the result Hugr
    pub node_map: HashMap<Node, HostNode>,
    /// Nodes removed from the result Hugr and their weights
    pub removed_nodes: HashMap<HostNode, OpType>,
}

impl<N, InMap, OutMap> PatchHugrMut for SimpleReplacement<N, InMap, OutMap>
where
    N: HugrNode,
    InMap: BoundaryMap<Node, N>,
    OutMap: BoundaryMap<N, Node>,
{
    type Outcome = Outcome<N>;
    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<Self::Outcome, Self::Error> {
        self.is_valid_rewrite(h)?;

        let parent = self.subgraph.get_parent(h);

        // We proceed to connect the edges between the newly inserted
        // replacement and the rest of the graph.
        //
        // Existing connections to the removed subgraph will be automatically
        // removed when the nodes are removed.

        // 1. Get the boundary edges
        let boundary_edges = self.all_boundary_edges(h).collect_vec();

        let Self {
            replacement,
            subgraph,
            ..
        } = self;

        // 2. Insert the replacement as a whole.
        let InsertionResult { new_root, node_map } = h.insert_hugr(parent, replacement);

        // remove the Input and Output nodes from the replacement graph
        let replace_children = h.children(new_root).collect::<Vec<N>>();
        for &io in &replace_children[..2] {
            h.remove_node(io);
        }
        // make all replacement top level children children of the parent
        for &child in &replace_children[2..] {
            h.set_parent(child, parent);
        }
        // remove the replacement root (which now has no children and no edges)
        h.remove_node(new_root);

        // 3. Insert all boundary edges.
        for (src, tgt) in boundary_edges {
            let (src_node, src_port) = src.map_replacement(&node_map);
            let (tgt_node, tgt_port) = tgt.map_replacement(&node_map);
            h.connect(src_node, src_port, tgt_node, tgt_port);
        }

        // 4. Remove all nodes in subgraph and edges between them.
        let removed_nodes = subgraph
            .nodes()
            .iter()
            .map(|&node| (node, h.remove_node(node)))
            .collect();

        Ok(Outcome {
            node_map,
            removed_nodes,
        })
    }
}

/// Error from a [`SimpleReplacement`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimpleReplacementError {
    /// Invalid parent node.
    #[error("Parent node is invalid.")]
    InvalidParentNode(),
    /// Node requested for removal is invalid.
    #[error("A node requested for removal is invalid.")]
    InvalidRemovedNode(),
    /// Node in replacement graph is invalid.
    #[error("A node in the replacement graph is invalid.")]
    InvalidReplacementNode(),
    /// Inlining replacement failed.
    #[error("Inlining replacement failed: {0}")]
    InliningFailed(#[from] InlineDFGError),
}

/// Invalid boundary for a [`SimpleReplacement`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidBoundaryError<HostNode> {
    /// Invalid input port.
    #[error("Invalid port in input boundary map: ({0:?}, {1:?})")]
    InvalidInputPort(WhichNode<HostNode>, IncomingPort),
    /// Missing input ports.
    #[error("Missing ports in input boundary map: {0:?}")]
    MissingInputPorts(BTreeSet<(WhichNode<HostNode>, IncomingPort)>),

    /// Invalid output port.
    #[error("Invalid port in output boundary map: ({0:?}, {1:?})")]
    InvalidOutputPort(WhichNode<HostNode>, IncomingPort),
    /// Missing output ports.
    #[error("Missing ports in output boundary map: {0:?}")]
    MissingOutputPorts(BTreeSet<(WhichNode<HostNode>, IncomingPort)>),
}

/// A node in the host or replacement graph.
#[derive(Debug, Clone, Error, PartialEq, Eq, PartialOrd, Ord)]
pub enum WhichNode<HostNode> {
    /// A node in the replacement graph.
    Replacement(Node),
    /// A node in the host graph.
    Host(HostNode),
}

#[cfg(test)]
pub(in crate::hugr::patch) mod test {
    use itertools::Itertools;

    use rstest::{fixture, rstest};

    use std::collections::{HashMap, HashSet};

    use crate::builder::test::n_identity;
    use crate::builder::{
        endo_sig, inout_sig, BuildError, Container, DFGBuilder, Dataflow, DataflowHugr,
        DataflowSubContainer, HugrBuilder, ModuleBuilder,
    };
    use crate::extension::prelude::{bool_t, qb_t};

    use crate::hugr::patch::{BoundaryMap, OutputNodeBoundaryMap, PatchVerification};
    use crate::hugr::views::{HugrView, SiblingSubgraph};
    use crate::hugr::{Hugr, HugrMut, Patch};
    use crate::ops::dataflow::DataflowOpTrait;
    use crate::ops::handle::NodeHandle;
    use crate::ops::OpTag;
    use crate::ops::OpTrait;
    use crate::std_extensions::logic::test::and_op;
    use crate::std_extensions::logic::LogicOp;
    use crate::types::{Signature, Type};
    use crate::utils::test_quantum_extension::{cx_gate, h_gate};
    use crate::{IncomingPort, Node, OutgoingPort};

    use super::SimpleReplacement;

    /// Creates a hugr like the following:
    /// --   H   --
    /// -- [DFG] --
    /// where [DFG] is:
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    fn make_hugr() -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let _f_id = {
            let mut func_builder = module_builder
                .define_function("main", Signature::new_endo(vec![qb_t(), qb_t(), qb_t()]))?;

            let [qb0, qb1, qb2] = func_builder.input_wires_arr();

            let q_out = func_builder.add_dataflow_op(h_gate(), vec![qb2])?;

            let mut inner_builder =
                func_builder.dfg_builder_endo([(qb_t(), qb0), (qb_t(), qb1)])?;
            let inner_graph = {
                let [wire0, wire1] = inner_builder.input_wires_arr();
                let wire2 = inner_builder.add_dataflow_op(h_gate(), vec![wire0])?;
                let wire3 = inner_builder.add_dataflow_op(h_gate(), vec![wire1])?;
                let wire45 = inner_builder
                    .add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
                let [wire4, wire5] = wire45.outputs_arr();
                let wire6 = inner_builder.add_dataflow_op(h_gate(), vec![wire4])?;
                let wire7 = inner_builder.add_dataflow_op(h_gate(), vec![wire5])?;
                inner_builder.finish_with_outputs(wire6.outputs().chain(wire7.outputs()))
            }?;

            func_builder.finish_with_outputs(inner_graph.outputs().chain(q_out.outputs()))?
        };
        Ok(module_builder.finish_hugr()?)
    }

    #[fixture]
    pub(in crate::hugr::patch) fn simple_hugr() -> Hugr {
        make_hugr().unwrap()
    }
    /// Creates a hugr with a DFG root like the following:
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn make_dfg_hugr() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;
        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire0])?;
        let wire3 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire45 =
            dfg_builder.add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
        dfg_builder.finish_hugr_with_outputs(wire45.outputs())
    }

    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr() -> Hugr {
        make_dfg_hugr().unwrap()
    }

    /// Creates a hugr with a DFG root like the following:
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn make_dfg_hugr2() -> Result<Hugr, BuildError> {
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;

        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
        let wire2out = wire2.outputs().exactly_one().unwrap();
        let wireoutvec = vec![wire0, wire2out];
        dfg_builder.finish_hugr_with_outputs(wireoutvec)
    }

    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr2() -> Hugr {
        make_dfg_hugr2().unwrap()
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (1) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (0) NOT ├───┤
    ///  └─────────┘   │    ┌─────────┐
    ///                └────┤ (2) NOT ├──
    ///                     └─────────┘
    /// This can be replaced with an empty hugr coping the input to both
    /// outputs.
    ///
    /// Returns the hugr and the nodes of the NOT gates, in order.
    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr_copy_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let not_1 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b1] = not_1.outputs_arr();

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node(), not_1.node()],
        )
    }

    /// A hugr with a DFG root mapping bool_t() to (bool_t(), bool_t())
    ///                     ┌─────────┐
    ///                ┌────┤ (1) NOT ├──
    ///  ┌─────────┐   │    └─────────┘
    /// ─┤ (0) NOT ├───┤
    ///  └─────────┘   │
    ///                └─────────────────
    ///
    /// This can be replaced with a single NOT op, coping the input to the first
    /// output.
    ///
    /// Returns the hugr and the nodes of the NOT ops, in order.
    #[fixture]
    pub(in crate::hugr::patch) fn dfg_hugr_half_not_bools() -> (Hugr, Vec<Node>) {
        let mut dfg_builder =
            DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
        let [b] = dfg_builder.input_wires_arr();

        let not_inp = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b] = not_inp.outputs_arr();

        let not_0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b]).unwrap();
        let [b0] = not_0.outputs_arr();
        let b1 = b;

        (
            dfg_builder.finish_hugr_with_outputs([b0, b1]).unwrap(),
            vec![not_inp.node(), not_0.node()],
        )
    }

    #[rstest]
    /// Replace the
    ///      ┌───┐
    /// ──■──┤ H ├
    /// ┌─┴─┐├───┤
    /// ┤ X ├┤ H ├
    /// └───┘└───┘
    /// part of
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    /// with
    /// ┌───┐
    /// ┤ H ├──■──
    /// ├───┤┌─┴─┐
    /// ┤ H ├┤ X ├
    /// └───┘└───┘
    fn test_simple_replacement(
        simple_hugr: Hugr,
        dfg_hugr: Hugr,
        #[values(apply_simple, apply_replace)] applicator: impl Fn(&mut Hugr, SimpleReplacement),
    ) {
        let mut h: Hugr = simple_hugr;
        // 1. Locate the CX and its successor H's in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let s: Vec<Node> = vec![h_node_cx, h_node_h0, h_node_h1].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr;
        // 3. Construct the input and output matchings
        // 3.1. Locate the CX and its predecessor H's in n
        let n_node_cx = n
            .nodes()
            .find(|node: &Node| *n.get_optype(*node) == cx_gate().into())
            .unwrap();
        let (n_node_h0, n_node_h1) = n.input_neighbours(n_node_cx).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let n_port_0 = n.node_inputs(n_node_h0).next().unwrap();
        let n_port_1 = n.node_inputs(n_node_h1).next().unwrap();
        let (n_cx_out_0, n_cx_out_1) = n.node_outputs(n_node_cx).take(2).collect_tuple().unwrap();
        let n_port_2 = n.linked_inputs(n_node_cx, n_cx_out_0).next().unwrap().1;
        let n_port_3 = n.linked_inputs(n_node_cx, n_cx_out_1).next().unwrap().1;
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let h_h0_out = h.node_outputs(h_node_h0).next().unwrap();
        let h_h1_out = h.node_outputs(h_node_h1).next().unwrap();
        let (h_outp_node, h_port_2) = h.linked_inputs(h_node_h0, h_h0_out).next().unwrap();
        let h_port_3 = h.linked_inputs(h_node_h1, h_h1_out).next().unwrap().1;
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_h0, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h1, n_port_1), (h_node_cx, h_port_1));
        nu_out.insert((h_outp_node, h_port_2), n_port_2);
        nu_out.insert((h_outp_node, h_port_3), n_port_3);
        // 4. Define the replacement
        let r = SimpleReplacement::try_new(
            SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            &h,
            n,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();
        assert_eq!(
            HashSet::<_>::from_iter(r.invalidation_set()),
            HashSet::<_>::from_iter([h_node_cx, h_node_h0, h_node_h1, h_outp_node]),
        );

        applicator(&mut h, r);
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├──■──
        // ├───┤├───┤┌─┴─┐
        // ┤ H ├┤ H ├┤ X ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(), Ok(()));
    }

    #[rstest]
    /// Replace the
    ///
    /// ──■──
    /// ┌─┴─┐
    /// ┤ X ├
    /// └───┘
    /// part of
    /// ┌───┐     ┌───┐
    /// ┤ H ├──■──┤ H ├
    /// ├───┤┌─┴─┐├───┤
    /// ┤ H ├┤ X ├┤ H ├
    /// └───┘└───┘└───┘
    /// with
    /// ─────
    /// ┌───┐
    /// ┤ H ├
    /// └───┘
    fn test_simple_replacement_with_empty_wires(simple_hugr: Hugr, dfg_hugr2: Hugr) {
        let mut h: Hugr = simple_hugr;

        // 1. Locate the CX in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let s: Vec<Node> = vec![h_node_cx].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr2;
        // 3. Construct the input and output matchings
        // 3.1. Locate the Output and its predecessor H in n
        let n_node_output = n
            .nodes()
            .find(|node: &Node| n.get_optype(*node).tag() == OpTag::Output)
            .unwrap();
        let (_n_node_input, n_node_h) = n.input_neighbours(n_node_output).collect_tuple().unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let (n_port_0, n_port_1) = n
            .node_inputs(n_node_output)
            .take(2)
            .collect_tuple()
            .unwrap();
        let n_port_2 = n.node_inputs(n_node_h).next().unwrap();
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        let (h_node_h0, h_node_h1) = h.output_neighbours(h_node_cx).collect_tuple().unwrap();
        let h_port_2 = h.node_inputs(h_node_h0).next().unwrap();
        let h_port_3 = h.node_inputs(h_node_h1).next().unwrap();
        // 3.4. Construct the maps
        let mut nu_inp: HashMap<(Node, IncomingPort), (Node, IncomingPort)> = HashMap::new();
        let mut nu_out: HashMap<(Node, IncomingPort), IncomingPort> = HashMap::new();
        nu_inp.insert((n_node_output, n_port_0), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_h, n_port_2), (h_node_cx, h_port_1));
        nu_out.insert((h_node_h0, h_port_2), n_port_0);
        nu_out.insert((h_node_h1, h_port_3), n_port_1);
        // 4. Define the replacement
        let r = SimpleReplacement::try_new(
            SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            &h,
            n,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();
        h.apply_patch(r).unwrap();
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├
        // ├───┤├───┤┌───┐
        // ┤ H ├┤ H ├┤ H ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(), Ok(()));
    }

    #[test]
    fn test_replace_cx_cross() {
        let q_row: Vec<Type> = vec![qb_t(), qb_t()];
        let mut builder = DFGBuilder::new(endo_sig(q_row)).unwrap();
        let mut circ = builder.as_circuit(builder.input_wires());
        circ.append(cx_gate(), [0, 1]).unwrap();
        circ.append(cx_gate(), [1, 0]).unwrap();
        let wires = circ.finish();
        let [input, output] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(wires).unwrap();
        let replacement = h.clone();
        let orig = h.clone();

        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();
        let inputs: HashMap<_, _> = h
            .node_outputs(input)
            .filter(|&p| {
                h.get_optype(input)
                    .as_input()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| {
                let link = h.linked_inputs(input, p).next().unwrap();
                (link, link)
            })
            .collect();
        let outputs: HashMap<_, _> = h
            .node_inputs(output)
            .filter(|&p| {
                h.get_optype(output)
                    .as_output()
                    .unwrap()
                    .signature()
                    .port_type(p)
                    .is_some()
            })
            .map(|p| ((output, p), p))
            .collect();
        h.apply_patch(
            SimpleReplacement::try_new(
                SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
                &h,
                replacement,
                inputs,
                OutputNodeBoundaryMap::from(outputs),
            )
            .unwrap(),
        )
        .unwrap();

        // They should be the same, up to node indices
        assert_eq!(h.num_edges(), orig.num_edges());
    }

    #[test]
    fn test_replace_after_copy() {
        let one_bit = vec![bool_t()];
        let two_bit = vec![bool_t(), bool_t()];

        let mut builder = DFGBuilder::new(endo_sig(one_bit.clone())).unwrap();
        let inw = builder.input_wires().exactly_one().unwrap();
        let outw = builder
            .add_dataflow_op(and_op(), [inw, inw])
            .unwrap()
            .outputs();
        let [input, _] = builder.io();
        let mut h = builder.finish_hugr_with_outputs(outw).unwrap();

        let mut builder = DFGBuilder::new(inout_sig(two_bit, one_bit)).unwrap();
        let inw = builder.input_wires();
        let outw = builder.add_dataflow_op(and_op(), inw).unwrap().outputs();
        let [repl_input, repl_output] = builder.io();
        let repl = builder.finish_hugr_with_outputs(outw).unwrap();

        let orig = h.clone();

        let removal = h
            .nodes()
            .filter(|&n| h.get_optype(n).tag() == OpTag::Leaf)
            .collect_vec();

        let first_out_p = h.node_outputs(input).next().unwrap();
        let embedded_inputs = h.linked_inputs(input, first_out_p);
        let repl_inputs = repl
            .node_outputs(repl_input)
            .map(|p| repl.linked_inputs(repl_input, p).next().unwrap());
        let inputs: HashMap<_, _> = embedded_inputs.zip(repl_inputs).collect();

        let outputs: HashMap<_, _> = repl
            .node_inputs(repl_output)
            .filter(|&p| repl.signature(repl_output).unwrap().port_type(p).is_some())
            .map(|p| ((repl_output, p), p))
            .collect();

        h.apply_patch(
            SimpleReplacement::try_new(
                SiblingSubgraph::try_from_nodes(removal, &h).unwrap(),
                &h,
                repl,
                inputs,
                OutputNodeBoundaryMap::from(outputs),
            )
            .unwrap(),
        )
        .unwrap();

        // Nothing changed
        assert_eq!(h.num_nodes(), orig.num_nodes());
    }

    /// Remove all the NOT gates in [`dfg_hugr_copy_bools`] by connecting the
    /// input directly to the outputs.
    ///
    /// https://github.com/CQCL/hugr/issues/1190
    #[rstest]
    fn test_copy_inputs(dfg_hugr_copy_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_copy_bools;
        let (input_not, output_not_0, output_not_1) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let replacement = {
            let b =
                DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            b.finish_hugr_with_outputs([w, w]).unwrap()
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0, output_not_1], &hugr)
                .unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to
        // (target ports of edges from nodes not in `removal` to nodes in
        // `removal`).
        let nu_inp: HashMap<_, _> = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_output, IncomingPort::from(1)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in
        // `removal`) to (input ports of the Output node of `replacement`).
        let nu_out: HashMap<_, _> = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let rewrite = SimpleReplacement::try_new(
            subgraph,
            &hugr,
            replacement,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.num_nodes(), 3);
    }

    /// Remove one of the NOT ops in [`dfg_hugr_half_not_bools`] by connecting
    /// the input directly to the output.
    ///
    /// https://github.com/CQCL/hugr/issues/1323
    #[rstest]
    fn test_half_nots(dfg_hugr_half_not_bools: (Hugr, Vec<Node>)) {
        let (mut hugr, nodes) = dfg_hugr_half_not_bools;
        let (input_not, output_not_0) = nodes.into_iter().collect_tuple().unwrap();

        let [_input, output] = hugr.get_io(hugr.root()).unwrap();

        let (replacement, repl_not) = {
            let mut b =
                DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t(), bool_t()])).unwrap();
            let [w] = b.input_wires_arr();
            let not = b.add_dataflow_op(LogicOp::Not, vec![w]).unwrap();
            let [w_not] = not.outputs_arr();
            (b.finish_hugr_with_outputs([w, w_not]).unwrap(), not.node())
        };
        let [_repl_input, repl_output] = replacement.get_io(replacement.root()).unwrap();

        let subgraph =
            SiblingSubgraph::try_from_nodes(vec![input_not, output_not_0], &hugr).unwrap();
        // A map from (target ports of edges from the Input node of `replacement`) to
        // (target ports of edges from nodes not in `removal` to nodes in
        // `removal`).
        let nu_inp: HashMap<_, _> = [
            (
                (repl_output, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
            (
                (repl_not, IncomingPort::from(0)),
                (input_not, IncomingPort::from(0)),
            ),
        ]
        .into_iter()
        .collect();
        // A map from (target ports of edges from nodes in `removal` to nodes not in
        // `removal`) to (input ports of the Output node of `replacement`).
        let nu_out: HashMap<_, _> = [
            ((output, IncomingPort::from(0)), IncomingPort::from(0)),
            ((output, IncomingPort::from(1)), IncomingPort::from(1)),
        ]
        .into_iter()
        .collect();

        let rewrite = SimpleReplacement::try_new(
            subgraph,
            &hugr,
            replacement,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();
        rewrite.apply(&mut hugr).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(hugr.validate(), Ok(()));
        assert_eq!(hugr.num_nodes(), 4);
    }

    #[rstest]
    fn test_nested_replace(dfg_hugr2: Hugr) {
        // replace a node with a hugr with children

        let mut h = dfg_hugr2;
        let h_node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == h_gate().into())
            .unwrap();

        // build a nested identity dfg
        let mut nest_build = DFGBuilder::new(Signature::new_endo(qb_t())).unwrap();
        let [input] = nest_build.input_wires_arr();
        let inner_build = nest_build.dfg_builder_endo([(qb_t(), input)]).unwrap();
        let inner_dfg = n_identity(inner_build).unwrap();
        let inner_dfg_node = inner_dfg.node();
        let replacement = nest_build
            .finish_hugr_with_outputs([inner_dfg.out_wire(0)])
            .unwrap();
        let subgraph = SiblingSubgraph::try_from_nodes(vec![h_node], &h).unwrap();
        let nu_inp: HashMap<_, _> = vec![(
            (inner_dfg_node, IncomingPort::from(0)),
            (h_node, IncomingPort::from(0)),
        )]
        .into_iter()
        .collect();

        let nu_out = HashMap::from_iter([(
            (h.get_io(h.root()).unwrap()[1], IncomingPort::from(1)),
            IncomingPort::from(0),
        )]);

        let rewrite = SimpleReplacement::try_new(
            subgraph,
            &h,
            replacement,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();

        assert_eq!(h.num_nodes(), 4);

        rewrite.apply(&mut h).unwrap_or_else(|e| panic!("{e}"));
        h.validate().unwrap_or_else(|e| panic!("{e}"));

        assert_eq!(h.num_nodes(), 6);
    }

    #[rstest]
    fn test_simple_replacement_with_empty_wires_using_outgoing_ports(
        simple_hugr: Hugr,
        dfg_hugr2: Hugr,
    ) {
        let mut h: Hugr = simple_hugr;

        // 1. Locate the CX in h
        let h_node_cx: Node = h
            .nodes()
            .find(|node: &Node| *h.get_optype(*node) == cx_gate().into())
            .unwrap();
        let s: Vec<Node> = vec![h_node_cx].into_iter().collect();
        // 2. Construct a new DFG-rooted hugr for the replacement
        let n: Hugr = dfg_hugr2;
        // 3. Construct the input and output matchings
        // 3.1. Locate the Output and its predecessor H in n
        let [n_node_input, n_node_output] = n.get_io(n.root()).unwrap();
        // 3.2. Locate the ports we need to specify as "glue" in n
        let (n_port_0, n_port_1) = n
            .node_inputs(n_node_output)
            .take(2)
            .collect_tuple()
            .unwrap();
        // 3.3. Locate the ports we need to specify as "glue" in h
        let (h_port_0, h_port_1) = h.node_inputs(h_node_cx).take(2).collect_tuple().unwrap();
        // 3.4. Construct the maps
        let mut nu_inp = HashMap::new();
        let mut nu_out = HashMap::new();
        nu_inp.insert((n_node_input, OutgoingPort::from(0)), (h_node_cx, h_port_0));
        nu_inp.insert((n_node_input, OutgoingPort::from(1)), (h_node_cx, h_port_1));
        nu_out.insert((h_node_cx, OutgoingPort::from(0)), n_port_0);
        nu_out.insert((h_node_cx, OutgoingPort::from(1)), n_port_1);
        // 4. Define the replacement
        let r = SimpleReplacement::try_new(
            SiblingSubgraph::try_from_nodes(s, &h).unwrap(),
            &h,
            n,
            nu_inp,
            OutputNodeBoundaryMap::from(nu_out),
        )
        .unwrap();
        h.apply_patch(r).unwrap();
        // Expect [DFG] to be replaced with:
        // ┌───┐┌───┐
        // ┤ H ├┤ H ├
        // ├───┤├───┤┌───┐
        // ┤ H ├┤ H ├┤ H ├
        // └───┘└───┘└───┘
        assert_eq!(h.validate(), Ok(()));
    }

    use crate::hugr::patch::replace::Replacement;
    fn to_replace(h: &impl HugrView<Node = Node>, s: SimpleReplacement) -> Replacement {
        use crate::hugr::patch::replace::{NewEdgeKind, NewEdgeSpec};

        let mut replacement = s.replacement;
        let (in_, out) = replacement
            .children(replacement.root())
            .take(2)
            .collect_tuple()
            .unwrap();
        let mu_inp = s
            .nu_inp
            .iter()
            .map(|((tgt, tgt_port), (r_n, r_p))| {
                if *tgt == out {
                    unimplemented!()
                };
                let (src, src_port) = h.single_linked_output(*r_n, *r_p).unwrap();
                NewEdgeSpec {
                    src,
                    tgt: *tgt,
                    kind: NewEdgeKind::Value {
                        src_pos: src_port,
                        tgt_pos: *tgt_port,
                    },
                }
            })
            .collect();
        let mu_out = s
            .nu_out
            .iter(h, &replacement)
            .map(|((tgt, tgt_port), (out, out_port))| {
                let (src, src_port) = replacement.single_linked_output(out, out_port).unwrap();
                if src == in_ {
                    unimplemented!()
                };
                NewEdgeSpec {
                    src,
                    tgt,
                    kind: NewEdgeKind::Value {
                        src_pos: src_port,
                        tgt_pos: tgt_port,
                    },
                }
            })
            .collect();
        replacement.remove_node(in_);
        replacement.remove_node(out);
        Replacement {
            removal: s.subgraph.nodes().to_vec(),
            replacement,
            adoptions: HashMap::new(),
            mu_inp,
            mu_out,
            mu_new: vec![],
        }
    }

    fn apply_simple(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_patch(rw).unwrap();
    }

    fn apply_replace(h: &mut Hugr, rw: SimpleReplacement) {
        h.apply_patch(to_replace(h, rw)).unwrap();
    }
}
