//! Iterators used by the implementation of HugrView for Hugr.

use std::iter::{Enumerate, FusedIterator};
use std::ops::Range;

use portgraph::portgraph::NodePorts;
use portgraph::{NodeIndex, PortIndex};

use super::{MultiPortGraph, SubportIndex};

/// Iterator over the nodes of a graph.
#[derive(Clone)]
pub struct Nodes<'a> {
    // We use portgraph's iterator, but filter out the copy nodes.
    pub(super) multigraph: &'a MultiPortGraph,
    pub(super) iter: portgraph::portgraph::Nodes<'a>,
    pub(super) len: usize,
}

impl<'a> Iterator for Nodes<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let next = self
            .iter
            .find(|node| !self.multigraph.is_copy_node(*node))?;
        self.len -= 1;
        Some(next)
    }

    #[inline]
    fn count(self) -> usize {
        self.len
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a> ExactSizeIterator for Nodes<'a> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a> DoubleEndedIterator for Nodes<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let node = self.iter.next_back()?;
            if !self.multigraph.is_copy_node(node) {
                self.len -= 1;
                return Some(node);
            }
        }
    }
}

impl<'a> FusedIterator for Nodes<'a> {}

/// Iterator over the ports of a node.
#[derive(Clone)]
pub struct NodeSubports<'a> {
    multigraph: &'a MultiPortGraph,
    ports: portgraph::portgraph::NodePorts,
    current_port: Option<PortIndex>,
    current_subports: Range<usize>,
}

impl<'a> NodeSubports<'a> {
    pub(super) fn new(
        multigraph: &'a MultiPortGraph,
        ports: portgraph::portgraph::NodePorts,
    ) -> Self {
        Self {
            multigraph,
            ports,
            current_port: None,
            current_subports: 0..0,
        }
    }
}

impl<'a> Iterator for NodeSubports<'a> {
    type Item = SubportIndex;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(offset) = self.current_subports.next() {
                // We are in the middle of iterating over the subports of a port.
                let current_port = self
                    .current_port
                    .expect("NodeSubports set an invalid current_port value.");
                return Some(SubportIndex::new_multi(current_port, offset));
            }

            // Proceed to the next port.
            let port = self.ports.next()?;
            self.current_port = Some(port);
            if self.multigraph.is_multiport(port) {
                let dir = self.multigraph.graph.port_direction(port).unwrap();
                let copy_node = self
                    .multigraph
                    .get_copy_node(port)
                    .expect("A port was marked as multiport, but no copy node was found.");
                self.current_subports = self
                    .multigraph
                    .graph
                    .port_offsets(copy_node, dir)
                    .as_range(dir);
            } else {
                // The port is not a multiport, return the single subport.
                return Some(SubportIndex::new_unique(port));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.ports.len() + self.current_subports.len(), None)
    }
}

impl<'a> FusedIterator for NodeSubports<'a> {}

/// Iterator over the ports of a node.
#[derive(Clone)]
pub struct Neighbours<'a> {
    multigraph: &'a MultiPortGraph,
    subports: NodeSubports<'a>,
    current_copy_node: Option<portgraph::NodeIndex>,
}

impl<'a> Neighbours<'a> {
    pub(super) fn new(multigraph: &'a MultiPortGraph, subports: NodeSubports<'a>) -> Self {
        Self {
            multigraph,
            subports,
            current_copy_node: None,
        }
    }
}

impl<'a> Iterator for Neighbours<'a> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let link = self.subports.find_map(|subport| {
            let port_index = subport.port();
            if !self.multigraph.is_multiport(port_index) {
                self.multigraph.graph.port_link(port_index)
            } else {
                // There is a copy node
                if subport.offset() == 0 {
                    self.current_copy_node = self.multigraph.get_copy_node(port_index);
                }
                let copy_node = self
                    .current_copy_node
                    .expect("Copy node not connected to a multiport.");
                let dir = self.multigraph.graph.port_direction(port_index).unwrap();
                let offset = portgraph::PortOffset::new(dir, subport.offset());
                let subport_index = self.multigraph.graph.port_index(copy_node, offset).unwrap();
                self.multigraph.graph.port_link(subport_index)
            }
        })?;
        let link_subport = self.multigraph.get_subport_from_index(link).unwrap();
        self.multigraph.graph.port_node(link_subport.port())
    }
}

impl<'a> FusedIterator for Neighbours<'a> {}

/// Iterator over the links from a node, created by
/// [`MultiPortGraph::node_links`].
///
/// In contrast to [`portgraph::portgraph::NodeLinks`], this iterator
/// only returns linked subports, and includes the source subport.
#[derive(Clone)]
#[allow(dead_code)]
pub struct NodeLinks<'a> {
    multigraph: &'a MultiPortGraph,
    ports: NodePorts,
    current_links: Option<PortLinks<'a>>,
}

impl<'a> NodeLinks<'a> {
    pub(super) fn new(multigraph: &'a MultiPortGraph, ports: NodePorts) -> Self {
        Self {
            multigraph,
            ports,
            current_links: None,
        }
    }
}

impl<'a> Iterator for NodeLinks<'a> {
    /// A link from one of the node's subports to another subport.
    type Item = (SubportIndex, SubportIndex);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(links) = &mut self.current_links {
                if let Some(link) = links.next() {
                    return Some(link);
                }
                self.current_links = None;
            }
            let port = self.ports.next()?;
            self.current_links = Some(PortLinks::new(self.multigraph, port));
        }
    }
}

impl<'a> FusedIterator for NodeLinks<'a> {}

/// Iterator over the links between two nodes, created by
/// [`MultiPortGraph::get_connections`].
#[derive(Clone)]
#[allow(dead_code)]
pub struct NodeConnections<'a> {
    multigraph: &'a MultiPortGraph,
    target: NodeIndex,
    links: NodeLinks<'a>,
}

impl<'a> NodeConnections<'a> {
    pub(super) fn new(
        multigraph: &'a MultiPortGraph,
        target: NodeIndex,
        links: NodeLinks<'a>,
    ) -> Self {
        Self {
            multigraph,
            target,
            links,
        }
    }
}

impl<'a> Iterator for NodeConnections<'a> {
    /// A link from one of the node's subports to another subport.
    type Item = (SubportIndex, SubportIndex);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (source, target) = self.links.next()?;
            let target_node = self.multigraph.graph.port_node(target.port())?;
            if target_node == self.target {
                return Some((source, target));
            }
        }
    }
}

impl<'a> FusedIterator for NodeConnections<'a> {}

/// Iterator over the links of a port
#[derive(Clone)]
#[allow(missing_docs)]
pub enum PortLinks<'a> {
    /// The port is not a multiport. The iterator returns exactly one link.
    SinglePort {
        multigraph: &'a MultiPortGraph,
        port: PortIndex,
        empty: bool,
    },
    /// The port is a multiport. The iterator may return any number of links.
    Multiport {
        multigraph: &'a MultiPortGraph,
        port: PortIndex,
        subports: Enumerate<portgraph::portgraph::NodePorts>,
    },
}

impl<'a> PortLinks<'a> {
    pub(super) fn new(multigraph: &'a MultiPortGraph, port: PortIndex) -> Self {
        if multigraph.is_multiport(port) {
            let copy_node = multigraph.get_copy_node(port).unwrap();
            let dir = multigraph.graph.port_direction(port).unwrap();
            let subports = multigraph.graph.ports(copy_node, dir).enumerate();
            Self::Multiport {
                multigraph,
                port,
                subports,
            }
        } else {
            Self::SinglePort {
                multigraph,
                port,
                empty: false,
            }
        }
    }
}

/// Returns the link of a single port for a `PortLinks` iterator.
#[inline(always)]
fn port_links_single(
    multigraph: &MultiPortGraph,
    port: PortIndex,
) -> Option<(SubportIndex, SubportIndex)> {
    let link = multigraph.graph.port_link(port)?;
    let link = multigraph.get_subport_from_index(link)?;
    Some((SubportIndex::new_unique(port), link))
}

/// Try to get the next link of a multiport for a `PortLinks` iterator.
#[inline(always)]
fn port_links_multiport(
    multigraph: &MultiPortGraph,
    port: PortIndex,
    subport_offset: usize,
    copy_port_index: PortIndex,
) -> Option<(SubportIndex, SubportIndex)> {
    let link = multigraph.graph.port_link(copy_port_index)?;
    let link = multigraph.get_subport_from_index(link)?;
    Some((SubportIndex::new_multi(port, subport_offset), link))
}

impl<'a> Iterator for PortLinks<'a> {
    type Item = (SubportIndex, SubportIndex);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PortLinks::SinglePort {
                multigraph,
                port,
                empty,
            } if !*empty => {
                *empty = true;
                port_links_single(multigraph, *port)
            }
            PortLinks::SinglePort { .. } => None,
            PortLinks::Multiport {
                multigraph,
                port,
                subports,
                ..
            } => subports
                .find_map(|(offset, index)| port_links_multiport(multigraph, *port, offset, index)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            PortLinks::SinglePort { empty, .. } => {
                if *empty {
                    (0, Some(0))
                } else {
                    (1, Some(1))
                }
            }
            PortLinks::Multiport { subports, .. } => (0, Some(subports.len())),
        }
    }
}

impl<'a> DoubleEndedIterator for PortLinks<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            PortLinks::SinglePort {
                multigraph,
                port,
                empty,
            } if !*empty => {
                *empty = true;
                port_links_single(multigraph, *port)
            }
            PortLinks::SinglePort { .. } => None,
            PortLinks::Multiport {
                multigraph,
                port,
                subports,
                ..
            } => loop {
                let (offset, index) = subports.next_back()?;
                if let Some(res) = port_links_multiport(multigraph, *port, offset, index) {
                    return Some(res);
                }
            },
        }
    }
}

impl<'a> FusedIterator for PortLinks<'a> {}

/// Iterator over all the ports of the multiport graph.
#[derive(Clone)]
pub struct Ports<'a> {
    /// The multiport graph.
    multigraph: &'a MultiPortGraph,
    /// The wrapped ports iterator.
    ///
    /// We filter out the copy nodes from here.
    ports: portgraph::portgraph::Ports<'a>,
}

impl<'a> Ports<'a> {
    pub(super) fn new(
        multigraph: &'a MultiPortGraph,
        ports: portgraph::portgraph::Ports<'a>,
    ) -> Self {
        Self { multigraph, ports }
    }
}

impl<'a> Iterator for Ports<'a> {
    type Item = PortIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.ports.find(|&port| {
            let node = self.multigraph.port_node(port).unwrap();
            !self.multigraph.is_copy_node(node)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.ports.size_hint().1)
    }
}

impl<'a> DoubleEndedIterator for Ports<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            let port = self.ports.next_back()?;
            let node = self.multigraph.port_node(port).unwrap();
            if !self.multigraph.is_copy_node(node) {
                return Some(port);
            }
        }
    }
}

impl<'a> FusedIterator for Ports<'a> {}
