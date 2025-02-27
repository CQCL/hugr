//! Helper methods to compute the node/edge/port style when rendering a HUGR
//! into dot or mermaid format.

use portgraph::render::{EdgeStyle, NodeStyle, PortStyle};
use portgraph::{LinkView, NodeIndex, PortIndex, PortView};

use crate::ops::{NamedOp, OpType};
use crate::types::EdgeKind;
use crate::HugrView;

/// Configuration for rendering a HUGR graph.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RenderConfig {
    /// Show the node index in the graph nodes.
    pub node_indices: bool,
    /// Show port offsets in the graph edges.
    pub port_offsets_in_edges: bool,
    /// Show type labels on edges.
    pub type_labels_in_edges: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            node_indices: true,
            port_offsets_in_edges: true,
            type_labels_in_edges: true,
        }
    }
}

/// Formatter method to compute a node style.
pub(super) fn node_style<H: HugrView + ?Sized>(
    h: &H,
    config: RenderConfig,
) -> Box<dyn FnMut(NodeIndex) -> NodeStyle + '_> {
    fn node_name<H: HugrView + ?Sized>(h: &H, n: NodeIndex) -> String {
        match h.get_optype(h.get_node(n)) {
            OpType::FuncDecl(f) => format!("FuncDecl: \"{}\"", f.name),
            OpType::FuncDefn(f) => format!("FuncDefn: \"{}\"", f.name),
            op => op.name().to_string(),
        }
    }

    if config.node_indices {
        Box::new(move |n| {
            NodeStyle::Box(format!(
                "({ni}) {name}",
                ni = n.index(),
                name = node_name(h, n)
            ))
        })
    } else {
        Box::new(move |n| NodeStyle::Box(node_name(h, n)))
    }
}

/// Formatter method to compute a port style.
pub(super) fn port_style<H: HugrView + ?Sized>(
    h: &H,
    _config: RenderConfig,
) -> Box<dyn FnMut(PortIndex) -> PortStyle + '_> {
    let graph = h.portgraph();
    Box::new(move |port| {
        let node = graph.port_node(port).unwrap();
        let optype = h.get_optype(h.get_node(node));
        let offset = graph.port_offset(port).unwrap();
        match optype.port_kind(offset).unwrap() {
            EdgeKind::Function(pf) => PortStyle::new(html_escape::encode_text(&format!("{}", pf))),
            EdgeKind::Const(ty) | EdgeKind::Value(ty) => {
                PortStyle::new(html_escape::encode_text(&format!("{}", ty)))
            }
            EdgeKind::StateOrder => match graph.port_links(port).count() > 0 {
                true => PortStyle::text("", false),
                false => PortStyle::Hidden,
            },
            _ => PortStyle::text("", true),
        }
    })
}

/// Formatter method to compute an edge style.
#[allow(clippy::type_complexity)]
pub(super) fn edge_style<H: HugrView + ?Sized>(
    h: &H,
    config: RenderConfig,
) -> Box<
    dyn FnMut(
            <H::Portgraph<'_> as LinkView>::LinkEndpoint,
            <H::Portgraph<'_> as LinkView>::LinkEndpoint,
        ) -> EdgeStyle
        + '_,
> {
    let graph = h.portgraph();
    Box::new(move |src, tgt| {
        let src_node = graph.port_node(src).unwrap();
        let src_optype = h.get_optype(h.get_node(src_node));
        let src_offset = graph.port_offset(src).unwrap();
        let tgt_offset = graph.port_offset(tgt).unwrap();

        let port_kind = src_optype.port_kind(src_offset).unwrap();

        // StateOrder edges: Dotted line.
        // Control flow edges: Dashed line.
        // Static and Value edges: Solid line with label.
        let style = match port_kind {
            EdgeKind::StateOrder => EdgeStyle::Dotted,
            EdgeKind::ControlFlow => EdgeStyle::Dashed,
            EdgeKind::Const(_) | EdgeKind::Function(_) | EdgeKind::Value(_) => EdgeStyle::Solid,
        };

        // Compute the label for the edge, given the setting flags.
        fn type_label(e: EdgeKind) -> Option<String> {
            match e {
                EdgeKind::Const(ty) | EdgeKind::Value(ty) => Some(format!("{}", ty)),
                EdgeKind::Function(pf) => Some(format!("{}", pf)),
                _ => None,
            }
        }
        //
        // Only static and value edges have types to display.
        let label = match (
            config.port_offsets_in_edges,
            type_label(port_kind).filter(|_| config.type_labels_in_edges),
        ) {
            (true, Some(ty)) => {
                format!("{}:{}\n{ty}", src_offset.index(), tgt_offset.index())
            }
            (true, _) => format!("{}:{}", src_offset.index(), tgt_offset.index()),
            (false, Some(ty)) => ty.to_string(),
            _ => return style,
        };
        style.with_label(label)
    })
}
