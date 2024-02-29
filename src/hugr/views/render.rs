//! Helper methods to compute the node/edge/port style when rendering a HUGR
//! into dot or mermaid format.

use portgraph::render::{EdgeStyle, NodeStyle, PortStyle};
use portgraph::{LinkView, NodeIndex, PortIndex, PortView};

use crate::ops::OpName;
use crate::types::EdgeKind;
use crate::HugrView;

/// Formatter method to compute a node style.
pub fn node_style<H: HugrView>(h: &H) -> Box<dyn FnMut(NodeIndex) -> NodeStyle + '_> {
    Box::new(move |n| {
        NodeStyle::Box(format!(
            "({ni}) {name}",
            ni = n.index(),
            name = h.get_optype(n.into()).name()
        ))
    })
}

/// Formatter method to compute a port style.
pub fn port_style<H: HugrView>(h: &H) -> Box<dyn FnMut(PortIndex) -> PortStyle + '_> {
    let graph = h.portgraph();
    Box::new(move |port| {
        let node = graph.port_node(port).unwrap();
        let optype = h.get_optype(node.into());
        let offset = graph.port_offset(port).unwrap();
        match optype.port_kind(offset).unwrap() {
            EdgeKind::Static(ty) => PortStyle::new(html_escape::encode_text(&format!("{}", ty))),
            EdgeKind::Value(ty) => PortStyle::new(html_escape::encode_text(&format!("{}", ty))),
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
pub fn edge_style<H: HugrView>(
    h: &H,
    show_port_offsets: bool,
) -> Box<
    dyn FnMut(
            <H::Portgraph<'_> as LinkView>::LinkEndpoint,
            <H::Portgraph<'_> as LinkView>::LinkEndpoint,
        ) -> EdgeStyle
        + '_,
> {
    let hugr = h.base_hugr();
    let graph = h.portgraph();
    Box::new(move |src, tgt| {
        let src_node = graph.port_node(src).unwrap();
        let src_optype = h.get_optype(src_node.into());
        let src_offset = graph.port_offset(src).unwrap();
        let tgt_node = graph.port_node(tgt).unwrap();
        let tgt_offset = graph.port_offset(tgt).unwrap();

        let port_kind = src_optype.port_kind(src_offset).unwrap();

        let style = if hugr.hierarchy.parent(src_node) != hugr.hierarchy.parent(tgt_node) {
            EdgeStyle::Solid
        } else if port_kind == EdgeKind::StateOrder {
            EdgeStyle::Dotted
        } else {
            EdgeStyle::Solid
        };

        if !show_port_offsets {
            return style;
        }
        let label = match port_kind {
            EdgeKind::StateOrder | EdgeKind::ControlFlow => {
                format!("{}:{}", src_offset.index(), tgt_offset.index())
            }
            EdgeKind::Static(ty) | EdgeKind::Value(ty) => {
                format!("{}:{}\n{ty}", src_offset.index(), tgt_offset.index())
            }
        };
        style.with_label(label)
    })
}
