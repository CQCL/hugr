//! Helper methods to compute the node/edge/port style when rendering a HUGR
//! into dot or mermaid format.

use std::collections::HashMap;

use portgraph::render::{EdgeStyle, NodeStyle, PortStyle, PresentationStyle};
use portgraph::{LinkView, MultiPortGraph, NodeIndex, PortIndex, PortView};

use crate::core::HugrNode;
use crate::ops::{NamedOp, OpType};
use crate::types::EdgeKind;
use crate::{Hugr, HugrView, Node};

/// Reduced configuration for rendering a HUGR graph.
///
/// Additional options are available in the [`FullRenderConfig`] struct.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct RenderConfig<N = Node> {
    /// Show the node index in the graph nodes.
    pub node_indices: bool,
    /// Show port offsets in the graph edges.
    pub port_offsets_in_edges: bool,
    /// Show type labels on edges.
    pub type_labels_in_edges: bool,
    /// A node to highlight as the graph entrypoint.
    pub entrypoint: Option<N>,
}

/// Configuration for rendering a HUGR graph.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FullRenderConfig<N: HugrNode = Node> {
    /// How to display the node indices.
    pub node_labels: NodeLabel<N>,
    /// Show port offsets in the graph edges.
    pub port_offsets_in_edges: bool,
    /// Show type labels on edges.
    pub type_labels_in_edges: bool,
    /// A node to highlight as the graph entrypoint.
    pub entrypoint: Option<N>,
}

impl<N: HugrNode> From<RenderConfig<N>> for FullRenderConfig<N> {
    fn from(config: RenderConfig<N>) -> Self {
        let node_labels = if config.node_indices {
            NodeLabel::Numeric
        } else {
            NodeLabel::None
        };
        Self {
            node_labels,
            port_offsets_in_edges: config.port_offsets_in_edges,
            type_labels_in_edges: config.type_labels_in_edges,
            entrypoint: config.entrypoint,
        }
    }
}

/// An error that occurs when trying to convert a `FullRenderConfig` into a
/// `RenderConfig`.
#[derive(Debug, thiserror::Error)]
pub enum UnsupportedRenderConfig {
    /// Custom node labels are not supported in the `RenderConfig` struct.
    #[error("Custom node labels are not supported in the `RenderConfig` struct")]
    CustomNodeLabels,
}

impl<N: HugrNode> TryFrom<FullRenderConfig<N>> for RenderConfig<N> {
    type Error = UnsupportedRenderConfig;

    fn try_from(value: FullRenderConfig<N>) -> Result<Self, Self::Error> {
        if matches!(value.node_labels, NodeLabel::Custom(_)) {
            return Err(UnsupportedRenderConfig::CustomNodeLabels);
        }
        let node_indices = matches!(value.node_labels, NodeLabel::Numeric);
        Ok(Self {
            node_indices,
            port_offsets_in_edges: value.port_offsets_in_edges,
            type_labels_in_edges: value.type_labels_in_edges,
            entrypoint: value.entrypoint,
        })
    }
}

/// How to display the node indices.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub enum NodeLabel<N: HugrNode = Node> {
    /// Do not display the node index.
    None,
    /// Display the node index as a number.
    #[default]
    Numeric,
    /// Display the labels corresponding to the node indices.
    Custom(HashMap<N, String>),
}

impl<N> Default for RenderConfig<N> {
    fn default() -> Self {
        Self {
            node_indices: true,
            port_offsets_in_edges: true,
            type_labels_in_edges: true,
            entrypoint: None,
        }
    }
}

impl<N: HugrNode> Default for FullRenderConfig<N> {
    fn default() -> Self {
        Self {
            node_labels: NodeLabel::Numeric,
            port_offsets_in_edges: true,
            type_labels_in_edges: true,
            entrypoint: None,
        }
    }
}

/// Formatter method to compute a node style.
pub(in crate::hugr) fn node_style<'a>(
    h: &'a Hugr,
    config: impl Into<FullRenderConfig>,
) -> Box<dyn FnMut(NodeIndex) -> NodeStyle + 'a> {
    let config = config.into();
    fn node_name(h: &Hugr, n: NodeIndex) -> String {
        match h.get_optype(n.into()) {
            OpType::FuncDecl(f) => format!("FuncDecl: \"{}\"", f.func_name()),
            OpType::FuncDefn(f) => format!("FuncDefn: \"{}\"", f.func_name()),
            op => op.name().to_string(),
        }
    }

    let mut entrypoint_style = PresentationStyle::default();
    entrypoint_style.stroke = Some("#832561".to_string());
    entrypoint_style.stroke_width = Some("3px".to_string());
    let entrypoint = config.entrypoint.map(Node::into_portgraph);

    match config.node_labels {
        NodeLabel::Numeric => Box::new(move |n| {
            if Some(n) == entrypoint {
                NodeStyle::boxed(format!(
                    "({ni}) [**{name}**]",
                    ni = n.index(),
                    name = node_name(h, n)
                ))
                .with_attrs(entrypoint_style.clone())
            } else {
                NodeStyle::boxed(format!(
                    "({ni}) {name}",
                    ni = n.index(),
                    name = node_name(h, n)
                ))
            }
        }),
        NodeLabel::None => Box::new(move |n| {
            if Some(n) == entrypoint {
                NodeStyle::boxed(format!("[**{name}**]", name = node_name(h, n)))
                    .with_attrs(entrypoint_style.clone())
            } else {
                NodeStyle::boxed(node_name(h, n))
            }
        }),
        NodeLabel::Custom(labels) => Box::new(move |n| {
            if Some(n) == entrypoint {
                NodeStyle::boxed(format!(
                    "({label}) [**{name}**]",
                    label = labels.get(&n.into()).unwrap_or(&n.index().to_string()),
                    name = node_name(h, n)
                ))
                .with_attrs(entrypoint_style.clone())
            } else {
                NodeStyle::boxed(format!(
                    "({label}) {name}",
                    label = labels.get(&n.into()).unwrap_or(&n.index().to_string()),
                    name = node_name(h, n)
                ))
            }
        }),
    }
}

/// Formatter method to compute a port style.
pub(in crate::hugr) fn port_style(h: &Hugr) -> Box<dyn FnMut(PortIndex) -> PortStyle + '_> {
    let graph = &h.graph;
    Box::new(move |port| {
        let node = graph.port_node(port).unwrap();
        let optype = h.get_optype(node.into());
        let offset = graph.port_offset(port).unwrap();
        match optype.port_kind(offset).unwrap() {
            EdgeKind::Function(pf) => PortStyle::new(html_escape::encode_text(&format!("{pf}"))),
            EdgeKind::Const(ty) | EdgeKind::Value(ty) => {
                PortStyle::new(html_escape::encode_text(&format!("{ty}")))
            }
            EdgeKind::StateOrder => {
                if graph.port_links(port).count() > 0 {
                    PortStyle::text("", false)
                } else {
                    PortStyle::Hidden
                }
            }
            _ => PortStyle::text("", true),
        }
    })
}

/// Formatter method to compute an edge style.
#[allow(clippy::type_complexity)]
pub(in crate::hugr) fn edge_style(
    h: &Hugr,
    config: impl Into<FullRenderConfig>,
) -> Box<
    dyn FnMut(
            <MultiPortGraph as LinkView>::LinkEndpoint,
            <MultiPortGraph as LinkView>::LinkEndpoint,
        ) -> EdgeStyle
        + '_,
> {
    let graph = &h.graph;
    let config = config.into();
    Box::new(move |src, tgt| {
        let src_node = graph.port_node(src).unwrap();
        let src_optype = h.get_optype(src_node.into());
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
                EdgeKind::Const(ty) | EdgeKind::Value(ty) => Some(format!("{ty}")),
                EdgeKind::Function(pf) => Some(format!("{pf}")),
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

#[cfg(test)]
mod tests {
    use crate::{NodeIndex, builder::test::simple_dfg_hugr};

    use super::*;

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[test]
    fn test_custom_node_labels() {
        let h = simple_dfg_hugr();
        let node_labels = h
            .nodes()
            .map(|n| (n, format!("node_{}", n.index())))
            .collect();
        let config = FullRenderConfig {
            node_labels: NodeLabel::Custom(node_labels),
            ..Default::default()
        };
        insta::assert_snapshot!(h.mermaid_string_with_full_config(config));
    }

    #[test]
    fn convert_full_render_config_to_render_config() {
        let config: FullRenderConfig = FullRenderConfig {
            node_labels: NodeLabel::Custom(HashMap::new()),
            ..Default::default()
        };
        assert!(RenderConfig::try_from(config).is_err());
    }
}
