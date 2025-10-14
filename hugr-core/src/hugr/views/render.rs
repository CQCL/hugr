//! Helper methods to compute the node/edge/port style when rendering a HUGR
//! into dot or mermaid format.

use std::collections::HashMap;

use portgraph::render::{EdgeStyle, NodeStyle, PortStyle, PresentationStyle};
use portgraph::{LinkView, MultiPortGraph, NodeIndex, PortIndex, PortView};

use crate::core::HugrNode;
use crate::hugr::internal::HugrInternals;
use crate::ops::{NamedOp, OpType};
use crate::types::EdgeKind;
use crate::{Hugr, HugrView, Node};

/// Reduced configuration for rendering a HUGR graph.
///
/// Additional options are available in the [`MermaidFormatter`] struct.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[deprecated(note = "Use `MermaidFormatter` instead", since = "0.20.2")]
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
pub struct MermaidFormatter<'h, H: HugrInternals + ?Sized = Hugr> {
    /// The HUGR to render.
    hugr: &'h H,
    /// How to display the node indices.
    node_labels: NodeLabel<H::Node>,
    /// Show port offsets in the graph edges.
    port_offsets_in_edges: bool,
    /// Show type labels on edges.
    type_labels_in_edges: bool,
    /// A node to highlight as the graph entrypoint.
    entrypoint: Option<H::Node>,
}

impl<'h, H: HugrInternals + ?Sized> MermaidFormatter<'h, H> {
    /// Create a new [`MermaidFormatter`] from a [`RenderConfig`].
    #[expect(deprecated)]
    pub fn from_render_config(config: RenderConfig<H::Node>, hugr: &'h H) -> Self {
        let node_labels = if config.node_indices {
            NodeLabel::Numeric
        } else {
            NodeLabel::None
        };
        Self {
            hugr,
            node_labels,
            port_offsets_in_edges: config.port_offsets_in_edges,
            type_labels_in_edges: config.type_labels_in_edges,
            entrypoint: config.entrypoint,
        }
    }

    /// Create a new [`MermaidFormatter`] for the given [`Hugr`].
    pub fn new(hugr: &'h H) -> Self {
        Self {
            hugr,
            node_labels: NodeLabel::Numeric,
            port_offsets_in_edges: true,
            type_labels_in_edges: true,
            entrypoint: None,
        }
    }

    /// The entrypoint to highlight in the rendered graph.
    pub fn entrypoint(&self) -> Option<H::Node> {
        self.entrypoint
    }

    /// The rendering style of the node labels.
    pub fn node_labels(&self) -> &NodeLabel<H::Node> {
        &self.node_labels
    }

    /// Whether to show port offsets on edges.
    pub fn port_offsets(&self) -> bool {
        self.port_offsets_in_edges
    }

    /// Whether to show type labels on edges.
    pub fn type_labels(&self) -> bool {
        self.type_labels_in_edges
    }

    /// Set the node labels style.
    pub fn with_node_labels(mut self, node_labels: NodeLabel<H::Node>) -> Self {
        self.node_labels = node_labels;
        self
    }

    /// Set whether to show port offsets in edges.
    pub fn with_port_offsets(mut self, show: bool) -> Self {
        self.port_offsets_in_edges = show;
        self
    }

    /// Set whether to show type labels in edges.
    pub fn with_type_labels(mut self, show: bool) -> Self {
        self.type_labels_in_edges = show;
        self
    }

    /// Set the entrypoint node to highlight.
    pub fn with_entrypoint(mut self, entrypoint: impl Into<Option<H::Node>>) -> Self {
        self.entrypoint = entrypoint.into();
        self
    }

    /// Render the graph into a Mermaid string.
    pub fn finish(self) -> String
    where
        H: HugrView,
    {
        self.hugr.mermaid_string_with_formatter(self)
    }

    pub(crate) fn with_hugr<NewH: HugrInternals<Node = H::Node>>(
        self,
        hugr: &NewH,
    ) -> MermaidFormatter<'_, NewH> {
        let MermaidFormatter {
            hugr: _,
            node_labels,
            port_offsets_in_edges,
            type_labels_in_edges,
            entrypoint,
        } = self;
        MermaidFormatter {
            hugr,
            node_labels,
            port_offsets_in_edges,
            type_labels_in_edges,
            entrypoint,
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

#[expect(deprecated)]
impl<'h, H: HugrInternals + ?Sized> TryFrom<MermaidFormatter<'h, H>> for RenderConfig<H::Node> {
    type Error = UnsupportedRenderConfig;

    fn try_from(value: MermaidFormatter<'h, H>) -> Result<Self, Self::Error> {
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

macro_rules! impl_mermaid_formatter_from {
    ($t:ty, $($lifetime:tt)?) => {
        impl<'h, $($lifetime,)? H: HugrView> From<MermaidFormatter<'h, $t>> for MermaidFormatter<'h, H> {
            fn from(value: MermaidFormatter<'h, $t>) -> Self {
                let MermaidFormatter {
                    hugr,
                    node_labels,
                    port_offsets_in_edges,
                    type_labels_in_edges,
                    entrypoint,
                } = value;
                MermaidFormatter {
                    hugr,
                    node_labels,
                    port_offsets_in_edges,
                    type_labels_in_edges,
                    entrypoint,
                }
            }
        }
    };
}

impl_mermaid_formatter_from!(&'hh H, 'hh);
impl_mermaid_formatter_from!(&'hh mut H, 'hh);
impl_mermaid_formatter_from!(std::rc::Rc<H>,);
impl_mermaid_formatter_from!(std::sync::Arc<H>,);
impl_mermaid_formatter_from!(Box<H>,);

impl<'h, H: HugrView + ToOwned> From<MermaidFormatter<'h, std::borrow::Cow<'_, H>>>
    for MermaidFormatter<'h, H>
{
    fn from(value: MermaidFormatter<'h, std::borrow::Cow<'_, H>>) -> Self {
        let MermaidFormatter {
            hugr,
            node_labels,
            port_offsets_in_edges,
            type_labels_in_edges,
            entrypoint,
        } = value;
        MermaidFormatter {
            hugr,
            node_labels,
            port_offsets_in_edges,
            type_labels_in_edges,
            entrypoint,
        }
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

#[expect(deprecated)]
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

/// Formatter method to compute a node style.
pub(in crate::hugr) fn node_style<'a>(
    h: &'a Hugr,
    formatter: MermaidFormatter<'a>,
) -> Box<dyn FnMut(NodeIndex) -> NodeStyle + 'a> {
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
    let entrypoint = formatter.entrypoint.map(Node::into_portgraph);

    match formatter.node_labels {
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
pub(in crate::hugr) fn edge_style<'a>(
    h: &'a Hugr,
    config: MermaidFormatter<'_>,
) -> Box<
    dyn FnMut(
            <MultiPortGraph<u32, u32, u32> as LinkView>::LinkEndpoint,
            <MultiPortGraph<u32, u32, u32> as LinkView>::LinkEndpoint,
        ) -> EdgeStyle
        + 'a,
> {
    let graph = &h.graph;
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
        let config = h
            .mermaid_format()
            .with_node_labels(NodeLabel::Custom(node_labels));
        insta::assert_snapshot!(h.mermaid_string_with_formatter(config));
    }

    #[test]
    fn convert_full_render_config_to_render_config() {
        let h = simple_dfg_hugr();
        let config: MermaidFormatter =
            MermaidFormatter::new(&h).with_node_labels(NodeLabel::Custom(HashMap::new()));
        #[expect(deprecated)]
        {
            assert!(RenderConfig::try_from(config).is_err());
        }
    }
}
