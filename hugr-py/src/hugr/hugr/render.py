"""Visualise HUGR using graphviz."""

from collections.abc import Iterable
from dataclasses import dataclass, field

import graphviz as gv  # type: ignore[import-untyped]
from graphviz import Digraph
from typing_extensions import assert_never

from hugr.hugr import Hugr
from hugr.ops import AsExtOp
from hugr.tys import CFKind, ConstKind, FunctionKind, Kind, OrderKind, ValueKind

from .node_port import InPort, Node, OutPort


@dataclass(frozen=True)
class Palette:
    """A set of colours used for rendering."""

    background: str
    node: str
    edge: str
    entrypoint_edge: str
    dark: str
    const: str
    discard: str
    node_border: str
    port_border: str

    @classmethod
    def named(cls, name: str) -> "Palette":
        return PALETTE[name]


PALETTE: dict[str, Palette] = {
    "default": Palette(
        background="white",
        node="#ACCBF9",
        edge="#1CADE4",
        entrypoint_edge="#F4A261",
        dark="black",
        const="#77CEEF",
        discard="#ff8888",
        node_border="white",
        port_border="#1CADE4",
    ),
    "nb": Palette(
        background="white",
        node="#7952B3",
        edge="#FFC107",
        entrypoint_edge="#00CFC1",
        dark="#343A40",
        const="#7c55b4",
        discard="#ff8888",
        node_border="#9d80c7",
        port_border="#ffd966",
    ),
    "zx": Palette(
        background="white",
        node="#629DD1",
        edge="#297FD5",
        entrypoint_edge="#FF8243",
        dark="#112D4E",
        const="#a1eea1",
        discard="#ff8888",
        node_border="#D8F8D8",
        port_border="#E8A5A5",
    ),
}


@dataclass
class RenderConfig:
    """Configuration for rendering a HUGR to a graphviz dot file."""

    #: The palette to use for rendering. See :obj:`PALETTE` for the included options.
    palette: Palette = field(default_factory=lambda: PALETTE["default"])
    #: If true prepend extension name to operation name.
    qualify_op_name: bool = False


class DotRenderer:
    """Render a HUGR to a graphviz dot file.

    Args:
        config: Render config
    """

    config: RenderConfig

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    def render(self, hugr: Hugr) -> Digraph:
        """Render a HUGR to a graphviz dot object."""
        graph_attr = {
            "rankdir": "",
            "ranksep": "0.1",
            "nodesep": "0.15",
            "margin": "0",
            "bgcolor": self.config.palette.background,
        }
        if not (name := hugr[hugr.module_root].metadata.get("name", None)):
            name = ""

        graph = gv.Digraph(name, strict=False)
        graph.attr(**graph_attr)

        self._viz_node(hugr.module_root, hugr, graph)

        for src_port, tgt_port in hugr.links():
            kind = hugr.port_kind(src_port)
            self._viz_link(src_port, tgt_port, kind, graph)

        return graph

    def store(self, hugr: Hugr, filename: str, format: str = "svg") -> None:
        """Render a HUGR and save it to a file.

        Args:
            hugr: The HUGR to render.
            filename: Filename for saving the rendered graph.
            format: The format used for rendering ('pdf', 'png', etc.).
                Defaults to SVG.
        """
        gv_graph = self.render(hugr)
        gv_graph.render(filename, format=format)

    _FONTFACE = "monospace"

    _HTML_LABEL_TEMPLATE = """
    <TABLE BORDER="{border_width}" CELLBORDER="0" CELLSPACING="1" CELLPADDING="1"
        BGCOLOR="{node_back_color}" COLOR="{border_colour}">
    {inputs_row}
    <TR>
        <TD>
        <TABLE BORDER="0" CELLBORDER="0">
            <TR><TD><FONT POINT-SIZE="{fontsize}" FACE="{fontface}"
                COLOR="{label_color}"><B>{node_label}</B>{node_data}</FONT></TD></TR>
        </TABLE>
        </TD>
    </TR>
    {outputs_row}
    </TABLE>
    """

    _HTML_PORTS_ROW_TEMPLATE = """
        <TR>
            <TD>
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="2">
                    <TR>
                        {port_cells}
                    </TR>
                </TABLE>
            </TD>
        </TR>
    """

    _HTML_PORT_TEMPLATE = (
        '<TD BGCOLOR="{back_colour}" COLOR="{border_colour}"'
        ' PORT="{port_id}" BORDER="{border_width}">'
        '<FONT POINT-SIZE="10.0" FACE="{fontface}" COLOR="{font_colour}">{port}</FONT>'
        "</TD>"
    )

    _INPUT_PREFIX = "in."
    _OUTPUT_PREFIX = "out."

    def _format_html_label(self, **kwargs: str) -> str:
        _HTML_LABEL_DEFAULTS = {
            "label_color": self.config.palette.dark,
            "node_back_color": self.config.palette.node,
            "inputs_row": "",
            "outputs_row": "",
            "border_colour": self.config.palette.port_border,
            "border_width": "1",
            "fontface": self._FONTFACE,
            "fontsize": 11.0,
        }
        return self._HTML_LABEL_TEMPLATE.format(**{**_HTML_LABEL_DEFAULTS, **kwargs})

    def _html_ports(self, ports: Iterable[str], id_prefix: str) -> str:
        return self._HTML_PORTS_ROW_TEMPLATE.format(
            port_cells="".join(
                self._HTML_PORT_TEMPLATE.format(
                    port=port,
                    # differentiate input and output node identifiers
                    # with a prefix
                    port_id=id_prefix + port,
                    back_colour=self.config.palette.background,
                    font_colour=self.config.palette.dark,
                    border_width="1",
                    border_colour=self.config.palette.port_border,
                    fontface=self._FONTFACE,
                )
                for port in ports
            )
        )

    def _in_port_name(self, p: InPort) -> str:
        return f"{p.node.idx}:{self._INPUT_PREFIX}{p.offset}"

    def _out_port_name(self, p: OutPort) -> str:
        return f"{p.node.idx}:{self._OUTPUT_PREFIX}{p.offset}"

    def _in_order_name(self, n: Node) -> str:
        return f"{n.idx}:{self._INPUT_PREFIX}None"

    def _out_order_name(self, n: Node) -> str:
        return f"{n.idx}:{self._OUTPUT_PREFIX}None"

    def _viz_node(self, node: Node, hugr: Hugr, graph: Digraph) -> None:
        """Render a (possibly nested) node to a graphviz graph."""
        meta = hugr[node].metadata
        if len(meta) > 0:
            data = "<BR/><BR/>" + "<BR/>".join(
                f"{key}: {value}" for key, value in meta.items()
            )
        else:
            data = ""

        in_ports = [str(i) for i in range(hugr.num_in_ports(node))]
        out_ports = [str(i) for i in range(hugr.num_out_ports(node))]
        inputs_row = (
            self._html_ports(in_ports, self._INPUT_PREFIX) if len(in_ports) > 0 else ""
        )
        outputs_row = (
            self._html_ports(out_ports, self._OUTPUT_PREFIX)
            if len(out_ports) > 0
            else ""
        )

        op = hugr[node].op
        if isinstance(op, AsExtOp) and not self.config.qualify_op_name:
            op_name = op.op_def().name
        else:
            op_name = op.name()

        label_config = {
            "node_back_color": self.config.palette.node,
            "node_label": op_name,
            "node_data": data,
            "inputs_row": inputs_row,
            "outputs_row": outputs_row,
            "border_colour": self.config.palette.background,
            "border_width": "1",
        }
        if hugr.children(node):
            # Some overrides when rendering a container node
            label_config["node_back_color"] = self.config.palette.edge
            label_config["border_colour"] = self.config.palette.port_border
        if node == hugr.entrypoint:
            label_config["node_label"] = "<b>[" + label_config["node_label"] + "]</b>"
            label_config["border_colour"] = self.config.palette.entrypoint_edge
            label_config["border_width"] = "2"

        if hugr.children(node):
            with graph.subgraph(name=f"cluster{node.idx}") as sub:
                for child in hugr.children(node):
                    self._viz_node(child, hugr, sub)
                html_label = self._format_html_label(**label_config)
                sub.node(f"{node.idx}", shape="plain", label=f"<{html_label}>")
                sub.attr(
                    label="",
                    margin="10",
                    color=label_config["border_colour"],
                    penwidth=label_config["border_width"],
                )
        else:
            html_label = self._format_html_label(**label_config)
            graph.node(f"{node.idx}", label=f"<{html_label}>", shape="plain")

    def _viz_link(
        self, src_port: OutPort, tgt_port: InPort, kind: Kind, graph: Digraph
    ) -> None:
        edge_attr = {
            "penwidth": "1.5",
            "arrowhead": "none",
            "arrowsize": "1.0",
            "fontname": self._FONTFACE,
            "fontsize": "9",
            "fontcolor": "black",
        }

        label = ""
        match kind:
            case ValueKind(ty):
                label = str(ty)
                color = self.config.palette.edge
            case OrderKind():
                color = self.config.palette.dark
            case ConstKind() | FunctionKind():
                color = self.config.palette.const
            case CFKind():
                color = self.config.palette.dark
            case _:
                assert_never(kind)

        graph.edge(
            self._out_port_name(src_port),
            self._in_port_name(tgt_port),
            label=label,
            color=color,
            **edge_attr,
        )
