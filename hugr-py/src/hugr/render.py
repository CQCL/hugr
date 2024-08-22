"""Visualise HUGR using graphviz."""

from collections.abc import Iterable
from dataclasses import dataclass

import graphviz as gv  # type: ignore[import-untyped]
from graphviz import Digraph

from hugr.hugr import Hugr
from hugr.tys import ConstKind, Kind, OrderKind, ValueKind

from .node_port import InPort, Node, OutPort


@dataclass(frozen=True)
class Palette:
    """A set of colours used for rendering."""

    background: str
    node: str
    edge: str
    dark: str
    const: str
    discard: str
    node_border: str
    port_border: str


PALETTE: dict[str, Palette] = {
    "default": Palette(
        background="white",
        node="#ACCBF9",
        edge="#1CADE4",
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
        dark="#112D4E",
        const="#a1eea1",
        discard="#ff8888",
        node_border="#D8F8D8",
        port_border="#E8A5A5",
    ),
}


class DotRenderer:
    """Render a HUGR to a graphviz dot file.

    Args:
        palette: The palette to use for rendering. See :obj:`PALETTE` for the
        included options.
    """

    palette: Palette

    def __init__(self, palette: Palette | str | None = None) -> None:
        if palette is None:
            palette = "default"
        if isinstance(palette, str):
            palette = PALETTE[palette]
        self.palette = palette

    def render(self, hugr: Hugr) -> Digraph:
        """Render a HUGR to a graphviz dot object."""
        graph_attr = {
            "rankdir": "",
            "ranksep": "0.1",
            "nodesep": "0.15",
            "margin": "0",
            "bgcolor": self.palette.background,
        }
        if not (name := hugr[hugr.root].metadata.get("name")):
            name = ""

        graph = gv.Digraph(name, strict=False)
        graph.attr(**graph_attr)

        self._viz_node(hugr.root, hugr, graph)

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
            "label_color": self.palette.dark,
            "node_back_color": self.palette.node,
            "inputs_row": "",
            "outputs_row": "",
            "border_colour": self.palette.port_border,
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
                    back_colour=self.palette.background,
                    font_colour=self.palette.dark,
                    border_width="1",
                    border_colour=self.palette.port_border,
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
        # TODO: Port the CFG special-case rendering from guppy, and use it here
        # when a node is a CFG node.
        # See https://github.com/CQCL/guppylang/blob/7d5106cd59ad452046d0dfffd10eea9d9b617431/guppylang/hugr_builder/visualise.py#L250

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

        if hugr.children(node):
            with graph.subgraph(name=f"cluster{node.idx}") as sub:
                for child in hugr.children(node):
                    self._viz_node(child, hugr, sub)
                html_label = self._format_html_label(
                    node_back_color=self.palette.edge,
                    node_label=str(op),
                    node_data=data,
                    border_colour=self.palette.port_border,
                    inputs_row=inputs_row,
                    outputs_row=outputs_row,
                )
                sub.node(f"{node.idx}", shape="plain", label=f"<{html_label}>")
                sub.attr(label="", margin="10", color=self.palette.edge)
        else:
            html_label = self._format_html_label(
                node_back_color=self.palette.node,
                node_label=str(op),
                node_data=data,
                inputs_row=inputs_row,
                outputs_row=outputs_row,
                border_colour=self.palette.background,
            )
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
                color = self.palette.edge
            case OrderKind():
                color = self.palette.dark
            case ConstKind():
                color = self.palette.const
            case _:
                color = self.palette.dark

        graph.edge(
            self._out_port_name(src_port),
            self._in_port_name(tgt_port),
            label=label,
            color=color,
            **edge_attr,
        )
