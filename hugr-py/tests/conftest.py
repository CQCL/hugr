from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

from hugr import ext, tys
from hugr.envelope import EnvelopeConfig
from hugr.hugr import Hugr
from hugr.ops import AsExtOp, Command, Const, Custom, DataflowOp, ExtOp, RegisteredOp
from hugr.package import Package
from hugr.std.float import FLOAT_T

if TYPE_CHECKING:
    import typing

    from syrupy.assertion import SnapshotAssertion

    from hugr.hugr.node_port import Node
    from hugr.ops import ComWire

QUANTUM_EXT = ext.Extension("pytest.quantum", ext.Version(0, 1, 0))
QUANTUM_EXT.add_op_def(
    ext.OpDef(
        name="H",
        description="Hadamard gate",
        signature=ext.OpDefSig(tys.FunctionType.endo([tys.Qubit])),
    )
)

QUANTUM_EXT.add_op_def(
    ext.OpDef(
        name="CX",
        description="CNOT gate",
        signature=ext.OpDefSig(tys.FunctionType.endo([tys.Qubit] * 2)),
    )
)


E = TypeVar("E", bound=Enum)


def _load_enum(enum_cls: type[E], custom: ExtOp) -> E | None:
    ext = custom._op_def._extension
    assert ext is not None
    if ext.name == QUANTUM_EXT.name and custom._op_def.name in enum_cls.__members__:
        return enum_cls(custom._op_def.name)
    return None


@dataclass(frozen=True)
class OneQbGate(AsExtOp):
    # Have to nest enum to avoid meta class conflict
    class _Enum(Enum):
        H = "H"

    _enum: _Enum

    def __call__(self, q: ComWire) -> Command:
        return DataflowOp.__call__(self, q)

    def op_def(self) -> ext.OpDef:
        return QUANTUM_EXT.operations[self._enum.value]

    @classmethod
    def from_ext(cls, custom: ExtOp) -> Self | None:
        return cls(e) if (e := _load_enum(cls._Enum, custom)) else None


H = OneQbGate(OneQbGate._Enum.H)


@dataclass(frozen=True)
class TwoQbGate(AsExtOp):
    class _Enum(Enum):
        CX = "CX"

    _enum: _Enum

    def op_def(self) -> ext.OpDef:
        return QUANTUM_EXT.operations[self._enum.value]

    @classmethod
    def from_ext(cls, custom: ExtOp) -> Self | None:
        return cls(e) if (e := _load_enum(cls._Enum, custom)) else None

    def __call__(self, q0: ComWire, q1: ComWire) -> Command:
        return DataflowOp.__call__(self, q0, q1)


CX = TwoQbGate(TwoQbGate._Enum.CX)


@QUANTUM_EXT.register_op(
    "Measure",
    signature=tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool]),
)
@dataclass(frozen=True)
class MeasureDef(RegisteredOp):
    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@QUANTUM_EXT.register_op(
    "MeasureFree",
    signature=tys.FunctionType([tys.Qubit], [tys.Bool]),
)
@dataclass(frozen=True)
class MeasureFreeDef(RegisteredOp):
    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


MeasureFree = MeasureFreeDef()


@QUANTUM_EXT.register_op(
    "QAlloc",
    signature=tys.FunctionType([], [tys.Qubit]),
)
@dataclass(frozen=True)
class QAllocDef(RegisteredOp):
    def __call__(self) -> Command:
        return super().__call__()


QAlloc = QAllocDef()


@QUANTUM_EXT.register_op(
    "Rz",
    signature=tys.FunctionType([tys.Qubit, FLOAT_T], [tys.Qubit]),
)
@dataclass(frozen=True)
class RzDef(RegisteredOp):
    def __call__(self, q: ComWire, fl_wire: ComWire) -> Command:
        return super().__call__(q, fl_wire)


Rz = RzDef()


def _base_command() -> list[str]:
    workspace_dir = pathlib.Path(__file__).parent.parent.parent
    # use the HUGR_BIN environment variable if set, otherwise use the debug build
    bin_loc = os.environ.get("HUGR_BIN", str(workspace_dir / "target/debug/hugr"))
    return [bin_loc]


def mermaid(h: Hugr):
    """Render the Hugr as a mermaid diagram for debugging."""
    cmd = [*_base_command(), "mermaid", "-"]
    _run_hugr_cmd(h.to_str().encode(), cmd)


def validate(
    h: Hugr | Package,
    *,
    roundtrip: bool = True,
    snap: SnapshotAssertion | None = None,
):
    """Validate a HUGR or package.

    args:
        h: The HUGR or package to validate.
        roundtrip: Whether to roundtrip the HUGR through the CLI.
        snapshot: A hugr render snapshot. If not None, it will be compared against the
        rendered HUGR. Pass `--snapshot-update` to pytest to update the snapshot file.
    """
    if snap is not None:
        dot = h.render_dot() if isinstance(h, Hugr) else h.modules[0].render_dot()
        assert snap == dot.source
        if os.environ.get("HUGR_RENDER_DOT"):
            dot.pipe("svg")

    # Encoding formats to test, indexed by the format name as used by
    # `hugr convert --format`.
    FORMATS = {
        "json": EnvelopeConfig.TEXT,
        "model-exts": EnvelopeConfig.BINARY,
    }
    # Envelope formats used when exporting test hugrs.
    WRITE_FORMATS = ["json", "model-exts"]
    # Envelope formats used as target for `hugr convert` before loading back the
    # test hugrs.
    #
    # Model envelopes cannot currently be loaded from python.
    # TODO: Add model envelope loading to python, and add it to the list.
    LOAD_FORMATS = ["json"]

    cmd = [*_base_command(), "validate", "-"]

    # validate text and binary formats
    for write_fmt in WRITE_FORMATS:
        serial = h.to_bytes(FORMATS[write_fmt])
        _run_hugr_cmd(serial, cmd)

        if roundtrip:
            # Roundtrip tests:
            # Try converting to all possible LOAD_FORMATS, load them back in,
            # and check that the loaded HUGR corresponds to the original using
            # a node hash comparison.
            #
            # Run `pytest` with `-vv` to see the hash diff.
            for load_fmt in LOAD_FORMATS:
                if load_fmt != write_fmt:
                    cmd = [*_base_command(), "convert", "--format", load_fmt, "-"]
                    out = _run_hugr_cmd(serial, cmd)
                    converted_serial = out.stdout
                else:
                    converted_serial = serial
                loaded = Package.from_bytes(converted_serial)

                modules = [h] if isinstance(h, Hugr) else h.modules

                assert len(loaded.modules) == len(modules)
                for m1, m2 in zip(loaded.modules, modules, strict=True):
                    h1_hash = _NodeHash.hash_hugr(m1, "original")
                    h2_hash = _NodeHash.hash_hugr(m2, "loaded")
                    assert (
                        h1_hash == h2_hash
                    ), f"HUGRs are not the same for {write_fmt} -> {load_fmt}"

                # Lowering functions are currently ignored in Python,
                # because we don't support loading -model envelopes yet.
                for ext in loaded.extensions:
                    for op in ext.operations.values():
                        assert op.lower_funcs == []


@dataclass(frozen=True, order=True)
class _NodeHash:
    op: _OpHash
    entrypoint: bool
    input_neighbours: int
    output_neighbours: int
    input_ports: int
    output_ports: int
    input_order_edges: int
    output_order_edges: int
    is_region: bool
    node_depth: int
    children_hashes: list[_NodeHash]
    metadata: dict[str, str]

    @classmethod
    def hash_hugr(cls, h: Hugr, name: str) -> _NodeHash:
        """Returns an order-independent hash of a HUGR."""
        return cls._hash_node(h, h.module_root, 0, name)

    @classmethod
    def _hash_node(cls, h: Hugr, n: Node, depth: int, name: str) -> _NodeHash:
        children = h.children(n)
        child_hashes = sorted(cls._hash_node(h, c, depth + 1, name) for c in children)
        metadata = {k: str(v) for k, v in h[n].metadata.items()}

        # Pick a normalized representation of the op name.
        op_type = h[n].op
        if isinstance(op_type, AsExtOp):
            op_type = op_type.ext_op.to_custom_op()
            op = _OpHash(f"{op_type.extension}.{op_type.op_name}")
        elif isinstance(op_type, Custom):
            op = _OpHash(f"{op_type.extension}.{op_type.op_name}")
        elif isinstance(op_type, Const):
            # We need every custom value to have the same repr if they compare
            # equal. For example, an `IntVal(42)` should be the same as the
            # equivalent `Extension` value. This needs a lot of extra
            # unwrapping, since each class implements different `__repr__`
            # methods.
            #
            # Our solution here is to encode the value into JSON and compare those.
            # This may miss some errors, but it's the best we can do for now. Note that
            # roundtripping via `sops.Value` is not enough, since nested
            # specialized values don't get serialized straight away. (e.g.
            # StaticArrayVal's dictionary payload containing a SumValue
            # internally, see `test_val_static_array`).
            value_dict = op_type.val._to_serial_root().model_dump(mode="json")
            op = _OpHash("Const", value_dict)
        else:
            op = _OpHash(op_type.name())

        return _NodeHash(
            entrypoint=n == h.entrypoint,
            op=op,
            input_neighbours=h.num_incoming(n),
            output_neighbours=h.num_outgoing(n),
            input_ports=h.num_in_ports(n),
            output_ports=h.num_out_ports(n),
            input_order_edges=len(list(h.incoming_order_links(n))),
            output_order_edges=len(list(h.outgoing_order_links(n))),
            is_region=len(children) > 0,
            node_depth=depth,
            children_hashes=child_hashes,
            metadata=metadata,
        )


@dataclass(frozen=True)
class _OpHash:
    name: str
    payload: None | typing.Any = None

    def __lt__(self, other: _OpHash) -> bool:
        """Compare two op hashes by name and payload."""
        return (self.name, repr(self.payload)) < (other.name, repr(other.payload))


def _get_mermaid(serial: bytes) -> str:  #
    """Render a HUGR as a mermaid diagram using the CLI."""
    return _run_hugr_cmd(serial, [*_base_command(), "mermaid", "-"]).stdout.decode()


def _run_hugr_cmd(serial: bytes, cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run a HUGR command.

    The `serial` argument is the serialized HUGR to pass to the command via stdin.
    """
    try:
        return subprocess.run(cmd, check=True, input=serial, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode()
        raise RuntimeError(error) from e
