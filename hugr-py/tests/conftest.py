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
from hugr.ops import AsExtOp, Command, DataflowOp, ExtOp, RegisteredOp
from hugr.package import Package
from hugr.std.float import FLOAT_T

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from hugr.ops import ComWire

QUANTUM_EXT = ext.Extension("pytest.quantum,", ext.Version(0, 1, 0))
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
    # TODO: Use envelopes instead of legacy hugr-json
    cmd = [*_base_command(), "validate", "-"]

    serial = h.to_bytes(EnvelopeConfig.BINARY)
    _run_hugr_cmd(serial, cmd)

    if not roundtrip:
        return

    # Roundtrip checks
    if isinstance(h, Hugr):
        starting_json = h.to_str()
        h2 = Hugr.from_str(starting_json)
        roundtrip_json = h2.to_str()
        assert roundtrip_json == starting_json

        if snap is not None:
            dot = h.render_dot()
            assert snap == dot.source
            if os.environ.get("HUGR_RENDER_DOT"):
                dot.pipe("svg")
    else:
        # Package
        encoded = h.to_str(EnvelopeConfig.TEXT)
        loaded = Package.from_str(encoded)
        roundtrip_encoded = loaded.to_str(EnvelopeConfig.TEXT)
        assert encoded == roundtrip_encoded


def _run_hugr_cmd(serial: bytes, cmd: list[str]):
    """Run a HUGR command.

    The `serial` argument is the serialized HUGR to pass to the command via stdin.
    """
    try:
        subprocess.run(cmd, check=True, input=serial, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode()
        raise RuntimeError(error) from e
