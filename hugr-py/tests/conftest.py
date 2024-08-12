from __future__ import annotations

import json
import os
import pathlib
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import Self

from hugr import ext, tys
from hugr.hugr import Hugr
from hugr.ops import AsExtOp, Command, DataflowOp, ExtOp
from hugr.serialization.serial_hugr import SerialHugr
from hugr.std.float import FLOAT_T

if TYPE_CHECKING:
    from hugr.ops import ComWire

EXTENSION = ext.Extension("pytest.quantum,", ext.Version(0, 1, 0))
EXTENSION.add_op_def(
    ext.OpDef(
        name="H",
        description="Hadamard gate",
        signature=ext.OpDefSig(tys.FunctionType.endo([tys.Qubit])),
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="CX",
        description="CNOT gate",
        signature=ext.OpDefSig(tys.FunctionType.endo([tys.Qubit] * 2)),
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="Measure",
        description="Measurement operation",
        signature=ext.OpDefSig(tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])),
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="Rz",
        description="Rotation around the z-axis",
        signature=ext.OpDefSig(tys.FunctionType([tys.Qubit, FLOAT_T], [tys.Qubit])),
    )
)

E = TypeVar("E", bound=Enum)


def _load_enum(enum_cls: type[E], custom: ExtOp) -> E | None:
    ext = custom._op_def._extension
    assert ext is not None
    if ext.name == EXTENSION.name and custom._op_def.name in enum_cls.__members__:
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
        return EXTENSION.operations[self._enum.value]

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
        return EXTENSION.operations[self._enum.value]

    @classmethod
    def from_ext(cls, custom: ExtOp) -> Self | None:
        return cls(e) if (e := _load_enum(cls._Enum, custom)) else None

    def __call__(self, q0: ComWire, q1: ComWire) -> Command:
        return DataflowOp.__call__(self, q0, q1)


CX = TwoQbGate(TwoQbGate._Enum.CX)


@dataclass(frozen=True)
class MeasureDef(AsExtOp):
    def op_def(self) -> ext.OpDef:
        return EXTENSION.operations["Measure"]

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@dataclass(frozen=True)
class RzDef(AsExtOp):
    def op_def(self) -> ext.OpDef:
        return EXTENSION.operations["Rz"]

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
    _run_hugr_cmd(h.to_serial().to_json(), cmd)


def validate(h: Hugr, roundtrip: bool = True):
    cmd = [*_base_command(), "validate", "-"]
    serial = h.to_serial().to_json()
    _run_hugr_cmd(serial, cmd)

    if roundtrip:
        h2 = Hugr.from_serial(SerialHugr.load_json(json.loads(serial)))
        assert serial == h2.to_serial().to_json()


def _run_hugr_cmd(serial: str, cmd: list[str]):
    try:
        subprocess.run(cmd, check=True, input=serial.encode(), capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode()
        raise RuntimeError(error) from e
