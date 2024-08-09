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
from hugr.ops import AsCustomOp, Command, Custom, DataflowOp, ExtOp
from hugr.serialization.serial_hugr import SerialHugr
from hugr.std.float import FLOAT_T

if TYPE_CHECKING:
    from hugr.ops import ComWire

EXTENSION = ext.Extension("pytest.quantum,", ext.Version(0, 1, 0))
_SINGLE_QUBIT = ext.OpDefSig(tys.FunctionType.endo([tys.Qubit]))
_TWO_QUBIT = ext.OpDefSig(tys.FunctionType.endo([tys.Qubit] * 2))
_MEAS_SIG = ext.OpDefSig(tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool]))
_RZ_SIG = ext.OpDefSig(tys.FunctionType([tys.Qubit, FLOAT_T], [tys.Qubit]))

EXTENSION.add_op_def(
    ext.OpDef(
        name="H",
        description="Hadamard gate",
        signature=_SINGLE_QUBIT,
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="CX",
        description="CNOT gate",
        signature=_TWO_QUBIT,
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="Measure",
        description="Measurement operation",
        signature=_MEAS_SIG,
    )
)

EXTENSION.add_op_def(
    ext.OpDef(
        name="Rz",
        description="Rotation around the z-axis",
        signature=_RZ_SIG,
    )
)

E = TypeVar("E", bound=Enum)


def _load_enum(enum_cls: type[E], custom: Custom) -> E | None:
    if custom.extension == EXTENSION.name and custom.name in enum_cls.__members__:
        return enum_cls(custom.name)
    return None


@dataclass(frozen=True)
class OneQbGate(AsCustomOp):
    # Have to nest enum to avoid meta class conflict
    class _Enum(Enum):
        H = "H"

    _enum: _Enum

    def __call__(self, q: ComWire) -> Command:
        return DataflowOp.__call__(self, q)

    def to_custom(self) -> Custom:
        return ExtOp(EXTENSION.operations[self._enum.value]).to_custom()

    @classmethod
    def from_custom(cls, custom: Custom) -> Self | None:
        return cls(e) if (e := _load_enum(cls._Enum, custom)) else None


H = OneQbGate(OneQbGate._Enum.H)


@dataclass(frozen=True)
class TwoQbGate(AsCustomOp):
    class _Enum(Enum):
        CX = "CX"

    _enum: _Enum

    def to_custom(self) -> Custom:
        return ExtOp(EXTENSION.operations[self._enum.value]).to_custom()

    @classmethod
    def from_custom(cls, custom: Custom) -> Self | None:
        return cls(e) if (e := _load_enum(cls._Enum, custom)) else None

    def __call__(self, q0: ComWire, q1: ComWire) -> Command:
        return DataflowOp.__call__(self, q0, q1)


CX = TwoQbGate(TwoQbGate._Enum.CX)


@dataclass(frozen=True)
class MeasureDef(AsCustomOp):
    def to_custom(self) -> Custom:
        return ExtOp(EXTENSION.operations["Measure"]).to_custom()

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@dataclass(frozen=True)
class RzDef(AsCustomOp):
    def to_custom(self) -> Custom:
        return ExtOp(EXTENSION.operations["Rz"]).to_custom()

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
