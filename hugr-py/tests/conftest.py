from __future__ import annotations

import json
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import tys
from hugr.hugr import Hugr
from hugr.ops import Command, Custom
from hugr.serialization.serial_hugr import SerialHugr
from hugr.std.float import FLOAT_T

if TYPE_CHECKING:
    from hugr.ops import ComWire


@dataclass(frozen=True)
class QuantumOps(Custom):
    extension: tys.ExtensionId = "tket2.quantum"


_OneQbSig = tys.FunctionType.endo([tys.Qubit])


@dataclass(frozen=True)
class OneQbGate(QuantumOps):
    op_name: str
    num_out: int = 1
    signature: tys.FunctionType = _OneQbSig

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


H = OneQbGate("H")


_TwoQbSig = tys.FunctionType.endo([tys.Qubit] * 2)


@dataclass(frozen=True)
class TwoQbGate(QuantumOps):
    op_name: str
    num_out: int = 2
    signature: tys.FunctionType = _TwoQbSig

    def __call__(self, q0: ComWire, q1: ComWire) -> Command:
        return super().__call__(q0, q1)


CX = TwoQbGate("CX")

_MeasSig = tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])


@dataclass(frozen=True)
class MeasureDef(QuantumOps):
    op_name: str = "Measure"
    num_out: int = 2
    signature: tys.FunctionType = _MeasSig

    def __call__(self, q: ComWire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()

_RzSig = tys.FunctionType([tys.Qubit, FLOAT_T], [tys.Qubit])


@dataclass(frozen=True)
class RzDef(QuantumOps):
    op_name: str = "Rz"
    num_out: int = 1
    signature: tys.FunctionType = _RzSig

    def __call__(self, q: ComWire, fl_wire: ComWire) -> Command:
        return super().__call__(q, fl_wire)


Rz = RzDef()


def validate(h: Hugr, mermaid: bool = False, roundtrip: bool = True):
    workspace_dir = pathlib.Path(__file__).parent.parent.parent
    # use the HUGR_BIN environment variable if set, otherwise use the debug build
    bin_loc = os.environ.get("HUGR_BIN", str(workspace_dir / "target/debug/hugr"))
    cmd = [bin_loc, "-"]

    if mermaid:
        cmd.append("--mermaid")
    serial = h.to_serial().to_json()
    subprocess.run(cmd, check=True, input=serial.encode())  # noqa: S603

    if roundtrip:
        h2 = Hugr.from_serial(SerialHugr.load_json(json.loads(serial)))
        assert serial == h2.to_serial().to_json()
