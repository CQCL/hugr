from __future__ import annotations

import json
import os
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hugr import tys, val
from hugr.hugr import Hugr
from hugr.ops import Command, Custom
from hugr.serialization.serial_hugr import SerialHugr

if TYPE_CHECKING:
    from hugr.node_port import Wire


def int_t(width: int) -> tys.Opaque:
    return tys.Opaque(
        extension="arithmetic.int.types",
        id="int",
        args=[tys.BoundedNatArg(n=width)],
        bound=tys.TypeBound.Eq,
    )


INT_T = int_t(5)


@dataclass
class IntVal(val.ExtensionValue):
    v: int

    def to_value(self) -> val.Extension:
        return val.Extension("int", INT_T, self.v)


@dataclass
class LogicOps(Custom):
    extension: tys.ExtensionId = "logic"


_NotSig = tys.FunctionType.endo([tys.Bool])


# TODO get from YAML
@dataclass
class NotDef(LogicOps):
    num_out: int = 1
    op_name: str = "Not"
    signature: tys.FunctionType = _NotSig

    def __call__(self, a: Wire) -> Command:
        return super().__call__(a)


Not = NotDef()


@dataclass
class QuantumOps(Custom):
    extension: tys.ExtensionId = "tket2.quantum"


_OneQbSig = tys.FunctionType.endo([tys.Qubit])


@dataclass
class OneQbGate(QuantumOps):
    op_name: str
    num_out: int = 1
    signature: tys.FunctionType = _OneQbSig

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


H = OneQbGate("H")

_MeasSig = tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])


@dataclass
class MeasureDef(QuantumOps):
    op_name: str = "Measure"
    num_out: int = 2
    signature: tys.FunctionType = _MeasSig

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@dataclass
class IntOps(Custom):
    extension: tys.ExtensionId = "arithmetic.int"


ARG_5 = tys.BoundedNatArg(n=5)


@dataclass
class DivModDef(IntOps):
    num_out: int = 2
    extension: tys.ExtensionId = "arithmetic.int"
    op_name: str = "idivmod_u"
    signature: tys.FunctionType = field(
        default_factory=lambda: tys.FunctionType(input=[INT_T] * 2, output=[INT_T] * 2)
    )
    args: list[tys.TypeArg] = field(default_factory=lambda: [ARG_5, ARG_5])


DivMod = DivModDef()


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
