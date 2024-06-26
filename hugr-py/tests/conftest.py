from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import os
import pathlib
from hugr.node_port import Wire

from hugr.hugr import Hugr
from hugr.ops import Custom, Command
from hugr.serialization import SerialHugr
import hugr.tys as tys
import hugr.val as val
import json


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


# TODO get from YAML
@dataclass
class NotDef(LogicOps):
    num_out: int | None = 1
    op_name: str = "Not"
    signature: tys.FunctionType = tys.FunctionType.endo([tys.Bool])

    def __call__(self, a: Wire) -> Command:
        return super().__call__(a)


Not = NotDef()


@dataclass
class QuantumOps(Custom):
    extension: tys.ExtensionId = "tket2.quantum"


@dataclass
class OneQbGate(QuantumOps):
    op_name: str
    num_out: int | None = 1
    signature: tys.FunctionType = tys.FunctionType.endo([tys.Qubit])

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


H = OneQbGate("H")


@dataclass
class MeasureDef(QuantumOps):
    op_name: str = "Measure"
    num_out: int | None = 2
    signature: tys.FunctionType = tys.FunctionType([tys.Qubit], [tys.Qubit, tys.Bool])

    def __call__(self, q: Wire) -> Command:
        return super().__call__(q)


Measure = MeasureDef()


@dataclass
class IntOps(Custom):
    extension: tys.ExtensionId = "arithmetic.int"


ARG_5 = tys.BoundedNatArg(n=5)


@dataclass
class DivModDef(IntOps):
    num_out: int | None = 2
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
    subprocess.run(cmd, check=True, input=serial.encode())

    if roundtrip:
        h2 = Hugr.from_serial(SerialHugr.load_json(json.loads(serial)))
        assert serial == h2.to_serial().to_json()
