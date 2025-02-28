from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hugr._hugr as rust
from abc import ABC

class Term(ABC):
    def __str__(self) -> str:
        return rust.term_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Term":
        return rust.string_to_term(s)

@dataclass(frozen = True)
class Wildcard(Term):
    """Standin for any term."""
    pass

@dataclass(frozen = True)
class Var(Term):
    """Local variable, identified by its name."""
    name: str

@dataclass(frozen = True)
class Apply(Term):
    """Symbol application."""
    symbol: str
    args: list[Term]

@dataclass(frozen = True)
class Splice:
    """A sequence spliced into the parent sequence."""
    seq: Term

SeqPart = Term | Splice

@dataclass(frozen = True)
class List(Term):
    """List of static data."""
    parts: list[SeqPart] = field(default_factory = list)

@dataclass(frozen = True)
class Tuple(Term):
    """Tuple of static data."""
    parts: list[SeqPart] = field(default_factory = list)

@dataclass(frozen = True)
class Literal(Term):
    """Static literal value."""
    value: str | float | int | bytes

@dataclass(frozen = True)
class Func(Term):
    """Function constant."""
    region: "Region"

@dataclass(frozen = True)
class ExtSet(Term):
    """Extension set. (deprecated)"""
    pass

@dataclass
class Param:
    name: str
    type: Term

    def __str__(self):
        return rust.param_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Param":
        return rust.string_to_param(s)

@dataclass
class Symbol:
    name: str
    params: list[Param] = field(default_factory = list)
    constraints: list[Term] = field(default_factory = list)
    signature: Term = field(default_factory = Wildcard)

    def __str__(self):
        return rust.symbol_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Symbol":
        return rust.string_to_symbol(s)

class Op(ABC):
    pass

@dataclass(frozen = True)
class InvalidOp(Op):
    pass

@dataclass(frozen = True)
class Dfg(Op):
    pass

@dataclass(frozen = True)
class Cfg(Op):
    pass

@dataclass(frozen = True)
class Block(Op):
    pass

@dataclass(frozen = True)
class DefineFunc(Op):
    symbol: Symbol

@dataclass(frozen = True)
class DeclareFunc(Op):
    symbol: Symbol

@dataclass(frozen = True)
class CustomOp(Op):
    operation: Term

@dataclass(frozen = True)
class DefineAlias(Op):
    symbol: Symbol
    value: Term

@dataclass(frozen = True)
class DeclareAlias(Op):
    symbol: Symbol

@dataclass(frozen = True)
class TailLoop(Op):
    pass

@dataclass(frozen = True)
class Conditional(Op):
    pass

@dataclass(frozen = True)
class DeclareConstructor(Op):
    symbol: Symbol

@dataclass(frozen = True)
class DeclareOperation(Op):
    symbol: Symbol

@dataclass(frozen = True)
class Import(Op):
    name: str

@dataclass
class Node:
    operation: Op = field(default_factory=lambda: InvalidOp())
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    regions: list["Region"] = field(default_factory=list)
    meta: list[Term] = field(default_factory=list)
    signature: Optional[Term] = None

    def __str__(self) -> str:
        return rust.node_to_string(self)

class RegionKind(Enum):
    DATA_FLOW = 0
    CONTROL_FLOW = 1
    MODULE = 2

class ScopeClosure(Enum):
    OPEN = 0
    CLOSED = 1

@dataclass
class Region:
    kind: RegionKind = RegionKind.DATA_FLOW
    sources: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)
    meta: list[Term] = field(default_factory=list)
    signature: Optional[Term] = None

    def __str__(self):
        return rust.region_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Region":
        return rust.string_to_region(s)

@dataclass
class Module:
    root: Region

    def __str__(self):
        return rust.module_to_string(self)

    def __bytes__(self):
        return rust.module_to_bytes(self)

    @staticmethod
    def from_str(s: str) -> "Module":
        return rust.string_to_module(s)

    @staticmethod
    def from_bytes(b: bytes) -> "Module":
        return rust.bytes_to_module(b)
