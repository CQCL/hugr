"""HUGR model data structures."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import hugr._hugr as rust


class Term(Protocol):
    """A model term for static data such as types, constants and metadata."""

    def __str__(self) -> str:
        return rust.term_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Term":
        return rust.string_to_term(s)


@dataclass(frozen=True)
class Wildcard(Term):
    """Standin for any term."""


@dataclass(frozen=True)
class Var(Term):
    """Local variable, identified by its name."""

    name: str


@dataclass(frozen=True)
class Apply(Term):
    """Symbol application."""

    symbol: str
    args: Sequence[Term] = field(default_factory=list)


@dataclass(frozen=True)
class Splice:
    """A sequence spliced into the parent sequence."""

    seq: Term


SeqPart = Term | Splice


@dataclass(frozen=True)
class List(Term):
    """List of static data."""

    parts: Sequence[SeqPart] = field(default_factory=list)


@dataclass(frozen=True)
class Tuple(Term):
    """Tuple of static data."""

    parts: Sequence[SeqPart] = field(default_factory=list)


@dataclass(frozen=True)
class Literal(Term):
    """Static literal value."""

    value: str | float | int | bytes


@dataclass(frozen=True)
class Func(Term):
    """Function constant."""

    region: "Region"


@dataclass(frozen=True)
class ExtSet(Term):
    """Extension set. (deprecated)."""


@dataclass
class Param:
    """A parameter to a Symbol."""

    name: str
    type: Term

    def __str__(self):
        return rust.param_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Param":
        return rust.string_to_param(s)


@dataclass
class Symbol:
    """A named symbol."""

    name: str
    params: Sequence[Param] = field(default_factory=list)
    constraints: Sequence[Term] = field(default_factory=list)
    signature: Term = field(default_factory=Wildcard)

    def __str__(self):
        return rust.symbol_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Symbol":
        return rust.string_to_symbol(s)


class Op(Protocol):
    """The operation of a node."""


@dataclass(frozen=True)
class InvalidOp(Op):
    """Invalid operation intended to serve as a placeholder."""


@dataclass(frozen=True)
class Dfg(Op):
    """Dataflow graph."""


@dataclass(frozen=True)
class Cfg(Op):
    """Control flow graph."""


@dataclass(frozen=True)
class Block(Op):
    """Basic block in a control flow graph."""


@dataclass(frozen=True)
class DefineFunc(Op):
    """Function definiton."""

    symbol: Symbol


@dataclass(frozen=True)
class DeclareFunc(Op):
    """Function declaration."""

    symbol: Symbol


@dataclass(frozen=True)
class CustomOp(Op):
    """Custom operation."""

    operation: Term


@dataclass(frozen=True)
class DefineAlias(Op):
    """Alias definition."""

    symbol: Symbol
    value: Term


@dataclass(frozen=True)
class DeclareAlias(Op):
    """Alias declaration."""

    symbol: Symbol


@dataclass(frozen=True)
class TailLoop(Op):
    """Tail-controlled loop operation."""


@dataclass(frozen=True)
class Conditional(Op):
    """Conditional branch operation."""


@dataclass(frozen=True)
class DeclareConstructor(Op):
    """Constructor declaration."""

    symbol: Symbol


@dataclass(frozen=True)
class DeclareOperation(Op):
    """Operation declaration."""

    symbol: Symbol


@dataclass(frozen=True)
class Import(Op):
    """Import operation."""

    name: str


@dataclass
class Node:
    """A node in a hugr graph."""

    operation: Op = field(default_factory=lambda: InvalidOp())
    inputs: Sequence[str] = field(default_factory=list)
    outputs: Sequence[str] = field(default_factory=list)
    regions: Sequence["Region"] = field(default_factory=list)
    meta: Sequence[Term] = field(default_factory=list)
    signature: Term | None = None

    def __str__(self) -> str:
        return rust.node_to_string(self)


class RegionKind(Enum):
    """The kind of a hugr region."""

    DATA_FLOW = 0
    CONTROL_FLOW = 1
    MODULE = 2


@dataclass
class Region:
    """A hugr region containing an unordered collection of nodes."""

    kind: RegionKind = RegionKind.DATA_FLOW
    sources: Sequence[str] = field(default_factory=list)
    targets: Sequence[str] = field(default_factory=list)
    children: Sequence[Node] = field(default_factory=list)
    meta: Sequence[Term] = field(default_factory=list)
    signature: Term | None = None

    def __str__(self):
        return rust.region_to_string(self)

    @staticmethod
    def from_str(s: str) -> "Region":
        return rust.string_to_region(s)


@dataclass
class Module:
    """A top level hugr module."""

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
