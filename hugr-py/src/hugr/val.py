"""HUGR values, used for static constants in HUGR programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import hugr.serialization.ops as sops
import hugr.serialization.tys as stys
from hugr import tys
from hugr.utils import ser_it

if TYPE_CHECKING:
    from hugr.hugr import Hugr


@runtime_checkable
class Value(Protocol):
    """Abstract value definition. Must be serializable into a HUGR value."""

    def to_serial(self) -> sops.BaseValue:
        """Convert to serializable model."""
        ...  # pragma: no cover

    def to_serial_root(self) -> sops.Value:
        return sops.Value(root=self.to_serial())  # type: ignore[arg-type]

    def type_(self) -> tys.Type:
        """Report the type of the value.

        Example:
            >>> TRUE.type_()
            Bool
        """
        ...  # pragma: no cover


@dataclass
class Sum(Value):
    """Sum-of-product value.

    Example:
        >>> Sum(0, tys.Sum([[tys.Bool], [tys.Unit]]), [TRUE])
        Sum(tag=0, typ=Sum([[Bool], [Unit]]), vals=[TRUE])
    """

    #: Tag identifying the variant.
    tag: int
    #: Type of the sum: defines all possible variants.
    typ: tys.Sum
    #: The values of this variant row.
    vals: list[Value]

    @property
    def n_variants(self) -> int:
        return len(self.typ.variant_rows)

    def type_(self) -> tys.Sum:
        return self.typ

    def to_serial(self) -> sops.SumValue:
        return sops.SumValue(
            tag=self.tag,
            typ=stys.SumType(root=self.type_().to_serial()),
            vs=ser_it(self.vals),
        )


class UnitSum(Sum):
    """Simple :class:`Sum` with each variant being an empty row.

    Example:
        >>> UnitSum(0, 3)
        UnitSum(0, 3)
        >>> UnitSum(0, 1)
        Unit
        >>> assert UnitSum(0, 2) == FALSE
        >>> assert UnitSum(1, 2) == TRUE
    """

    def __init__(self, tag: int, size: int):
        super().__init__(
            tag=tag,
            typ=tys.UnitSum(size),
            vals=[],
        )

    def __repr__(self) -> str:
        if self == TRUE:
            return "TRUE"
        if self == FALSE:
            return "FALSE"
        if self == Unit:
            return "Unit"
        return f"UnitSum({self.tag}, {self.n_variants})"


def bool_value(b: bool) -> UnitSum:
    """Convert a python bool to a HUGR boolean value.

    Example:
        >>> bool_value(True)
        TRUE
        >>> bool_value(False)
        FALSE
    """
    return UnitSum(int(b), 2)


#: HUGR unit type. Sum with a single empty row variant.
Unit = UnitSum(0, 1)
#: HUGR true value.
TRUE = bool_value(True)
#: HUGR false value.
FALSE = bool_value(False)


@dataclass
class Tuple(Sum):
    """Tuple or product value, defined by a list of values.
    Internally a :class:`Sum` with a single variant row.

    Example:
        >>> tup = Tuple(TRUE, FALSE)
        >>> tup
        Tuple(TRUE, FALSE)
        >>> tup.type_()
        Tuple(Bool, Bool)

    """

    #: The values of this tuple.
    vals: list[Value]

    def __init__(self, *vals: Value):
        val_list = list(vals)
        super().__init__(
            tag=0, typ=tys.Tuple(*(v.type_() for v in val_list)), vals=val_list
        )

    # sops.TupleValue isn't an instance of sops.SumValue
    # so mypy doesn't like the override of Sum.to_serial
    def to_serial(self) -> sops.TupleValue:  # type: ignore[override]
        return sops.TupleValue(
            vs=ser_it(self.vals),
        )

    def __repr__(self) -> str:
        return f"Tuple({', '.join(map(repr, self.vals))})"


@dataclass
class Function(Value):
    """Higher order function value, defined by a :class:`Hugr <hugr.hugr.HUGR>`."""

    body: Hugr

    def type_(self) -> tys.FunctionType:
        return self.body.root_op().inner_signature()

    def to_serial(self) -> sops.FunctionValue:
        return sops.FunctionValue(
            hugr=self.body.to_serial(),
        )


@dataclass
class Extension(Value):
    """Non-core extension value."""

    #: Value name.
    name: str
    #: Value type.
    typ: tys.Type
    #: Value payload.
    val: Any
    extensions: tys.ExtensionSet = field(default_factory=tys.ExtensionSet)

    def type_(self) -> tys.Type:
        return self.typ

    def to_serial(self) -> sops.ExtensionValue:
        return sops.ExtensionValue(
            typ=self.typ.to_serial_root(),
            value=sops.CustomConst(c=self.name, v=self.val),
            extensions=self.extensions,
        )


class ExtensionValue(Value, Protocol):
    """Protocol which types can implement to be a HUGR extension value."""

    def to_value(self) -> Extension:
        """Convert to a HUGR extension value."""
        ...  # pragma: no cover

    def type_(self) -> tys.Type:
        return self.to_value().type_()

    def to_serial(self) -> sops.ExtensionValue:
        return self.to_value().to_serial()
