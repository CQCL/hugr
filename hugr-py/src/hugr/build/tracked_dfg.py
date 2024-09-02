"""Dfg builder that allows tracking a set of wires and appending operations by index."""

from collections.abc import Iterable
from typing import Any

from hugr import tys
from hugr.build.dfg import Dfg
from hugr.hugr.node_port import Node, Wire
from hugr.ops import Command, ComWire


class TrackedDfg(Dfg):
    """Dfg builder to append operations to wires by index.

    Args:
        *input_types: Input types of the Dfg.
        track_inputs: Whether to track the input wires.

    Examples:
        >>> dfg = TrackedDfg(tys.Bool, tys.Unit, track_inputs=True)
        >>> dfg.tracked
        [OutPort(Node(1), 0), OutPort(Node(1), 1)]
    """

    #: Tracked wires. None if index is no longer tracked.
    tracked: list[Wire | None]

    def __init__(self, *input_types: tys.Type, track_inputs: bool = False) -> None:
        super().__init__(*input_types)
        self.tracked = list(self.inputs()) if track_inputs else []

    def track_wire(self, wire: Wire) -> int:
        """Add a wire from this DFG to the tracked wires, and return its index.

        Args:
            wire: Wire to track.

        Returns:
            Index of the tracked wire.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit)
            >>> dfg.track_wire(dfg.inputs()[0])
            0
        """
        self.tracked.append(wire)
        return len(self.tracked) - 1

    def untrack_wire(self, index: int) -> Wire:
        """Untrack a wire by index and return it.

        Args:
            index: Index of the wire to untrack.

        Returns:
            Wire that was untracked.

        Raises:
            IndexError: If the index is not a tracked wire.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit)
            >>> w = dfg.inputs()[0]
            >>> idx = dfg.track_wire(w)
            >>> dfg.untrack_wire(idx) == w
            True
        """
        w = self.tracked_wire(index)
        self.tracked[index] = None
        return w

    def track_wires(self, wires: Iterable[Wire]) -> list[int]:
        """Set a list of wires to be tracked and return their indices.

        Args:
            wires: Wires to track.

        Returns:
            List of indices of the tracked wires.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit)
            >>> dfg.track_wires(dfg.inputs())
            [0, 1]
        """
        return [self.track_wire(w) for w in wires]

    def track_inputs(self) -> list[int]:
        """Track all input wires and return their indices.

        Returns:
            List of indices of the tracked input wires.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit)
            >>> dfg.track_inputs()
            [0, 1]
        """
        return self.track_wires(self.inputs())

    def tracked_wire(self, index: int) -> Wire:
        """Get the tracked wire at the given index.

        Args:
            index: Index of the tracked wire.

        Raises:
            IndexError: If the index is not a tracked wire.

        Returns:
            Tracked wire

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit, track_inputs=True)
            >>> dfg.tracked_wire(0) == dfg.inputs()[0]
            True
        """
        try:
            tracked = self.tracked[index]
        except IndexError:
            tracked = None
        if tracked is None:
            msg = f"Index {index} not a tracked wire."
            raise IndexError(msg)
        return tracked

    def add(self, com: Command, *, metadata: dict[str, Any] | None = None) -> Node:
        """Add a command to the DFG.

        Overrides :meth:`Dfg.add <hugr.dfg.Dfg.add>` to allow Command inputs
        to be either :class:`Wire <hugr.node_port.Wire>` or indices to tracked wires.

        Any incoming :class:`Wire <hugr.node_port.Wire>` will
        be connected directly, while any integer will be treated as a reference
        to the tracked wire at that index.

        Any tracked wires will be updated to the output of the new node at the same port
        as the incoming index.

        Args:
            com: Command to append.
            metadata: Metadata to attach to the function definition. Defaults to None.

        Returns:
            The new node.

        Raises:
            IndexError: If any input index is not a tracked wire.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, track_inputs=True)
            >>> dfg.tracked
            [OutPort(Node(1), 0)]
            >>> dfg.add(ops.Noop()(0))
            Node(3)
            >>> dfg.tracked
            [OutPort(Node(3), 0)]
        """
        wires = self._to_wires(com.incoming)
        n = self.add_op(com.op, *wires)

        for port_offset, com_wire in enumerate(com.incoming):
            if isinstance(com_wire, int):
                tracked_idx = com_wire
            else:
                continue
            # update tracked wires to matching port outputs of new node
            self.tracked[tracked_idx] = n.out(port_offset)

        return n

    def _to_wires(self, in_wires: Iterable[ComWire]) -> Iterable[Wire]:
        return (
            self.tracked_wire(inc) if isinstance(inc, int) else inc for inc in in_wires
        )

    def set_indexed_outputs(self, *in_wires: ComWire) -> None:
        """Set the Dfg outputs, using either :class:`Wire <hugr.node_port.Wire>` or
        indices to tracked wires.

        Args:
            *in_wires: Wires/indices to set as outputs.

        Raises:
            IndexError: If any input index is not a tracked wire.

        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit)
            >>> (b, i) = dfg.inputs()
            >>> dfg.track_wire(b)
            0
            >>> dfg.set_indexed_outputs(0, i)
        """
        self.set_outputs(*self._to_wires(in_wires))

    def set_tracked_outputs(self) -> None:
        """Set the Dfg outputs to the tracked wires.


        Examples:
            >>> dfg = TrackedDfg(tys.Bool, tys.Unit, track_inputs=True)
            >>> dfg.set_tracked_outputs()
        """
        self.set_outputs(*(w for w in self.tracked if w is not None))
