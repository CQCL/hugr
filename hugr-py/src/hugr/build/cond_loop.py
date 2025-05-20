"""Builder classes for structured control flow
in HUGR graphs (Conditional, TailLoop).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Self, deprecated

from hugr import ops
from hugr.build.base import ParentBuilder
from hugr.build.dfg import DfBase
from hugr.hugr.base import Hugr
from hugr.tys import Sum

if TYPE_CHECKING:
    from hugr.hugr.node_port import Node, ToNode, Wire
    from hugr.tys import TypeRow


class Case(DfBase[ops.Case]):
    """Dataflow graph builder for a case in a conditional."""

    _parent_cond: Conditional | None = None

    def set_outputs(self, *outputs: Wire) -> None:
        super().set_outputs(*outputs)
        if self._parent_cond is not None:
            self._parent_cond._update_outputs(self._wire_types(outputs))


class ConditionalError(Exception):
    """Error building a :class:`Conditional`."""


@dataclass
class _IfElse(Case):
    def __init__(self, case: Case) -> None:
        self.hugr = case.hugr
        self.parent_node = case.parent_node
        self.input_node = case.input_node
        self.output_node = case.output_node
        self._parent_cond = case._parent_cond

    def _parent_conditional(self) -> Conditional:
        if self._parent_cond is None:
            msg = "If must have a parent conditional."
            raise ConditionalError(msg)
        return self._parent_cond

    @property
    def conditional_node(self) -> Node:
        """The node that represents the parent conditional."""
        return self._parent_conditional().parent_node


class If(_IfElse):
    """Build the 'if' branch of a conditional branching on a boolean value.

    Examples:
        >>> from hugr.dfg import Dfg
        >>> dfg = Dfg(tys.Qubit)
        >>> (q,) = dfg.inputs()
        >>> if_ = dfg.add_if(dfg.load(val.TRUE), q)
        >>> if_.set_outputs(if_.input_node[0])
        >>> else_= if_.add_else()
        >>> else_.set_outputs(else_.input_node[0])
        >>> dfg.hugr[else_.conditional_node].op
        Conditional(sum_ty=Bool, other_inputs=[Qubit])
    """

    def add_else(self) -> Else:
        """Finish building the 'if' branch and start building the 'else' branch."""
        return Else(self._parent_conditional().add_case(0))


class Else(_IfElse):
    """Build the 'else' branch of a conditional branching on a boolean value.

    See :class:`If` for an example.
    """

    @deprecated(
        "`Else.finish` is deprecated, use `conditional_node` instead."
    )  # TODO: Remove in a breaking change
    def finish(self) -> Node:
        """Deprecated, use `conditional_node` property."""
        return self.conditional_node  # pragma: no cover


@dataclass
class Conditional(ParentBuilder[ops.Conditional], AbstractContextManager):
    """Build a conditional branching on a sum type.

    Args:
        sum_ty: The sum type to branch on.
        other_inputs: The inputs for the conditional that aren't included in the
        sum variants. These are passed to all cases.

    Examples:
        >>> cond = Conditional(tys.Bool, [tys.Qubit])
        >>> cond.parent_op
        Conditional(sum_ty=Bool, other_inputs=[Qubit])
    """

    #: builders for each case and whether they have been built by the user yet
    _case_builders: list[tuple[Case, bool]] = field(default_factory=list)

    def __init__(self, sum_ty: Sum, other_inputs: TypeRow) -> None:
        root_op = ops.Conditional(sum_ty, other_inputs)
        hugr = Hugr(root_op)
        self._init_impl(hugr, hugr.entrypoint, len(sum_ty.variant_rows))

    def _init_impl(
        self: Conditional, hugr: Hugr, entrypoint: Node, n_cases: int
    ) -> None:
        self.hugr = hugr
        self.parent_node = entrypoint
        self._case_builders = []

        for case_id in range(n_cases):
            new_case = Case.new_nested(
                ops.Case(self.parent_op.nth_inputs(case_id)),
                self.hugr,
                self.parent_node,
            )
            new_case._parent_cond = self
            self._case_builders.append((new_case, False))

    @property
    @deprecated(
        "The 'cases' property is deprecated and will be removed in a future version."
    )  # TODO: Remove in a breaking change
    def cases(self) -> dict[int, Node | None]:
        """Map from case index to node holding the :class:`Case <hugr.ops.Case>`."""
        return {
            i: case.parent_node if b else None
            for i, (case, b) in enumerate(self._case_builders)
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        if not all(built for _, built in self._case_builders):
            msg = "All cases must be added before exiting context."
            raise ConditionalError(msg)
        return None

    @classmethod
    def new_nested(
        cls,
        sum_ty: Sum,
        other_inputs: TypeRow,
        hugr: Hugr,
        parent: ToNode | None = None,
    ) -> Conditional:
        """Build a Conditional nested inside an existing HUGR graph.

        Args:
            sum_ty: The sum type to branch on.
            other_inputs: The inputs for the conditional that aren't included in the
                sum variants. These are passed to all cases.
            hugr: The HUGR instance this Conditional is part of.
            parent: The parent node for the Conditional: defaults to the entrypoint of
              the HUGR instance.

        Returns:
            The new Conditional builder.
        """
        new = cls.__new__(cls)
        entrypoint = hugr.add_node(
            ops.Conditional(sum_ty, other_inputs),
            parent or hugr.entrypoint,
        )
        new._init_impl(hugr, entrypoint, len(sum_ty.variant_rows))
        return new

    def _update_outputs(self, outputs: TypeRow) -> None:
        if self.parent_op._outputs is None:
            self.parent_op._outputs = outputs
            self.parent_node = self.hugr._update_node_outs(
                self.parent_node, len(outputs)
            )
            if (
                self.parent_op._entrypoint_requires_wiring
                and self.hugr.entrypoint == self.parent_node
            ):
                self.hugr._connect_df_entrypoint_outputs()
        else:
            if outputs != self.parent_op._outputs:
                msg = "Mismatched case outputs."
                raise ConditionalError(msg)

    def add_case(self, case_id: int) -> Case:
        """Start building a case for the conditional.

        Args:
            case_id: The index of the case to build. Input types for the case
            are the corresponding variant of the sum type concatenated with the
            other inputs to the conditional.

        Returns:
            The new case builder.

        Raises:
            ConditionalError: If the case index is out of range.

        Examples:
            >>> cond = Conditional(tys.Bool, [tys.Qubit])
            >>> with cond.add_case(0) as case:\
                    case.set_outputs(*case.inputs())
        """
        if case_id >= len(self._case_builders):
            msg = f"Case {case_id} out of possible range."
            raise ConditionalError(msg)
        case, built = self._case_builders[case_id]
        if built:
            msg = f"Case {case_id} already built."
            raise ConditionalError(msg)
        self._case_builders[case_id] = (case, True)
        return case

    # TODO insert_case


@dataclass
class TailLoop(DfBase[ops.TailLoop]):
    """Builder for a tail-controlled loop.

    Args:
        just_inputs: Types that are only inputs to the loop body.
        rest: The remaining input types that are also output types.

    Examples:
        >>> tl = TailLoop([tys.Bool], [tys.Qubit])
        >>> tl.parent_op
        TailLoop(just_inputs=[Bool], rest=[Qubit])
    """

    def __init__(self, just_inputs: TypeRow, rest: TypeRow) -> None:
        root_op = ops.TailLoop(just_inputs, rest)
        super().__init__(root_op)

    def set_outputs(self, *outputs: Wire) -> None:
        assert len(outputs) > 0
        sum_wire = outputs[0]
        sum_type = self.hugr.port_type(sum_wire.out_port())
        assert isinstance(sum_type, Sum)
        assert len(sum_type.variant_rows) == 2
        self._set_parent_output_count(len(sum_type.variant_rows[1]) + len(outputs) - 1)

        super().set_outputs(*outputs)

    def set_loop_outputs(self, sum_wire: Wire, *rest: Wire) -> None:
        """Set the outputs of the loop body. The first wire must be the sum type
        that controls loop termination.

        Args:
            sum_wire: The wire holding the sum type that controls loop termination.
            rest: The remaining output wires (corresponding to the 'rest' types).
        """
        self.set_outputs(sum_wire, *rest)
