"""Builder classes for structured control flow
in HUGR graphs (Conditional, TailLoop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hugr import ops

from .dfg import _DfBase
from .hugr import Hugr, ParentBuilder

if TYPE_CHECKING:
    from .node_port import Node, ToNode, Wire
    from .tys import Sum, TypeRow


class Case(_DfBase[ops.Case]):
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
        >>> dfg.hugr[else_.finish()].op
        Conditional(sum_ty=Bool, other_inputs=[Qubit])
    """

    def add_else(self) -> Else:
        """Finish building the 'if' branch and start building the 'else' branch."""
        return Else(self._parent_conditional().add_case(0))


class Else(_IfElse):
    """Build the 'else' branch of a conditional branching on a boolean value.

    See :class:`If` for an example.
    """

    def finish(self) -> Node:
        """Finish building the if/else.

        Returns:
            The node that represents the parent conditional.
        """
        return self._parent_conditional().parent_node


@dataclass
class Conditional(ParentBuilder[ops.Conditional]):
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

    #: map from case index to node holding the :class:`Case <hugr.ops.Case>`
    cases: dict[int, Node | None]

    def __init__(self, sum_ty: Sum, other_inputs: TypeRow) -> None:
        root_op = ops.Conditional(sum_ty, other_inputs)
        hugr = Hugr(root_op)
        self._init_impl(hugr, hugr.root, len(sum_ty.variant_rows))

    def _init_impl(self: Conditional, hugr: Hugr, root: Node, n_cases: int) -> None:
        self.hugr = hugr
        self.parent_node = root
        self.cases = {i: None for i in range(n_cases)}

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
            parent: The parent node for the Conditional: defaults to the root of
              the HUGR instance.

        Returns:
            The new Conditional builder.
        """
        new = cls.__new__(cls)
        root = hugr.add_node(
            ops.Conditional(sum_ty, other_inputs),
            parent or hugr.root,
        )
        new._init_impl(hugr, root, len(sum_ty.variant_rows))
        return new

    def _update_outputs(self, outputs: TypeRow) -> None:
        if self.parent_op._outputs is None:
            self.parent_op._outputs = outputs
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
            >>> case = cond.add_case(0)
            >>> case.set_outputs(*case.inputs())
        """
        if case_id not in self.cases:
            msg = f"Case {case_id} out of possible range."
            raise ConditionalError(msg)
        input_types = self.parent_op.nth_inputs(case_id)
        new_case = Case.new_nested(
            ops.Case(input_types),
            self.hugr,
            self.parent_node,
        )
        new_case._parent_cond = self
        self.cases[case_id] = new_case.parent_node
        return new_case

    # TODO insert_case


@dataclass
class TailLoop(_DfBase[ops.TailLoop]):
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

    def set_loop_outputs(self, sum_wire: Wire, *rest: Wire) -> None:
        """Set the outputs of the loop body. The first wire must be the sum type
        that controls loop termination.

        Args:
            sum_wire: The wire holding the sum type that controls loop termination.
            rest: The remaining output wires (corresponding to the 'rest' types).
        """
        self.set_outputs(sum_wire, *rest)
