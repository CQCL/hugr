from __future__ import annotations

from dataclasses import dataclass

import hugr._ops as ops

from ._dfg import _DfBase
from ._hugr import Hugr, Node, ParentBuilder, ToNode, Wire
from ._tys import Sum, TypeRow


class Case(_DfBase[ops.Case]):
    _parent: Conditional | None = None

    def set_outputs(self, *outputs: Wire) -> None:
        super().set_outputs(*outputs)
        if self._parent is not None:
            self._parent._update_outputs(self._wire_types(outputs))


@dataclass
class Conditional(ParentBuilder[ops.Conditional]):
    hugr: Hugr
    parent_node: Node
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
            assert outputs == self.parent_op._outputs, "Mismatched case outputs."

    def add_case(self, case_id: int) -> Case:
        assert case_id in self.cases, f"Case {case_id} out of possible range."
        input_types = self.parent_op.nth_inputs(case_id)
        new_case = Case.new_nested(
            ops.Case(input_types),
            self.hugr,
            self.parent_node,
        )
        new_case._parent = self
        self.cases[case_id] = new_case.parent_node
        return new_case

    # TODO insert_case
