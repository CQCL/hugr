from __future__ import annotations

import pytest

from hugr.build.dfg import Dfg
from hugr.std.int import INT_OPS_EXTENSION, INT_T
from tests.errors.util import error_snap


def test_insufficient_args_err(snapshot):
    try:
        dfg = Dfg(INT_T)
        op = INT_OPS_EXTENSION.operations["ineg"].instantiate()
        node = dfg.add_op(op, *dfg.inputs())
        dfg.set_outputs(node)
        pytest.fail("Didn't raise an error")
    except ValueError as e:
        error_snap(str(e), snapshot)
