"""Testing setup."""

import pytest

from hugr import ops, tys, val
from hugr.build import dfg
from hugr.hugr import base as hugr
from hugr.hugr import node_port


@pytest.fixture(autouse=True)
def _add_hugr(doctest_namespace):
    doctest_namespace.update(
        {
            "hugr": hugr,
            "node_port": node_port,
            "dfg": dfg,
            "ops": ops,
            "tys": tys,
            "val": val,
        }
    )
