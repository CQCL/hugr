"""Testing setup."""

import pytest

from hugr import dfg, hugr, node_port, ops, tys, val


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
