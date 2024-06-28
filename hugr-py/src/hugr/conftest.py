import pytest
import hugr.hugr as hugr
import hugr.node_port as node_port
import hugr.dfg as dfg
import hugr.ops as ops
import hugr.tys as tys
import hugr.val as val


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
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