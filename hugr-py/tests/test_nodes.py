import pytest

from hugr.hugr import Node, OutPort


def test_index():
    n = Node(0, _num_out_ports=3)
    assert n[0] == OutPort(n, 0)
    assert n[1] == OutPort(n, 1)
    assert n[2] == OutPort(n, 2)
    assert n[-1] == OutPort(n, 2)

    with pytest.raises(IndexError, match="Index 3 out of range"):
        _ = n[3]

    with pytest.raises(IndexError, match="Index -8 out of range"):
        _ = n[-8]


def test_slices():
    n = Node(0, _num_out_ports=3)
    all_ports = [OutPort(n, i) for i in range(3)]

    assert list(n) == all_ports
    assert list(n[:0]) == []
    assert list(n[0:0]) == []
    assert list(n[0:1]) == [OutPort(n, 0)]
    assert list(n[1:2]) == [OutPort(n, 1)]
    assert list(n[:]) == all_ports
    assert list(n[0:]) == all_ports
    assert list(n[:3]) == all_ports
    assert list(n[0:3]) == all_ports
    assert list(n[0:999]) == all_ports
    assert list(n[999:1000]) == []
    assert list(n[-1:]) == [OutPort(n, 2)]
    assert list(n[-3:]) == all_ports

    with pytest.raises(IndexError, match="Index -4 out of range"):
        _ = n[-4:]

    n0 = Node(0, _num_out_ports=0)
    assert list(n0) == []
    assert list(n0[:0]) == []
    assert list(n0[:10]) == []
    assert list(n0[0:0]) == []
    assert list(n0[0:]) == []
    assert list(n0[10:]) == []
    assert list(n0[:]) == []
