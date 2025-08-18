import re
from collections import Counter

import pytest

# test deprecated module
from hugr.qsystem.result import REG_INDEX_PATTERN, QsysResult, QsysShot


@pytest.mark.parametrize(
    ("identifier", "match"),
    [
        ("sadfj", None),
        ("asdf_sdf", None),
        ("asdf3h32", None),
        ("dsf[3]asdf", None),
        ("_s34fd_fd[12]", None),
        ("afsd3[34]sdf", None),
        ("asdf[2]", ("asdf", 2)),
        ("as3df[21234]", ("as3df", 21234)),
        ("as3ABdfAB[2]", ("as3ABdfAB", 2)),
    ],
)
def test_reg_index_pattern_match(identifier, match: tuple[str, int] | None):
    """Test regex pattern matches tags indexing in to registers."""
    mtch = re.match(REG_INDEX_PATTERN, identifier)
    if mtch is None:
        assert match is None
        return
    parsed = (mtch.group(1), int(mtch.group(2)))
    assert parsed == match


def test_as_dict():
    results = QsysShot()
    results.append("tag1", 1)
    results.append("tag2", 2)
    results.append("tag2", 3)
    assert results.as_dict() == {"tag1": 1, "tag2": 3}


def test_to_register_bits():
    results = QsysShot()
    results.append("c[0]", 1)
    results.append("c[1]", 0)
    results.append("c[3]", 1)
    results.append("d", [1, 0, 1, 0])
    results.append("x[5]", 1)
    results.append("x", 0)

    assert results.to_register_bits() == {"c": "1001", "d": "1010", "x": "0"}

    shots = QsysResult([results, results])
    assert shots.register_counts() == {
        "c": Counter({"1001": 2}),
        "d": Counter({"1010": 2}),
        "x": Counter({"0": 2}),
    }


@pytest.mark.parametrize(
    "results",
    [
        QsysShot([("t", 1.0)]),
        QsysShot([("t[1]", 1.0)]),
        QsysShot([("t", [1.0])]),
        QsysShot([("t[0]", [0])]),
        QsysShot([("t[0]", 3)]),
    ],
)
def test_to_register_bits_bad(results: QsysShot):
    with pytest.raises(ValueError, match="Expected bit"):
        _ = results.to_register_bits()


def test_counter():
    shot1 = QsysShot()
    shot1.append("c", [1, 0, 1, 0])
    shot1.append("d", [1, 0, 1])

    shot2 = QsysShot()
    shot2.append("c", [1, 0, 1])

    shots = QsysResult([shot1, shot2])
    assert shots.register_counts() == {
        "c": Counter({"1010": 1, "101": 1}),
        "d": Counter({"101": 1}),
    }
    with pytest.raises(ValueError, match="same length"):
        _ = shots.register_counts(strict_lengths=True)

    with pytest.raises(ValueError, match="All shots must have the same registers"):
        _ = shots.register_counts(strict_names=True)


def test_pytket():
    """Test that results observing strict tagging conventions can be converted to pytket
    shot results."""
    pytest.importorskip("pytket", reason="pytket not installed")

    hsim_shots = QsysResult(
        ([("c", [1, 0]), ("d", [1, 0, 0])], [("c", [0, 0]), ("d", [1, 0, 1])])
    )

    pytket_result = hsim_shots.to_pytket()
    from pytket._tket.unit_id import Bit
    from pytket.backends.backendresult import BackendResult
    from pytket.utils.outcomearray import OutcomeArray

    bits = [Bit("c", 0), Bit("c", 1), Bit("d", 0), Bit("d", 1), Bit("d", 2)]
    expected = BackendResult(
        c_bits=bits,
        shots=OutcomeArray.from_readouts([[1, 0, 1, 0, 0], [0, 0, 1, 0, 1]]),
    )

    assert pytket_result == expected


def test_collate_tag():
    # test use of same tag for all entries of array

    shotlist = []
    for _ in range(10):
        shot = QsysShot()
        _ = [
            shot.append(reg, 1)
            for reg, size in (("c", 3), ("d", 5))
            for _ in range(size)
        ]
        shotlist.append(shot)

    weird_shot = QsysShot((("c", 1), ("d", 1), ("d", 0), ("e", 1)))
    assert weird_shot.collate_tags() == {"c": [1], "d": [1, 0], "e": [1]}

    lst_shot = QsysShot([("lst", [1, 0, 1]), ("lst", [1, 0, 1])])
    shots = QsysResult([*shotlist, weird_shot, lst_shot])

    counter = shots.collated_counts()
    assert counter == Counter(
        {
            (("c", "111"), ("d", "11111")): 10,
            (("c", "1"), ("d", "10"), ("e", "1")): 1,
            (("lst", "101101"),): 1,
        }
    )

    float_shots = QsysResult(
        [QsysShot([("f", 1.0), ("f", 0.1)]), QsysShot([("f", [2.0]), ("g", 2.0)])]
    )

    assert float_shots.collated_shots() == [
        {"f": [1.0, 0.1]},
        {"f": [[2.0]], "g": [2.0]},
    ]


def test_qsys_shot_sequence_behavior():
    """Test that QsysShot implements Sequence protocol correctly."""
    shot = QsysShot([("a", 1), ("b", 2), ("c", 3)])

    # Test __len__
    assert len(shot) == 3

    # Test __getitem__ with int index
    assert shot[0] == ("a", 1)
    assert shot[1] == ("b", 2)
    assert shot[-1] == ("c", 3)

    # Test __getitem__ with slice
    assert shot[0:2] == [("a", 1), ("b", 2)]
    assert shot[1:] == [("b", 2), ("c", 3)]

    # Test __iter__
    entries = list(shot)
    assert entries == [("a", 1), ("b", 2), ("c", 3)]

    # Test iteration with for loop
    result = []
    for tag, value in shot:
        result.append((tag, value))
    assert result == [("a", 1), ("b", 2), ("c", 3)]


def test_qsys_result_sequence_behavior():
    """Test that QsysResult implements Sequence protocol correctly."""
    shot1 = QsysShot([("a", 1), ("b", 2)])
    shot2 = QsysShot([("c", 3), ("d", 4)])
    shot3 = QsysShot([("e", 5)])

    result = QsysResult([shot1, shot2, shot3])

    # Test __len__
    assert len(result) == 3

    # Test __getitem__ with int index
    assert result[0] == shot1
    assert result[1] == shot2
    assert result[-1] == shot3

    # Test __getitem__ with slice
    assert result[0:2] == [shot1, shot2]
    assert result[1:] == [shot2, shot3]

    # Test __iter__
    shots = list(result)
    assert shots == [shot1, shot2, shot3]

    # Test iteration behavior
    assert all(isinstance(shot, QsysShot) for shot in result)
