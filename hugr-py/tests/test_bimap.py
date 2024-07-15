import pytest

from hugr.utils import BiMap, NotBijection


def test_insert_left() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    assert bimap["a"] == 1
    assert bimap.get_left(1) == "a"


def test_insert_right() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_right(1, "a")
    assert bimap["a"] == 1
    assert bimap.get_left(1) == "a"


def test_delete_left() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    del bimap["a"]
    assert bimap.get_right("a") is None
    assert bimap.get_left(1) is None


def test_delete_right() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_right(1, "a")
    bimap.delete_right(1)
    assert bimap.get_right("a") is None
    assert bimap.get_left(1) is None


def test_iter() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    bimap.insert_left("b", 2)
    bimap.insert_left("c", 3)
    assert set(bimap) == {"a", "b", "c"}
    assert list(bimap.items()) == [("a", 1), ("b", 2), ("c", 3)]


def test_len() -> None:
    bimap: BiMap[str, int] = BiMap()
    assert len(bimap) == 0
    bimap.insert_left("a", 1)
    assert len(bimap) == 1
    bimap.insert_left("b", 2)
    assert len(bimap) == 2

    bimap.delete_left("a")
    assert len(bimap) == 1


def test_existing_key() -> None:
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    bimap.insert_left("b", 1)

    assert bimap.get_right("b") == 1
    assert bimap.get_left(1) == "b"

    assert bimap.get_right("a") is None


def test_bimap_init():
    # Test with empty initial map
    bm = BiMap()
    assert len(bm) == 0

    # Test with non-empty initial map
    initial_map = {"a": 1, "b": 2}
    bm = BiMap(initial_map)
    assert len(bm) == 2

    # Test with non-bijection initial map
    invalid_map = {"a": 1, "b": 1}
    with pytest.raises(NotBijection):
        bm = BiMap(invalid_map)
