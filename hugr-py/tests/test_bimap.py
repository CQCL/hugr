from hugr.utils import BiMap


def test_insert_left():
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    assert bimap["a"] == 1
    assert bimap.get_left(1) == "a"


def test_insert_right():
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_right(1, "a")
    assert bimap["a"] == 1
    assert bimap.get_left(1) == "a"


def test_delete_left():
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    del bimap["a"]
    assert bimap.get_right("a") is None
    assert bimap.get_left(1) is None


def test_delete_right():
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_right(1, "a")
    bimap.delete_right(1)
    assert bimap.get_right("a") is None
    assert bimap.get_left(1) is None


def test_iter():
    bimap: BiMap[str, int] = BiMap()
    bimap.insert_left("a", 1)
    bimap.insert_left("b", 2)
    bimap.insert_left("c", 3)
    assert set(bimap) == {"a", "b", "c"}
    assert list(bimap.items()) == [("a", 1), ("b", 2), ("c", 3)]


def test_len():
    bimap: BiMap[str, int] = BiMap()
    assert len(bimap) == 0
    bimap.insert_left("a", 1)
    assert len(bimap) == 1
    bimap.insert_left("b", 2)
    assert len(bimap) == 2

    bimap.delete_left("a")
    assert len(bimap) == 1