from hugr._cfg import Cfg
import hugr._tys as tys
from hugr._dfg import Dfg
import hugr._ops as ops
from .test_hugr_build import _validate, INT_T, DivMod


def build_basic_cfg(cfg: Cfg) -> None:
    entry = cfg.add_entry()

    entry.set_block_outputs(*entry.inputs())
    cfg.branch(entry[0], cfg.exit)


def test_basic_cfg() -> None:
    cfg = Cfg([tys.Unit, tys.Bool])
    build_basic_cfg(cfg)
    _validate(cfg.hugr)


def test_branch() -> None:
    cfg = Cfg([tys.Bool, tys.Unit, INT_T])
    entry = cfg.add_entry()
    entry.set_block_outputs(*entry.inputs())

    middle_1 = cfg.add_block([tys.Unit, INT_T])
    middle_1.set_block_outputs(*middle_1.inputs())
    middle_2 = cfg.add_block([tys.Unit, INT_T])
    u, i = middle_2.inputs()
    n = middle_2.add(DivMod(i, i))
    middle_2.set_block_outputs(u, n[0])

    cfg.branch(entry[0], middle_1)
    cfg.branch(entry[1], middle_2)

    cfg.branch(middle_1[0], cfg.exit)
    cfg.branch(middle_2[0], cfg.exit)

    _validate(cfg.hugr)


def test_nested_cfg() -> None:
    dfg = Dfg(tys.Unit, tys.Bool)

    cfg = dfg.add_cfg([tys.Unit, tys.Bool], [tys.Bool], *dfg.inputs())

    build_basic_cfg(cfg)
    dfg.set_outputs(cfg)

    _validate(dfg.hugr)


def test_dom_edge() -> None:
    cfg = Cfg([tys.Bool, tys.Unit, INT_T])
    entry = cfg.add_entry()
    b, u, i = entry.inputs()
    entry.set_block_outputs(b, i)

    # entry dominates both middles so Unit type can be used as inter-graph
    # value between basic blocks
    middle_1 = cfg.add_block([INT_T])
    middle_1.set_block_outputs(u, *middle_1.inputs())
    middle_2 = cfg.add_block([INT_T])
    middle_2.set_block_outputs(u, *middle_2.inputs())

    cfg.branch(entry[0], middle_1)
    cfg.branch(entry[1], middle_2)

    cfg.branch(middle_1[0], cfg.exit)
    cfg.branch(middle_2[0], cfg.exit)

    _validate(cfg.hugr)


def test_asymm_types() -> None:
    # test different types going to entry block's susccessors
    cfg = Cfg([tys.Bool, tys.Unit, INT_T])
    entry = cfg.add_entry()
    b, u, i = entry.inputs()

    tagged_int = entry.add(ops.Tag(0, [[INT_T], [tys.Bool]])(i))
    entry.set_block_outputs(tagged_int)

    middle = cfg.add_block([INT_T])
    # discard the int and return the bool from entry
    middle.set_block_outputs(u, b)

    # middle expects an int and exit expects a bool
    cfg.branch(entry[0], middle)
    cfg.branch(entry[1], cfg.exit)
    cfg.branch(middle[0], cfg.exit)

    _validate(cfg.hugr)
