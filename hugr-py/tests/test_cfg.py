from hugr._cfg import Cfg
import hugr._tys as tys
from hugr._dfg import Dfg
from .test_hugr_build import _validate, INT_T, DivMod


def build_basic_cfg(cfg: Cfg) -> None:
    entry = cfg.simple_entry(1, [tys.Bool])

    entry.set_block_outputs(*entry.inputs())
    cfg.branch(entry.root.out(0), cfg.exit)


def test_basic_cfg() -> None:
    cfg = Cfg([tys.Unit, tys.Bool], [tys.Bool])
    build_basic_cfg(cfg)
    _validate(cfg.hugr)


def test_branch() -> None:
    cfg = Cfg([tys.Bool, tys.Unit, INT_T], [INT_T])
    entry = cfg.simple_entry(2, [tys.Unit, INT_T])
    entry.set_block_outputs(*entry.inputs())

    middle_1 = cfg.simple_block([tys.Unit, INT_T], 1, [INT_T])
    middle_1.set_block_outputs(*middle_1.inputs())
    middle_2 = cfg.simple_block([tys.Unit, INT_T], 1, [INT_T])
    u, i = middle_2.inputs()
    n = middle_2.add(DivMod(i, i))
    middle_2.set_block_outputs(u, n[0])

    cfg.branch(entry.root.out(0), middle_1.root)
    cfg.branch(entry.root.out(1), middle_2.root)

    cfg.branch(middle_1.root.out(0), cfg.exit)
    cfg.branch(middle_2.root.out(0), cfg.exit)

    _validate(cfg.hugr)


def test_nested_cfg() -> None:
    dfg = Dfg([tys.Unit, tys.Bool], [tys.Bool])

    cfg = dfg.add_cfg([tys.Unit, tys.Bool], [tys.Bool], *dfg.inputs())

    build_basic_cfg(cfg)
    dfg.set_outputs(cfg.root)

    _validate(dfg.hugr)


def test_dom_edge() -> None:
    cfg = Cfg([tys.Bool, tys.Unit, INT_T], [INT_T])
    entry = cfg.simple_entry(2, [INT_T])
    b, u, i = entry.inputs()
    entry.set_block_outputs(b, i)

    # entry dominates both middles so Unit type can be used as inter-graph
    # value between basic blocks
    middle_1 = cfg.simple_block([INT_T], 1, [INT_T])
    middle_1.set_block_outputs(u, *middle_1.inputs())
    middle_2 = cfg.simple_block([INT_T], 1, [INT_T])
    middle_2.set_block_outputs(u, *middle_2.inputs())

    cfg.branch(entry.root.out(0), middle_1.root)
    cfg.branch(entry.root.out(1), middle_2.root)

    cfg.branch(middle_1.root.out(0), cfg.exit)
    cfg.branch(middle_2.root.out(0), cfg.exit)

    _validate(cfg.hugr)
