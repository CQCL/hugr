from hugr import ops, tys, val
from hugr.build import Cfg, Dfg
from hugr.std.int import INT_T, DivMod, IntVal

from .conftest import validate


def build_basic_cfg(cfg: Cfg) -> None:
    with cfg.add_entry() as entry:
        entry.set_single_succ_outputs(*entry.inputs())
    cfg.branch(entry[0], cfg.exit)


def test_basic_cfg() -> None:
    cfg = Cfg(tys.Bool)
    build_basic_cfg(cfg)
    validate(cfg.hugr)


def test_branch() -> None:
    cfg = Cfg(tys.Bool, INT_T)
    entry = cfg.add_entry()
    entry.set_block_outputs(*entry.inputs())

    middle_1 = cfg.add_successor(entry[0])
    middle_1.set_single_succ_outputs(*middle_1.inputs())
    middle_2 = cfg.add_successor(entry[1])
    (i,) = middle_2.inputs()
    n = middle_2.add(DivMod(i, i))
    middle_2.set_single_succ_outputs(n[0])

    cfg.branch_exit(middle_1[0])
    cfg.branch_exit(middle_2[0])

    validate(cfg.hugr)


def test_nested_cfg() -> None:
    dfg = Dfg(tys.Bool)

    cfg = dfg.add_cfg(*dfg.inputs())

    build_basic_cfg(cfg)
    dfg.set_outputs(cfg)

    validate(dfg.hugr)


def test_dom_edge() -> None:
    cfg = Cfg(tys.Bool, tys.Unit, INT_T)
    with cfg.add_entry() as entry:
        b, u, i = entry.inputs()
        entry.set_block_outputs(b, i)

    # entry dominates both middles so Unit type can be used as inter-graph
    # value between basic blocks
    with cfg.add_successor(entry[0]) as middle_1:
        middle_1.set_block_outputs(u, *middle_1.inputs())

    with cfg.add_successor(entry[1]) as middle_2:
        middle_2.set_block_outputs(u, *middle_2.inputs())

    cfg.branch_exit(middle_1[0])
    cfg.branch_exit(middle_2[0])

    validate(cfg.hugr)


def test_asymm_types() -> None:
    # test different types going to entry block's successors
    with Cfg() as cfg:
        with cfg.add_entry() as entry:
            int_load = entry.load(IntVal(34))

            sum_ty = tys.Sum([[INT_T], [tys.Bool]])
            tagged_int = entry.add(ops.Tag(0, sum_ty)(int_load))
            entry.set_block_outputs(tagged_int)

        with cfg.add_successor(entry[0]) as middle:
            # discard the int and return the bool from entry
            middle.set_single_succ_outputs(middle.load(val.TRUE))

        # middle expects an int and exit expects a bool
        cfg.branch_exit(entry[1])
        cfg.branch_exit(middle[0])

    validate(cfg.hugr)


# TODO loop
