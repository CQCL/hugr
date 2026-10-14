import pytest
from semver import Version

from hugr import model, tys
from hugr.build.cfg import Cfg
from hugr.model.load import ModelImportError
from hugr.package import Package


def test_symbol_version_text():
    symbol = model.Symbol.from_str("public ext.op@0.2.3 (core.fn [] [])")

    assert symbol.name == "ext.op"
    assert symbol.version == Version.parse("0.2.3")
    assert str(symbol) == "public ext.op@0.2.3 (core.fn [] [])"


def test_import_version_text():
    package = model.Package.from_str("(hugr 0)\n\n(mod)\n\n(import ext.op@0.2.3)")

    operation = package.modules[0].root.children[0].operation
    assert operation == model.Import("ext.op", Version.parse("0.2.3"))
    assert "(import ext.op@0.2.3)" in str(package)


def test_symbol_escaped_name_text_roundtrip():
    name = "tests.integration.test_basic.test_implicit_return.<locals>.ret"
    sig = model.Apply("core.fn", [model.List([]), model.List([])])
    symbol = model.Symbol(name, "Public", signature=sig)

    text = str(symbol)
    parsed = model.Symbol.from_str(text)

    assert parsed == symbol
    assert parsed.name == name


def test_package_escaped_name_text_roundtrip():
    name = "tests.integration.test_basic.test_implicit_return.<locals>.ret"
    source = f"""(hugr 0)

(mod)

(import core.fn)

(declare-func public r#"{name}"# (core.fn [] []))
"""

    text = str(model.Package.from_str(source))
    parsed = model.Package.from_str(text)
    operation = parsed.modules[0].root.children[1].operation

    assert str(parsed) == text
    assert operation.symbol_name() == name


def test_apply_escaped_name_text_roundtrip():
    name = "tests.integration.test_linear.test_return_call.<locals>.op"
    term = model.Apply(name)
    text = str(term)
    parsed = model.Term.from_str(text)

    assert parsed == term
    assert parsed.symbol == name


@pytest.fixture
def block_model() -> tuple[model.Package, model.Node]:
    cfg = Cfg(tys.Qubit)
    entry = cfg.add_entry()
    entry.set_single_succ_outputs(*entry.inputs())
    cfg.branch_exit(entry[0])
    package = cfg.hugr.to_package().to_model()
    function = package.modules[0].root.children[0]
    cfg_node = function.regions[0].children[0]
    block = cfg_node.regions[0].children[0]
    return package, block


@pytest.mark.parametrize("region_count", [0, 2])
def test_block_import_requires_one_region(block_model, region_count: int):
    package, block = block_model
    block.regions = list(block.regions) * region_count
    with pytest.raises(ModelImportError, match="expects a single dataflow region"):
        Package.from_model(package)


@pytest.mark.parametrize("outputs", [[], [tys.Qubit]])
def test_block_import_requires_sum_output(block_model, outputs: list[tys.Type]):
    package, block = block_model
    block.regions[0].signature = model.Apply(
        "core.fn",
        [
            model.List([tys.Qubit.to_model()]),
            model.List([t.to_model() for t in outputs]),
        ],
    )
    with pytest.raises(
        ModelImportError, match="expects a sum as its first output type"
    ):
        Package.from_model(package)
