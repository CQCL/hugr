from pathlib import Path

import toml  # type: ignore[import-untyped]

import hugr


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and package.__init__.py __version__ are in sync."""

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with Path.open(path, "r") as f:
        pyproject = toml.loads(f.read())
    pyproject_version = pyproject["project"]["version"]

    package_init_version = hugr.__version__

    assert package_init_version == pyproject_version
