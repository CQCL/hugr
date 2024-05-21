# from https://github.com/python-poetry/poetry/issues/144#issuecomment-877835259
import toml  # type: ignore[import-untyped]
from pathlib import Path
import hugr


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and package.__init__.py __version__ are in sync."""

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    pyproject_version = pyproject["tool"]["poetry"]["version"]

    package_init_version = hugr.__version__

    assert package_init_version == pyproject_version
