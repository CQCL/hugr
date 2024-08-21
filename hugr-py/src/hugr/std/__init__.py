"""Types and operations from the standard extension set."""

from pathlib import Path

from hugr.ext import Extension
from hugr.serialization.extension import Extension as PdExtension

_EXT_DIR = Path(__file__).parent / "_json_defs"


def _load_extension(name: str) -> Extension:
    replacement = name.replace(".", "/")
    path = _EXT_DIR / f"{replacement}.json"
    with path.open() as f:
        return PdExtension.model_validate_json(f.read()).deserialize()
