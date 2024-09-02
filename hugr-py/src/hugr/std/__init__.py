"""Types and operations from the standard extension set."""

import pkgutil

from hugr._serialization.extension import Extension as PdExtension
from hugr.ext import Extension


def _load_extension(name: str) -> Extension:
    replacement = name.replace(".", "/")
    json_str = pkgutil.get_data(__name__, f"_json_defs/{replacement}.json")
    assert json_str is not None
    return PdExtension.model_validate_json(json_str).deserialize()


PRELUDE = _load_extension("prelude")
