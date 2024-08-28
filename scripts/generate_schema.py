#!/usr/bin/env python
"""Dumps the json schema for `hugr._serialization.SerialHugr` to a file.

The schema is written to a file named `hugr_schema_v#.json`
in the specified output directory.
If no output directory is specified,
the schema is written to the current working directory.

usage: python generate_schema.py [<OUT_DIR>]
"""

import json
import sys
from pathlib import Path

from pydantic import ConfigDict
from pydantic.json_schema import models_json_schema

from hugr._serialization.extension import Extension, Package
from hugr._serialization.serial_hugr import SerialHugr
from hugr._serialization.testing_hugr import TestingHugr


def write_schema(
    out_dir: Path,
    name_prefix: str,
    schema: type[SerialHugr] | type[TestingHugr],
    config: ConfigDict | None = None,
    **kwargs,
):
    version = schema.get_version()
    filename = f"{name_prefix}_{version}.json"
    path = out_dir / filename
    print(f"Rebuilding model with config: {config}")
    schema._pydantic_rebuild(config or ConfigDict(), force=True, **kwargs)
    schemas = [schema, Extension, Package]
    print(f"Writing schema to {path}")
    _, top_level_schema = models_json_schema(
        [(s, "validation") for s in schemas], title="HUGR schema"
    )
    with path.open("w") as f:
        json.dump(top_level_schema, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        out_dir = Path.cwd()
    elif len(sys.argv) == 2:
        out_dir = Path(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(1)

    strict_config = ConfigDict(strict=True, extra="forbid")
    lax_config = ConfigDict(strict=False, extra="allow")
    write_schema(
        out_dir, "testing_hugr_schema_strict", TestingHugr, config=strict_config
    )
    write_schema(out_dir, "testing_hugr_schema", TestingHugr, config=lax_config)
    write_schema(out_dir, "hugr_schema_strict", SerialHugr, config=strict_config)
    write_schema(out_dir, "hugr_schema", SerialHugr, config=lax_config)
