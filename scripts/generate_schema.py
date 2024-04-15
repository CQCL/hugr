#!/usr/bin/env python
"""Dumps the json schema for `hugr.serialization.SerialHugr` to a file.

The schema is written to a file named `hugr_schema_v#.json` in the specified output directory.
If no output directory is specified, the schema is written to the current working directory.

usage: python generate_schema.py [<OUT_DIR>]
"""

import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

from hugr.serialization import SerialHugr

if __name__ == "__main__":
    if len(sys.argv) == 1:
        out_dir = Path.cwd()
    elif len(sys.argv) == 2:
        out_dir = Path(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(1)

    version = SerialHugr.get_version()
    filename = f"hugr_schema_{version}.json"
    path = out_dir / filename

    print(f"Writing schema to {path}")

    with path.open("w") as f:
        json.dump(TypeAdapter(SerialHugr).json_schema(), f, indent=4)
