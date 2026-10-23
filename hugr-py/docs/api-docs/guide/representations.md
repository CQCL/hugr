# What is HUGR?

The Hierarchical Unified Graph Representation (HUGR, pronounced hugger 🫂) is usually described as an [intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation) (IR), designed specifically for the representation of programs that work with quantum (as well as classical) data.

In practice, what we mean by HUGR can be a little ambiguous. The [formal specification](https://quantinuum.github.io/hugr/specification/index.html) defines it as a mathematical structure. The [Python](https://quantinuum.github.io/hugr/generated/hugr.html) and [Rust](https://docs.rs/hugr/latest/hugr/) implementations define it as an in-memory representation. Finally, there are multiple serialized representations. These are closest to a true "IR" and are most useful as interchange formats, but they cannot practically be manipulated except by going through one of the in-memory forms.

Additionally, HUGR is also used informally as a noun to refer to instances of graphs adhering to the HUGR specification.

## In-memory representations

There are two standalone implementations of the HUGR specification with corresponding APIs for manipulating HUGRs, one in Python and one in Rust.

The `hugr` Python package provides a high-level API for constructing HUGRs from basic operations. If you are just getting started with HUGR or are only looking to use it for relatively simple tasks, we recommend looking at the [introductory Python guide](python_intro_1.md) to begin with.

For any performance-critical tasks, or for writing custom compilation passes and extensions, you should look at the `hugr` Rust crate. It provides a low-level API for efficient manipulation of HUGRs. Among other features, the library includes a builder interface, a pass framework, and lowerings to LLVM IR.

For example, the [Guppy compiler](https://github.com/Quantinuum/guppylang) uses the Python interface to build an initial HUGR for a given Guppy program. This HUGR is then further optimized by passes and lowered to LLVM through the Rust interface before it can be submitted to devices or simulators.


## Serialized representations

To save HUGRs or share them between libraries, both the Python and Rust interfaces provide functionality for serializing and deserializing HUGRs.

For serialization, one or more HUGRs are grouped into a package together with any extension definitions they need. The package is then converted to the `hugr-model` representation.

The `hugr-model` representation sits between runtime HUGR objects and serialized payloads and is a key component in ensuring stable serialization logic. It is a table representation that stores package data in tables and uses IDs to refer between objects. It can be encoded into two different types of payload:
- **Cap’n Proto binary payload**: This encodes the table representation directly in a compact way and is the preferred format for storage, transfer, and any workflows where conversion speed is more important than readability.
- **AST representation and textual S-expression payload**: This is a human-readable representation intended for debugging, snapshots, tests, and review.

Payloads are also wrapped in an **envelope**, which records the payload format and optional compression settings.

> Note that extension definitions are still encoded as JSON, although JSON serialization of HUGR graphs is deprecated in general.

If you want to find out more details about the serialization workflow, you can read about it in the [specification](https://quantinuum.github.io/hugr/specification/serialization.html).

### Serialization in practice

Given an existing HUGR `h`, you can save and load a binary envelope in Python as follows:

```python
from pathlib import Path

from hugr import Hugr

path = Path("program.hugr")
# Used non-standard extensions are bundled by default.
# To choose which to bundle, use h.to_bytes(include_extensions=registry).
path.write_bytes(h.to_bytes())
loaded = Hugr.from_bytes(path.read_bytes())
```

The same thing can be done in Rust in the following manner:

```rust
use std::{fs::File, io::BufReader};

use hugr::{envelope::EnvelopeConfig, Hugr};

// To bundle extensions, replace store with store_with_exts(..., &registry).
h.store(File::create("program.hugr")?, EnvelopeConfig::binary())?;
let loaded = Hugr::load(BufReader::new(File::open("program.hugr")?), None)?;
```

> Note that there is also a HUGR CLI tool that directly operates on serialized HUGRs. It is particularly useful for validating them. Find out more about the tool [here](https://github.com/Quantinuum/hugr/tree/main/hugr-cli).
