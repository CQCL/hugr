
# Serialization Options

Given most of our tooling is in Rust it is useful to narrow our search
to options that have good [serde](https://serde.rs/) compatibility. For
most datastructures, serde allows us to to get serialization and
deserialization to many formats just by annotating the datastructures,
with no bespoke code. This is a maintainability godsend. It is also very
fast.

Unfortunately, this excludes the Tierkreis serialization technology
Protobuf, as its [serde support is
poor](https://docs.rs/serde-protobuf/latest/serde_protobuf/)
(serialization not supported at all currently). In general, as Protobuf
does its own code generation in the languages it supports, you have to
work with the datastructures it constructs (and as in the case of
Tierkreis, write a lot of boilerplate code to then convert those in to
the data you want to work with), wheras serde just handles your own
datastructures for you.

With that in mind, [this
article](https://blog.logrocket.com/rust-serialization-whats-ready-for-production-today/)
has a good summary of performance benchmarks for various options. An
interesting find here was
[FlatBuffers](https://google.github.io/flatbuffers/), Google's protobuf
alternative with zero-copy. Unfortunately, as the article describes it
is quite annoying to work with in Rust and shares the protobuf
schema-related problems mentioned above.

The highest performing target is
[bincode](https://github.com/bincode-org/bincode), but it does not seem
to be widely used and has poor python support. Another notable mention
is [CBOR](https://cbor.io/); it is however not very well performing on the benchmarks.

If we take a good balance between performance and language compatibility
MessagePack (or [msgpack](https://msgpack.org/) ) appears to be a very
solid option. It has good serde support (as well as very wide language
support in general, including a fast python package implemented in C),
is one of the top performers on benchmarks (see also [this
thesis](https://hdl.handle.net/10657/13140)),
and has small data size. Another nice benefit is that, like CBOR, it is
very similar to JSON when decoded, which, given that serde can easily
let us go between JSON and msgpack, gives us human-friendly text
visibility. The similarity to JSON also allows very easy conversion from
Python dictionaries.

# Conclusion

- Use serde to serialize and deserialize the HUGR rust struct.

- For serialised format we tentatively propose msgpack, but note that
  serde allows a very low cost change to this at a later date.

- In future if a human interpretable text format is required build a
  standalone module - this could well be [a set of MLIR
  dialects](https://github.com/PennyLaneAI/catalyst/tree/main/mlir) .

## Note

One important downside of this approach, particularly in comparison with
code-generating options like Protobuf, is that non-Rust languages (in
our case, most notably Python, and in future likely also C++) will
require code for handling the binary format and representing the data
model natively. However, for Python at least, this can be achieved
relatively simply with [Pydantic](https://docs.pydantic.dev/). This also
brings with it Python-side schema generation and validation. As an
example, the below fully implements serialization/deserialization of the
spec described in the [main document](hugr.md).

```python
from typing import Any
import ormsgpack
from pydantic import BaseModel

class MPBaseModel(BaseModel):
    def packb(self) -> bytes:
        return ormsgpack.packb(
            self, option=ormsgpack.OPT_SERIALIZE_PYDANTIC | ormsgpack.OPT_NON_STR_KEYS
        )

    @classmethod
    def unpackb(cls, b: bytes) -> "MPBaseModel":
        return cls(**ormsgpack.unpackb(b, option=ormsgpack.OPT_NON_STR_KEYS))


NodeID = int
Port = tuple[NodeID, int]  # (node, offset)
NodeWeight = Any

class Hugr(MPBaseModel):
    # (parent, #incoming, #outgoing, NodeWeight)
    nodes: list[tuple[NodeID, int, int, NodeWeight]]
    edges: list[tuple[Port, Port]]
    root: NodeID

# TODO: specify scheme for NodeWeight

with open("../hugr/foo.bin", "rb") as f:
    # print(Hugr.schema_json())
    pg = Hugr.unpackb(f.read())
    print(pg)
    outb = pg.packb()
    f.seek(0)
    assert outb == f.read()

```
