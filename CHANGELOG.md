# Changelog

## 0.1.0 (2024-01-15)

This is the initial release of the Hierarchical Unified Graph Representation.
See the representation specification available at [hugr.md](https://github.com/CQCL/hugr/blob/main/specification/hugr.md).

This release includes an up-to-date implementation of the spec, including the core definitions (control flow, data flow and module structures) as well as the Prelude extension with support for basic classical operations and types.

HUGRs can be loaded and stored using the versioned serialization format, or they can be constructed programmatically using the builder utility.
The modules `hugr::hugr::view` and `hugr::hugr::rewrite` provide an API for querying and mutating the HUGR.
For more complex operations, some algorithms are provided in `hugr::algorithms`.
