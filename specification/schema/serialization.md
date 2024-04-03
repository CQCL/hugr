# HUGR serialization schema

This folder contains the schema for the serialization of the HUGR objects
compliant with the [JSON Schema](https://json-schema.org/draft/2020-12/release-notes)
specification.

The model is generated from the pydantic model in the `hugr` python
package, and is used to validate the serialization format of the Rust
implementation.

A script `generate_schema.py` is provided to regenerate the schema. To update
the schema, run the following command:

```bash
just update-schema
```
