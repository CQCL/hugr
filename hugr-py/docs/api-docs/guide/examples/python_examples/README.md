# HUGR Guide Python Examples

Local project for full Python programs used as examples in the HUGR guide. The project allows us to use packages for which we do not want to introduce dependencies into `hugr-py` in general (for example, `tket`, which would create a circular dependency). The `pyproject.toml` serves as the single source of versioning for the examples so it should always be set to compatible versions.

Note that the code snippets in the guide are not automatically updated when the full example files change, so if examples need to be modified because of breaking changes, make sure to update the guide as well.
