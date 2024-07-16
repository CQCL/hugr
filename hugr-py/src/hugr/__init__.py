"""`hugr` is a Python package for the Quantinuum HUGR common
representation.
"""

# This is updated by our release-please workflow, triggered by this
# annotation: x-release-please-version
__version__ = "0.4.0"


def get_serialisation_version() -> str:
    """Return the current version of the serialization schema."""
    return "v2"
