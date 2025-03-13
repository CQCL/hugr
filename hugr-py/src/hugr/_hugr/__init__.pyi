import hugr.model

def term_to_string(term: "hugr.model.Term") -> str:
    """Convert a term into a string in the hugr text syntax."""
    ...

def string_to_term(string: str) -> "hugr.model.Term":
    """Parse a term from a string in the hugr text syntax."""

def node_to_string(node: "hugr.model.Node") -> str:
    """Convert a node into a string in the hugr text syntax."""
    ...

def string_to_node(string: str) -> "hugr.model.Node":
    """Parse a node from a string in the hugr text syntax."""
    ...

def region_to_string(region: "hugr.model.Region") -> str:
    """Convert a node into a string in the hugr text syntax."""
    ...

def string_to_region(string: str) -> "hugr.model.Region":
    """Parse a region from a string in the hugr text syntax."""
    ...

def param_to_string(region: "hugr.model.Param") -> str:
    """Convert a parameter into a string in the hugr text syntax."""
    ...

def string_to_param(string: str) -> "hugr.model.Param":
    """Parse a parameter from a string in the hugr text syntax."""
    ...

def symbol_to_string(region: "hugr.model.Symbol") -> str:
    """Convert a symbol into a string in the hugr text syntax."""
    ...

def string_to_symbol(string: str) -> "hugr.model.Symbol":
    """Parse a symbol from a string in the hugr text syntax."""
    ...

def module_to_string(module: "hugr.model.Module") -> str:
    """Convert a module into a string in the hugr text syntax."""
    ...

def string_to_module(string: str) -> "hugr.model.Module":
    """Parse a module from a string in the hugr text syntax."""
    ...

def module_to_bytes(module: "hugr.model.Module") -> bytes:
    """Convert a module into the hugr binary format."""
    ...

def bytes_to_module(binary: bytes) -> "hugr.model.Module":
    """Read a module from the hugr binary format."""
    ...
