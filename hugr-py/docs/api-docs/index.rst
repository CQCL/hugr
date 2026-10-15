HUGR Documentation
==================================

The Hierarchical Unified Graph Representation (HUGR, pronounced *hugger*) is the
common representation of quantum circuits and operations in the Quantinuum
ecosystem.

It provides a high-fidelity representation of operations, that facilitates
compilation and encodes runnable programs.

HUGR guide
----------

This is a collection of informal, conceptual guides to HUGR aimed at those who wish to
work directly with this structure and understand the motivations behind it.

.. toctree::
   :maxdepth: 1

   guide/guide_index.md

Python API v\ |hugr_py_version| reference
-----------------------------------------

The ``hugr`` python package provides a high-level API for constructing HUGRs from
basic operations, and for serializing and deserializing them.

It is intended to be used as a dependency for other high-level tools, but can
also be used directly for simple tasks.

For performance-critical tasks, see the rust API reference.

API documentation
^^^^^^^^^^^^^^^^^

.. autosummary::
   :caption: Python API Reference
   :toctree: generated
   :template: autosummary/module.rst
   :recursive:

   hugr

Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

External links
^^^^^^^^^^^^^^

.. toctree::

   PyPI <https://pypi.org/project/hugr/>
   Github <https://github.com/quantinuum/hugr/tree/main/hugr-py>

Rust API v\ |hugr_rs_version| reference
---------------------------------------

The ``hugr`` rust crate provides a low-level API for efficient manipulation
of HUGRs.

Among other features, the library includes a builder interface, serialization
and deserialization support, a pass framework, and lowerings to LLVM IR.

.. toctree::
   :caption: Rust API Reference

   Rust crate <https://crates.io/crates/hugr >
   Rust API docs <https://docs.rs/hugr/latest/hugr/>
   Github <https://github.com/quantinuum/hugr/tree/main/hugr>

HUGR specification
------------------

The formal specification of the HUGR data model is available here:

.. toctree::
   :caption: Specification
   :maxdepth: 1

   specification/index
