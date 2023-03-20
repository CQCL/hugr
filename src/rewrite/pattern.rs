//! Pattern matching operations on a HUGR.

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass)]
#[non_exhaustive]
pub struct Pattern {}
