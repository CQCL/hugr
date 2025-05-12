//! Version 0 (unstable).
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
//!
//! This module defines representations of the hugr IR as plain data, designed
//! to be as independent of implementation details as feasible. It can be used
//! by the core compiler, alternative implementations or tooling that does not
//! need the power/complexity of the full compiler. We provide the following
//! in-memory representations:
//!
//! - [Table]: Efficient intermediate data structure to facilitate conversions.
//! - [AST]: Abstract syntax tree that uses direct references rather than table indices.
//!
//! The table and AST format are interconvertible and can be serialised to
//! a binary and text format, respectively:
//!
//! - [Binary]: Binary serialisation format optimised for performance and size.
//! - [Text]: Human readable s-expression based text format.
//!
//! # Logical Format
//!
//! The hugr IR is a hierarchical graph data structure. __Nodes__ represent both
//! __instructions__ that manipulate runtime values and __symbols__ which
//! represent named language objects. Instructions have __input__ and __output__ ports
//! and runtime values flow between ports when they are connected by a __link__.
//!
//! Nodes are organised into __regions__ and do not have any explicit ordering
//! between them; any schedule that respects the data dependencies between nodes
//! is valid. Regions come in three different kinds. __Module regions__ form the
//! top level of a module and can only contain symbols. __Dataflow regions__
//! describe how data flows from the region's __source__ ports to the region's
//! __target__ ports. __Controlflow regions__ are control flow graphs containing
//! dataflow __blocks__, with control flow originating from the region's source
//! ports and ending in the region's target ports.
//!
//! __Terms__ form a meta language that is used to describe types, parameters and metadata that
//! are known statically. To allow types to be parameterized by values, types and values
//! are treated uniformly as terms, enabling a restricted form of dependent typing.
//! Terms are extensible declaratively via __constructors__.
//! __Constraints__ can be used to express more complex validation rules.
//!
//! # Remaining Mismatch with `hugr-core`
//!
//! This data model was designed to encode as much of `hugr-core` as possible while also
//! filling in conceptual gaps and providing a forward-compatible foundation for future
//! development. However, there are still some mismatches with `hugr-core` that are not
//! addressed by conversions in import/export:
//!
//! - Some static types can not yet be represented in `hugr-core` although they should be.
//! - The model does not have types with a copy bound as `hugr-core` does, and instead uses
//!   a more general form of type constraints ([#1556]). Similarly, the model does not have
//!   bounded naturals. We perform a conversion for compatibility where possible, but this does
//!   not fully cover all potential cases of bounds.
//! - `hugr-model` allows to declare term constructors that serve as blueprints for constructing
//!   runtime values. This allows constants to have potentially multiple representations,
//!   which can be essential in case of very large constants that require efficient encodings.
//!   `hugr-core` is more restricted, requiring a canonical representation for constant values.
//! - `hugr-model` has support for passing closed regions as static parameters to operations,
//!   which allows for higher-order operations that require their function arguments to be
//!   statically known. We currently do not yet support converting this to `hugr-core`.
//! - In a model module, ports are connected when they share the same link. This differs from
//!   the type of port connectivity in the graph data structure used by `hugr-core`. However,
//!   `hugr-core` restricts connectivity so that in any group of connected ports there is at
//!   most one output port (for dataflow) or at most one input port (for control flow). In
//!   these cases, there is no mismatch.
//! - `hugr-core` only allows to define type aliases, but not aliases for other terms.
//! - `hugr-model` has no concept of order edges, encoding a strong preference that ordering
//!   requirements be encoded within the dataflow paradigm.
//! - Both `hugr-model` and `hugr-core` support metadata, but they use different encodings.
//!   `hugr-core` encodes metadata as JSON objects, while `hugr-model` uses terms. Using
//!   terms has the advantage that metadata can be validated with the same type checking
//!   mechanism as the rest of the model ([#1553]).
//! - `hugr-model` have a root region that corresponds to a root `Module` in `hugr-core`.
//!   `hugr-core` however can have nodes with different operations as their root ([#1554]).
//!
//! [#1556]: https://github.com/CQCL/hugr/discussions/1556
//! [#1553]: https://github.com/CQCL/hugr/issues/1553
//! [#1554]: https://github.com/CQCL/hugr/issues/1554
//! [Text]: crate::v0::ast
//! [Binary]: crate::v0::binary
//! [Table]: crate::v0::table
//! [AST]: crate::v0::ast
use ordered_float::OrderedFloat;
#[cfg(feature = "pyo3")]
use pyo3::PyTypeInfo as _;
#[cfg(feature = "pyo3")]
use pyo3::types::PyAnyMethods as _;
use smol_str::SmolStr;
use std::sync::Arc;
use table::LinkIndex;

/// Core function types.
///
/// - **Parameter:** `?inputs : (core.list core.type)`
/// - **Parameter:** `?outputs : (core.list core.type)`
/// - **Result:** `core.type`
pub const CORE_FN: &str = "core.fn";

/// The type of runtime types.
///
/// Runtime types are the types of values that can flow between nodes at runtime.
///
/// - **Result:** `?type : core.static`
pub const CORE_TYPE: &str = "core.type";

/// The type of static types.
///
/// Static types are the types of statically known parameters.
///
/// This is the only term that is its own type.
///
/// - **Result:** `?type : core.static`
pub const CORE_STATIC: &str = "core.static";

/// The type of constraints.
///
/// - **Result:** `?type : core.static`
pub const CORE_CONSTRAINT: &str = "core.constraint";

/// The constraint for non-linear runtime data.
///
/// Runtime values are copied implicitly by connecting an output port to more
/// than one input port. Similarly runtime values can be deleted implicitly when
/// an output port is not connected to any input port. In either of these cases
/// the type of the runtime value must satisfy this constraint.
///
/// - **Parameter:** `?type : core.type`
/// - **Result:** `core.constraint`
pub const CORE_NON_LINEAR: &str = "core.nonlinear";

/// The type of metadata.
///
/// - **Result:** `?type : core.static`
pub const CORE_META: &str = "core.meta";

/// Runtime algebraic data types.
///
/// Algebraic data types are sums of products of other runtime types.
///
/// - **Parameter:** `?variants : (core.list (core.list core.type))`
/// - **Result:** `core.type`
pub const CORE_ADT: &str = "core.adt";

/// Type of string literals.
///
/// - **Result:** `core.static`
pub const CORE_STR_TYPE: &str = "core.str";

/// Type of natural number literals.
///
/// - **Result:** `core.static`
pub const CORE_NAT_TYPE: &str = "core.nat";

/// Type of bytes literals.
///
/// - **Result:** `core.static`
pub const CORE_BYTES_TYPE: &str = "core.bytes";

/// Type of float literals.
///
/// - **Result:** `core.static`
pub const CORE_FLOAT_TYPE: &str = "core.float";

/// Type of a control flow edge.
///
/// - **Parameter:** `?types : (core.list core.type)`
/// - **Result:** `core.ctrl_type`
pub const CORE_CTRL: &str = "core.ctrl";

/// The type of the types for control flow edges.
///
/// - **Result:** `?type : core.static`
pub const CORE_CTRL_TYPE: &str = "core.ctrl_type";

/// The type for runtime constants.
///
/// - **Parameter:** `?type : core.type`
/// - **Result:** `core.static`
pub const CORE_CONST: &str = "core.const";

/// Constants for runtime algebraic data types.
///
/// - **Parameter:** `?variants : (core.list core.type)`
/// - **Parameter:** `?types : (core.list core.static)`
/// - **Parameter:** `?tag : core.nat`
/// - **Parameter:** `?values : (core.tuple ?types)`
/// - **Result:** `(core.const (core.adt ?variants))`
pub const CORE_CONST_ADT: &str = "core.const.adt";

/// The type for lists of static data.
///
/// Lists are finite sequences such that all elements have the same type.
/// For heterogeneous sequences, see [`CORE_TUPLE_TYPE`].
///
/// - **Parameter:** `?type : core.static`
/// - **Result:** `core.static`
pub const CORE_LIST_TYPE: &str = "core.list";

/// The type for tuples of static data.
///
/// Tuples are finite sequences that allow elements to have different types.
/// For homogeneous sequences, see [`CORE_LIST_TYPE`].
///
/// - **Parameter:** `?types : (core.list core.static)`
/// - **Result:** `core.static`
pub const CORE_TUPLE_TYPE: &str = "core.tuple";

/// Operation to call a statically known function.
///
/// - **Parameter:** `?inputs : (core.list core.type)`
/// - **Parameter:** `?outputs : (core.list core.type)`
/// - **Parameter:** `?func : (core.const (core.fn ?inputs ?outputs))`
/// - **Result:** `(core.fn ?inputs ?outputs ?ext)`
pub const CORE_CALL: &str = "core.call";

/// Operation to call a functiion known at runtime.
///
/// - **Parameter:** `?inputs : (core.list core.type)`
/// - **Parameter:** `?outputs : (core.list core.type)`
/// - **Result:** `(core.fn [(core.fn ?inputs ?outputs) ?inputs ...] ?outputs)`
pub const CORE_CALL_INDIRECT: &str = "core.call_indirect";

/// Operation to load a constant value.
///
/// - **Parameter:** `?type : core.type`
/// - **Parameter:** `?value : (core.const ?type)`
/// - **Result:** `(core.fn [] [?type])`
pub const CORE_LOAD_CONST: &str = "core.load_const";

/// Operation to create a value of an algebraic data type.
///
/// - **Parameter:** `?variants : (core.list (core.list core.type))`
/// - **Parameter:** `?types : (core.list core.type)`
/// - **Parameter:** `?tag : core.nat`
/// - **Result:** `(core.fn ?types [(core.adt ?variants)])`
pub const CORE_MAKE_ADT: &str = "core.make_adt";

/// Constructor for documentation metadata.
///
/// - **Parameter:** `?description : core.str`
/// - **Result:** `core.meta`
pub const CORE_META_DESCRIPTION: &str = "core.meta.description";

/// Metadata to tag a node or region as the entrypoint of a module.
///
/// - **Result:** `core.meta`
pub const CORE_ENTRYPOINT: &str = "core.entrypoint";

/// Constructor for JSON encoded metadata.
///
/// This is included in the model to allow for compatibility with `hugr-core`.
/// The intention is to deprecate this in the future in favor of metadata
/// expressed with custom constructors.
///
/// - **Parameter:** `?name : core.str`
/// - **Parameter:** `?json : core.str`
/// - **Result:** `core.meta`
pub const COMPAT_META_JSON: &str = "compat.meta_json";

/// Constructor for JSON encoded constants.
///
/// This is included in the model to allow for compatibility with `hugr-core`.
/// The intention is to deprecate this in the future in favor of constants
/// expressed with custom constructors.
///
/// - **Parameter:** `?type : core.type`
/// - **Parameter:** `?json : core.str`
/// - **Result:** `(core.const ?type)`
pub const COMPAT_CONST_JSON: &str = "compat.const_json";

/// Metadata constructor for order hint keys.
///
/// Nodes in a dataflow region can be annotated with a key. Each node may have
/// at most one key and the key must be unique among all nodes in the same
/// dataflow region. The parent dataflow graph can then use the
/// `order_hint.order` metadata to imply a desired ordering relation, referring
/// to the nodes by their key.
///
/// - **Parameter:** `?key : core.nat`
/// - **Result:** `core.meta`
pub const ORDER_HINT_KEY: &str = "core.order_hint.key";

/// Metadata constructor for order hints.
///
/// When this metadata is attached to a dataflow region, it can indicate a
/// preferred ordering relation between child nodes. Code generation must take
/// this into account when deciding on an execution order. The child nodes are
/// identified by a key, using the `order_hint.key` metadata.
///
/// The graph consisting of both value dependencies between nodes and order
/// hints must be directed acyclic.
///
/// - **Parameter:** `?before : core.nat`
/// - **Parameter:** `?after : core.nat`
/// - **Result:** `core.meta`
pub const ORDER_HINT_ORDER: &str = "core.order_hint.order";

pub mod ast;
pub mod binary;
pub mod scope;
pub mod table;

pub use bumpalo;

/// Type to indicate whether scopes are open or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ScopeClosure {
    /// A scope that is open and therefore not isolated from its parent scope.
    #[default]
    Open,
    /// A scope that is closed and therefore isolated from its parent scope.
    Closed,
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for ScopeClosure {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let value: usize = ob.getattr("value")?.extract()?;
        match value {
            0 => Ok(Self::Open),
            1 => Ok(Self::Closed),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid ScopeClosure.",
            )),
        }
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for ScopeClosure {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("ScopeClosure")?;

        match self {
            ScopeClosure::Open => py_class.getattr("OPEN"),
            ScopeClosure::Closed => py_class.getattr("CLOSED"),
        }
    }
}

/// The kind of a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RegionKind {
    /// Data flow region.
    #[default]
    DataFlow = 0,
    /// Control flow region.
    ControlFlow = 1,
    /// Module region.
    Module = 2,
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for RegionKind {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let value: usize = ob.getattr("value")?.extract()?;
        match value {
            0 => Ok(Self::DataFlow),
            1 => Ok(Self::ControlFlow),
            2 => Ok(Self::Module),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid RegionKind.",
            )),
        }
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for RegionKind {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_module = py.import("hugr.model")?;
        let py_class = py_module.getattr("RegionKind")?;

        match self {
            RegionKind::DataFlow => py_class.getattr("DATA_FLOW"),
            RegionKind::ControlFlow => py_class.getattr("CONTROL_FLOW"),
            RegionKind::Module => py_class.getattr("MODULE"),
        }
    }
}

/// The name of a variable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarName(SmolStr);

impl VarName {
    /// Create a new variable name.
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self(name.into())
    }
}

impl AsRef<str> for VarName {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for VarName {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let name: String = ob.extract()?;
        Ok(Self::new(name))
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for &VarName {
    type Target = pyo3::types::PyString;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.as_ref().into_pyobject(py)?)
    }
}

/// The name of a symbol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolName(SmolStr);

impl SymbolName {
    /// Create a new symbol name.
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self(name.into())
    }
}

impl AsRef<str> for SymbolName {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for SymbolName {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let name: String = ob.extract()?;
        Ok(Self::new(name))
    }
}

/// The name of a link.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkName(SmolStr);

impl LinkName {
    /// Create a new link name.
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self(name.into())
    }

    /// Create a new link name from a link index.
    #[must_use]
    pub fn new_index(index: LinkIndex) -> Self {
        // TODO: Should named and numbered links have different namespaces?
        Self(format!("{index}").into())
    }
}

impl AsRef<str> for LinkName {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for LinkName {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let name: String = ob.extract()?;
        Ok(Self::new(name))
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for &LinkName {
    type Target = pyo3::types::PyString;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.as_ref().into_pyobject(py)?)
    }
}

/// A static literal value.
///
/// Literal values may be large since they can include strings and byte
/// sequences of arbitrary length. To enable cheap cloning and sharing,
/// strings and byte sequences use reference counting.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Literal {
    /// String literal.
    Str(SmolStr),
    /// Natural number literal (unsigned 64 bit).
    Nat(u64),
    /// Byte sequence literal.
    Bytes(Arc<[u8]>),
    /// Floating point literal
    Float(OrderedFloat<f64>),
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::FromPyObject<'py> for Literal {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        if pyo3::types::PyString::is_type_of(ob) {
            let value: String = ob.extract()?;
            Ok(Literal::Str(value.into()))
        } else if pyo3::types::PyInt::is_type_of(ob) {
            let value: u64 = ob.extract()?;
            Ok(Literal::Nat(value))
        } else if pyo3::types::PyFloat::is_type_of(ob) {
            let value: f64 = ob.extract()?;
            Ok(Literal::Float(value.into()))
        } else if pyo3::types::PyBytes::is_type_of(ob) {
            let value: Vec<u8> = ob.extract()?;
            Ok(Literal::Bytes(value.into()))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid literal value.",
            ))
        }
    }
}

#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for &Literal {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Literal::Str(s) => s.as_str().into_pyobject(py)?.into_any(),
            Literal::Nat(n) => n.into_pyobject(py)?.into_any(),
            Literal::Bytes(b) => pyo3::types::PyBytes::new(py, b)
                .into_pyobject(py)?
                .into_any(),
            Literal::Float(f) => f.0.into_pyobject(py)?.into_any(),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::{prelude::*, string::string_regex};

    impl Arbitrary for Literal {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            prop_oneof![
                any::<String>().prop_map(|s| Literal::Str(s.into())),
                any::<u64>().prop_map(Literal::Nat),
                prop::collection::vec(any::<u8>(), 0..100).prop_map(|v| Literal::Bytes(v.into())),
                any::<f64>().prop_map(|f| Literal::Float(OrderedFloat(f)))
            ]
            .boxed()
        }
    }

    impl Arbitrary for SymbolName {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            string_regex(r"[a-zA-Z\-_][0-9a-zA-Z\-_](\.[a-zA-Z\-_][0-9a-zA-Z\-_])*")
                .unwrap()
                .prop_map(Self::new)
                .boxed()
        }
    }

    impl Arbitrary for VarName {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            string_regex(r"[a-zA-Z\-_][0-9a-zA-Z\-_]")
                .unwrap()
                .prop_map(Self::new)
                .boxed()
        }
    }

    proptest! {
        #[test]
        fn test_literal_text(lit: Literal) {
            let text = lit.to_string();
            let parsed: Literal = text.parse().unwrap();
            assert_eq!(lit, parsed);
        }
    }
}
