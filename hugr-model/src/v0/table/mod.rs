//! Table representation of hugr modules.
//!
//! Instead of directly nesting data structures, we store them in tables and
//! refer to them by their id in the table. Variables, symbols and links are
//! fully resolved: uses refer to the id of the declaration. This allows the
//! table representation to be read from the [binary format] and imported into
//! the core data structures without having to perform potentially costly name
//! resolutions.
//!
//! The tabling is also used for deduplication of terms. In practice, many terms
//! will share the same subterms, and we can save memory and validation time by
//! storing them only once. However we allow non-deduplicated terms for cases in
//! which terms carry additional identity over just their structure. For
//! instance, structurally identical terms could originate from different
//! locations in a text file and therefore should be treated differently when
//! locating type errors.
//!
//! This format is intended to be used as an intermediary data structure to
//! convert between different representations (such as the [binary format], the
//! [text format] or internal compiler data structures). To make this efficient,
//! we use arena allocation via the [`bumpalo`] crate to efficiently construct and
//! tear down this representation. The data structures in this module therefore carry
//! a lifetime parameter that indicates the lifetime of the arena.
//!
//! [binary format]: crate::v0::binary
//! [text format]: crate::v0::ast

use smol_str::SmolStr;
use thiserror::Error;

mod view;
use super::{Literal, RegionKind, ast};
pub use view::View;

/// A package consisting of a sequence of [`Module`]s.
///
/// See [`ast::Package`] for the AST representation.
///
/// [`ast::Package`]: crate::v0::ast::Package
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Package<'a> {
    /// The modules in the package.
    pub modules: Vec<Module<'a>>,
}

impl Package<'_> {
    /// Convert the package to the [ast] representation.
    ///
    /// [ast]: crate::v0::ast
    #[must_use]
    pub fn as_ast(&self) -> Option<ast::Package> {
        let modules = self
            .modules
            .iter()
            .map(Module::as_ast)
            .collect::<Option<_>>()?;
        Some(ast::Package { modules })
    }
}

/// A module consisting of a hugr graph together with terms.
///
/// See [`ast::Module`] for the AST representation.
///
/// [`ast::Module`]: crate::v0::ast::Module
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Module<'a> {
    /// The id of the root region.
    pub root: RegionId,
    /// Table of [`Node`]s.
    pub nodes: Vec<Node<'a>>,
    /// Table of [`Region`]s.
    pub regions: Vec<Region<'a>>,
    /// Table of [`Term`]s.
    pub terms: Vec<Term<'a>>,
}

impl<'a> Module<'a> {
    /// Return the node data for a given node id.
    #[inline]
    #[must_use]
    pub fn get_node(&self, node_id: NodeId) -> Option<&Node<'a>> {
        self.nodes.get(node_id.index())
    }

    /// Return a mutable reference to the node data for a given node id.
    #[inline]
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut Node<'a>> {
        self.nodes.get_mut(node_id.index())
    }

    /// Insert a new node into the module and return its id.
    pub fn insert_node(&mut self, node: Node<'a>) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Return the term data for a given term id.
    ///
    /// Returns [`Term::Wildcard`] when the term id is invalid.
    #[inline]
    #[must_use]
    pub fn get_term(&self, term_id: TermId) -> Option<&Term<'a>> {
        if term_id.is_valid() {
            self.terms.get(term_id.index())
        } else {
            Some(&Term::Wildcard)
        }
    }

    /// Return a mutable reference to the term data for a given term id.
    #[inline]
    pub fn get_term_mut(&mut self, term_id: TermId) -> Option<&mut Term<'a>> {
        self.terms.get_mut(term_id.index())
    }

    /// Insert a new term into the module and return its id.
    pub fn insert_term(&mut self, term: Term<'a>) -> TermId {
        let id = TermId::new(self.terms.len());
        self.terms.push(term);
        id
    }

    /// Return the region data for a given region id.
    #[inline]
    #[must_use]
    pub fn get_region(&self, region_id: RegionId) -> Option<&Region<'a>> {
        self.regions.get(region_id.index())
    }

    /// Return a mutable reference to the region data for a given region id.
    #[inline]
    pub fn get_region_mut(&mut self, region_id: RegionId) -> Option<&mut Region<'a>> {
        self.regions.get_mut(region_id.index())
    }

    /// Insert a new region into the module and return its id.
    pub fn insert_region(&mut self, region: Region<'a>) -> RegionId {
        let id = RegionId::new(self.regions.len());
        self.regions.push(region);
        id
    }

    /// Attempt to view a part of this module via a [`View`] instance.
    pub fn view<S, V: View<'a, S>>(&'a self, src: S) -> Option<V> {
        V::view(self, src)
    }

    /// Convert the module to the [ast] representation.
    ///
    /// [ast]: crate::v0::ast
    #[must_use]
    pub fn as_ast(&self) -> Option<ast::Module> {
        let root = self.view(self.root)?;
        Some(ast::Module { root })
    }
}

/// Nodes in the hugr graph.
///
/// See [`ast::Node`] for the AST representation.
///
/// [`ast::Node`]: crate::v0::ast::Node
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Node<'a> {
    /// The operation that the node performs.
    pub operation: Operation<'a>,
    /// The input ports of the node.
    pub inputs: &'a [LinkIndex],
    /// The output ports of the node.
    pub outputs: &'a [LinkIndex],
    /// The regions of the node.
    pub regions: &'a [RegionId],
    /// The meta information attached to the node.
    pub meta: &'a [TermId],
    /// The signature of the node.
    ///
    /// Can be `None` to indicate that the node's signature should be inferred,
    /// or for nodes with operations that do not have a signature.
    pub signature: Option<TermId>,
}

/// Operations that nodes can perform.
///
/// See [`ast::Operation`] for the AST representation.
///
/// [`ast::Operation`]: crate::v0::ast::Operation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Operation<'a> {
    /// Invalid operation to be used as a placeholder.
    /// This is useful for modules that have non-contiguous node ids, or modules
    /// that have not yet been fully constructed.
    #[default]
    Invalid,
    /// Data flow graphs.
    Dfg,
    /// Control flow graphs.
    Cfg,
    /// Basic blocks in a control flow graph.
    Block,
    /// Function definitions.
    DefineFunc(&'a Symbol<'a>),
    /// Function declarations.
    DeclareFunc(&'a Symbol<'a>),
    /// Custom operation.
    Custom(TermId),
    /// Alias definitions.
    DefineAlias(&'a Symbol<'a>, TermId),
    /// Alias declarations.
    DeclareAlias(&'a Symbol<'a>),
    /// Tail controlled loop.
    /// Nodes with this operation contain a dataflow graph that is executed in a loop.
    /// The loop body is executed at least once, producing a result that indicates whether
    /// to continue the loop or return the result.
    ///
    /// # Port Types
    ///
    /// - **Inputs**: `inputs` + `rest`
    /// - **Outputs**: `outputs` + `rest`
    /// - **Sources**: `inputs` + `rest`
    /// - **Targets**: `(adt [inputs outputs])` + `rest`
    TailLoop,

    /// Conditional operation.
    ///
    /// # Port types
    ///
    /// - **Inputs**: `[(adt inputs)]` + `context`
    /// - **Outputs**: `outputs`
    Conditional,

    /// Declaration for a term constructor.
    ///
    /// Nodes with this operation must be within a module region.
    DeclareConstructor(&'a Symbol<'a>),

    /// Declaration for a operation.
    ///
    /// Nodes with this operation must be within a module region.
    DeclareOperation(&'a Symbol<'a>),

    /// Import a symbol.
    Import {
        /// The name of the symbol to be imported.
        name: &'a str,
    },
}

impl<'a> Operation<'a> {
    /// Returns the symbol introduced by the operation, if any.
    #[must_use]
    pub fn symbol(&self) -> Option<&'a str> {
        match self {
            Operation::DefineFunc(symbol) => Some(symbol.name),
            Operation::DeclareFunc(symbol) => Some(symbol.name),
            Operation::DefineAlias(symbol, _) => Some(symbol.name),
            Operation::DeclareAlias(symbol) => Some(symbol.name),
            Operation::DeclareConstructor(symbol) => Some(symbol.name),
            Operation::DeclareOperation(symbol) => Some(symbol.name),
            Operation::Import { name } => Some(name),
            _ => None,
        }
    }
}

/// A region in the hugr.
///
/// See [`ast::Region`] for the AST representation.
///
/// [`ast::Region`]: crate::v0::ast::Region
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Region<'a> {
    /// The kind of the region. See [`RegionKind`] for details.
    pub kind: RegionKind,
    /// The source ports of the region.
    pub sources: &'a [LinkIndex],
    /// The target ports of the region.
    pub targets: &'a [LinkIndex],
    /// The nodes in the region. The order of the nodes is not significant.
    pub children: &'a [NodeId],
    /// The metadata attached to the region.
    pub meta: &'a [TermId],
    /// The signature of the region.
    pub signature: Option<TermId>,
    /// Information about the scope defined by this region, if the region is closed.
    pub scope: Option<RegionScope>,
}

/// Information about the scope defined by a closed region.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegionScope {
    /// The number of links in the scope.
    pub links: u32,
    /// The number of ports in the scope.
    pub ports: u32,
}

/// A symbol.
///
/// See [`ast::Symbol`] for the AST representation.
///
/// [`ast::Symbol`]: crate::v0::ast::Symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol<'a> {
    /// The name of the symbol.
    pub name: &'a str,
    /// The static parameters.
    pub params: &'a [Param<'a>],
    /// The constraints on the static parameters.
    pub constraints: &'a [TermId],
    /// The signature of the symbol.
    pub signature: TermId,
}

/// An index of a variable within a node's parameter list.
pub type VarIndex = u16;

/// A term in the compile time meta language.
///
/// See [`ast::Term`] for the AST representation.
///
/// [`ast::Term`]: crate::v0::ast::Term
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Term<'a> {
    /// Standin for any term.
    #[default]
    Wildcard,

    /// A local variable.
    Var(VarId),

    /// Apply a symbol to a sequence of arguments.
    ///
    /// The symbol is defined by a node in the same graph. The type of this term
    /// is derived from instantiating the symbol's parameters in the symbol's
    /// signature.
    Apply(NodeId, &'a [TermId]),

    /// List of static data.
    ///
    /// Lists can include individual items or other lists to be spliced in.
    ///
    /// **Type:** `(core.list ?t)`
    List(&'a [SeqPart]),

    /// A static literal value.
    Literal(Literal),

    /// A constant anonymous function.
    ///
    /// **Type:** `(core.const (core.fn ?ins ?outs ?ext) (ext))`
    Func(RegionId),

    /// Tuple of static data.
    ///
    /// Tuples can include individual items or other tuples to be spliced in.
    ///
    /// **Type:** `(core.tuple ?types)`
    Tuple(&'a [SeqPart]),
}

impl From<Literal> for Term<'_> {
    fn from(value: Literal) -> Self {
        Self::Literal(value)
    }
}

/// A part of a list/tuple term.
///
/// See [`ast::SeqPart`] for the AST representation.
///
/// [`ast::SeqPart`]: crate::v0::ast::SeqPart
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeqPart {
    /// A single item.
    Item(TermId),
    /// A list to be spliced into the parent list/tuple.
    Splice(TermId),
}

/// A parameter to a function or alias.
///
/// Parameter names must be unique within a parameter list.
///
/// See [`ast::Param`] for the AST representation.
///
/// [`ast::Param`]: crate::v0::ast::Param
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Param<'a> {
    /// The name of the parameter.
    pub name: &'a str,
    /// The type of the parameter.
    pub r#type: TermId,
}

macro_rules! define_index {
    ($(#[$meta:meta])* $vis:vis struct $name:ident(pub u32);) => {
        #[repr(transparent)]
        $(#[$meta])*
        $vis struct $name(pub u32);

        impl $name {
            /// Create a new index.
            ///
            /// # Panics
            ///
            /// Panics if the index is 2^32 or larger.
            #[must_use] pub fn new(index: usize) -> Self {
                assert!(index < u32::MAX as usize, "index out of bounds");
                Self(index as u32)
            }

            /// Returns whether the index is valid.
            #[inline]
            #[must_use] pub fn is_valid(self) -> bool {
                self.0 < u32::MAX
            }

            /// Returns the index as a `usize` to conveniently use it as a slice index.
            #[inline]
            #[must_use] pub fn index(self) -> usize {
                self.0 as usize
            }

            /// Convert a slice of this index type into a slice of `u32`s.
            #[must_use] pub fn unwrap_slice(slice: &[Self]) -> &[u32] {
                // SAFETY: This type is just a newtype around `u32`.
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len()) }
            }

            /// Convert a slice of `u32`s into a slice of this index type.
            #[must_use] pub fn wrap_slice(slice: &[u32]) -> &[Self] {
                // SAFETY: This type is just a newtype around `u32`.
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(u32::MAX)
            }
        }
    };
}

define_index! {
    /// Id of a node in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct NodeId(pub u32);
}

define_index! {
    /// Index of a link in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct LinkIndex(pub u32);
}

define_index! {
    /// Id of a region in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct RegionId(pub u32);
}

define_index! {
    /// Id of a term in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct TermId(pub u32);
}

/// The id of a link consisting of its region and the link index.
#[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[display("{_0}#{_1}")]
pub struct LinkId(pub RegionId, pub LinkIndex);

/// The id of a variable consisting of its node and the variable index.
#[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[display("{_0}#{_1}")]
pub struct VarId(pub NodeId, pub VarIndex);

/// Errors that can occur when traversing and interpreting the model.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// There is a reference to a node that does not exist.
    #[error("node not found: {0}")]
    NodeNotFound(NodeId),
    /// There is a reference to a term that does not exist.
    #[error("term not found: {0}")]
    TermNotFound(TermId),
    /// There is a reference to a region that does not exist.
    #[error("region not found: {0}")]
    RegionNotFound(RegionId),
    /// Invalid variable reference.
    #[error("variable {0} invalid")]
    InvalidVar(VarId),
    /// Invalid symbol reference.
    #[error("symbol reference {0} invalid")]
    InvalidSymbol(NodeId),
    /// The model contains an operation in a place where it is not allowed.
    #[error("unexpected operation on node: {0}")]
    UnexpectedOperation(NodeId),
    /// There is a term that is not well-typed.
    #[error("type error in term: {0}")]
    TypeError(TermId),
    /// There is a node whose regions are not well-formed according to the node's operation.
    #[error("node has invalid regions: {0}")]
    InvalidRegions(NodeId),
    /// There is a name that is not well-formed.
    #[error("malformed name: {0}")]
    MalformedName(SmolStr),
    /// There is a condition node that lacks a case for a tag or
    /// defines two cases for the same tag.
    #[error("condition node is malformed: {0}")]
    MalformedCondition(NodeId),
    /// There is a node that is not well-formed or has the invalid operation.
    #[error("invalid operation on node: {0}")]
    InvalidOperation(NodeId),
}
