//! Version 0 (unstable).
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
//!
//! # Terms
//!
//! Terms form a meta language that is used to describe types, parameters and metadata that
//! are known statically. To allow types to be parameterized by values, types and values
//! are treated uniformly as terms, enabling a restricted form of dependent typing.
//! The type system is extensible and can be used to declaratively encode the desired shape
//! of operation parameters and metadata. Type constraints can be used to express more complex
//! validation rules.
//!
//! # Tabling
//!
//! Instead of directly nesting structures, we store them in tables and refer to them
//! by their index in the table. This allows us to attach additional data to the structures
//! without changing the data structure itself. This can be used, for example, to keep track
//! of metadata that has been parsed from its generic representation as a term into a more
//! specific in-memory representation.
//!
//! The tabling is also used for deduplication of terms. In practice, many terms will share
//! the same subterms, and we can save memory and validation time by storing them only once.
//! However we allow non-deduplicated terms for cases in which terms carry additional identity
//! over just their structure. For instance, structurally identical terms could originate
//! from different locations in a text file and therefore should be treated differently when
//! locating type errors.
//!
//! # Plain Data
//!
//! All types in the hugr model are plain data. This means that they can be serialized and
//! deserialized without loss of information. This is important for the model to be able to
//! serve as a stable interchange format between different tools and versions of the library.
//!
//! # Arena Allocation
//!
//! Since we intend to use the model data structures as an intermediary to convert between
//! different representations (such as text, binary or in-memory), we use arena allocation
//! to efficiently allocate and free the parts of the data structure that isn't directly stored
//! in the tables. For that purpose, we use the `'a` lifetime parameter to indicate the
//! lifetime of the arena.
//!
//! # Remaining Mismatch with `hugr-core`
//!
//! This data model was designed to encode as much of `hugr-core` as possible while also
//! filling in conceptual gaps and providing a forward-compatible foundation for future
//! development. However, there are still some mismatches with `hugr-core` that are not
//! addressed by conversions in import/export:
//!
//! - Some static types can not yet be represented in `hugr-core` although they should be.
//! - `hugr-model` does not have constants for runtime types as `hugr-core` does ([#1425]).
//!   The rationale for this is that runtime values can not be represented except at runtime
//!   (they might e.g. be qubits or part of some extension in which values lack any semantics
//!   in forms of sets altogether). We might resolve this by introducing ways to use static
//!   values as "blueprints" for runtime values.
//! - The model does not have types with a copy bound as `hugr-core` does, and instead uses
//!   a more general form of type constraints ([#1556]). Similarly, the model does not have
//!   bounded naturals. In both cases, we import these types with the most permissive bound
//!   for now.
//! - The model allows nodes to have multiple child regions, including for custom operations.
//!   `hugr-core` does not support multiple regions, or any nesting for custom operations ([#1546]).
//! - `hugr-core` has rows with multiple row variables, which can be in arbitrary positions
//!   in the row. `hugr-core` rows correspond to lists in the model, and only support a single
//!   variable at the end. The same applies to extension sets ([#1556]).
//! - In a model module, ports are connected when they share the same link. This differs from
//!   the type of port connectivity in the graph data structure used by `hugr-core`. However,
//!   `hugr-core` restricts connectivity so that in any group of connected ports there is at
//!   most one output port (for dataflow) or at most one input port (for control flow). In
//!   these cases, there is no mismatch.
//! - `hugr-core` has no support for constraints and does not make a distinction between
//!   explicit and implicit parameters.
//! - `hugr-core` only allows to define type aliases, but not aliases for other terms.
//! - The model does not have a concept of "order edges". These ordering hints can be useful,
//!   but expressing them via the same mechanism as data and control flow might not be the
//!   correct approach.
//! - Both `hugr-model` and `hugr-core` support metadata, but they use different encodings.
//!   `hugr-core` encodes metadata as JSON objects, while `hugr-model` uses terms. Using
//!   terms has the advantage that metadata can be validated with the same type checking
//!   mechanism as the rest of the model ([#1553]).
//! - `hugr-model` have a root region that corresponds to a root `Module` in `hugr-core`.
//!   `hugr-core` however can have nodes with different operations as their root ([#1554]).
//!
//! [#1425]: https://github.com/CQCL/hugr/discussions/1425
//! [#1556]: https://github.com/CQCL/hugr/discussions/1556
//! [#1546]: https://github.com/CQCL/hugr/issues/1546
//! [#1553]: https://github.com/CQCL/hugr/issues/1553
//! [#1554]: https://github.com/CQCL/hugr/issues/1554
use smol_str::SmolStr;
use thiserror::Error;

/// Constructor for documentation metadata.
///
/// - **Parameter:** `?description : str`
/// - **Result:** `meta`
pub const CORE_META_DESCRIPTION: &str = "core.meta.description";

/// Constructor for JSON encoded metadata.
///
/// This is included in the model to allow for compatibility with `hugr-core`.
/// The intention is to deprecate this in the future in favor of metadata
/// expressed with custom constructors.
///
/// - **Parameter:** `?name : str`
/// - **Parameter:** `?json : str`
/// - **Result:** `meta`
pub const COMPAT_META_JSON: &str = "compat.meta-json";

/// Constructor for JSON encoded constants.
///
/// This is included in the model to allow for compatibility with `hugr-core`.
/// The intention is to deprecate this in the future in favor of constants
/// expressed with custom constructors.
///
/// - **Parameter:** `?type : type`
/// - **Parameter:** `?json : str`
/// - **Parameter:** `?exts : ext-set`
/// - **Result:** `(const ?type ?exts)`
pub const COMPAT_CONST_JSON: &str = "compat.const-json";

pub mod binary;
pub mod scope;
pub mod text;

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
            pub fn new(index: usize) -> Self {
                assert!(index < u32::MAX as usize, "index out of bounds");
                Self(index as u32)
            }

            /// Returns the index as a `usize` to conveniently use it as a slice index.
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }

            /// Convert a slice of this index type into a slice of `u32`s.
            pub fn unwrap_slice(slice: &[Self]) -> &[u32] {
                // SAFETY: This type is just a newtype around `u32`.
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len()) }
            }

            /// Convert a slice of `u32`s into a slice of this index type.
            pub fn wrap_slice(slice: &[u32]) -> &[Self] {
                // SAFETY: This type is just a newtype around `u32`.
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
            }
        }
    };
}

define_index! {
    /// Id of a node in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct NodeId(pub u32);
}

define_index! {
    /// Index of a link in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct LinkIndex(pub u32);
}

define_index! {
    /// Id of a region in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct RegionId(pub u32);
}

define_index! {
    /// Id of a term in a hugr graph.
    #[derive(Debug, derive_more::Display, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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

/// A module consisting of a hugr graph together with terms.
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
    #[inline]
    pub fn get_term(&self, term_id: TermId) -> Option<&Term<'a>> {
        self.terms.get(term_id.index())
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
}

/// Nodes in the hugr graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Node<'a> {
    /// The operation that the node performs.
    pub operation: Operation<'a>,
    /// The input ports of the node.
    pub inputs: &'a [LinkIndex],
    /// The output ports of the node.
    pub outputs: &'a [LinkIndex],
    /// The parameters of the node.
    pub params: &'a [TermId],
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
    /// Basic blocks.
    Block,
    /// Function definitions.
    DefineFunc {
        /// The declaration of the function to be defined.
        decl: &'a FuncDecl<'a>,
    },
    /// Function declarations.
    DeclareFunc {
        /// The function to be declared.
        decl: &'a FuncDecl<'a>,
    },
    /// Function calls.
    CallFunc {
        /// The function to be called.
        func: TermId,
    },
    /// Function constants.
    LoadFunc {
        /// The function to be loaded.
        func: TermId,
    },
    /// Custom operation.
    ///
    /// The node's parameters correspond to the explicit parameter of the custom operation,
    /// leaving out the implicit parameters. Once the declaration of the custom operation
    /// becomes known by resolving the reference, the node can be transformed into a [`Operation::CustomFull`]
    /// by inferring terms for the implicit parameters or at least filling them in with a wildcard term.
    Custom {
        /// The symbol of the custom operation.
        operation: NodeId,
    },
    /// Custom operation with full parameters.
    ///
    /// The node's parameters correspond to both the explicit and implicit parameters of the custom operation.
    /// Since this can be tedious to write, the [`Operation::Custom`] variant can be used to indicate that
    /// the implicit parameters should be inferred.
    CustomFull {
        /// The symbol of the custom operation.
        operation: NodeId,
    },
    /// Alias definitions.
    DefineAlias {
        /// The declaration of the alias to be defined.
        decl: &'a AliasDecl<'a>,
        /// The value of the alias.
        value: TermId,
    },

    /// Alias declarations.
    DeclareAlias {
        /// The alias to be declared.
        decl: &'a AliasDecl<'a>,
    },

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

    /// Create an ADT value from a sequence of inputs.
    Tag {
        /// The tag of the ADT value.
        tag: u16,
    },

    /// Declaration for a term constructor.
    ///
    /// Nodes with this operation must be within a module region.
    DeclareConstructor {
        /// The declaration of the constructor.
        decl: &'a ConstructorDecl<'a>,
    },

    /// Declaration for a operation.
    ///
    /// Nodes with this operation must be within a module region.
    DeclareOperation {
        /// The declaration of the operation.
        decl: &'a OperationDecl<'a>,
    },

    /// Import a symbol.
    Import {
        /// The name of the symbol to be imported.
        name: &'a str,
    },

    /// Create a constant value.
    Const {
        /// The term that describes how to construct the constant value.
        value: TermId,
    },
}

impl<'a> Operation<'a> {
    /// Returns the symbol introduced by the operation, if any.
    pub fn symbol(&self) -> Option<&'a str> {
        match self {
            Operation::DefineFunc { decl } => Some(decl.name),
            Operation::DeclareFunc { decl } => Some(decl.name),
            Operation::DefineAlias { decl, .. } => Some(decl.name),
            Operation::DeclareAlias { decl } => Some(decl.name),
            Operation::DeclareConstructor { decl } => Some(decl.name),
            Operation::DeclareOperation { decl } => Some(decl.name),
            Operation::Import { name } => Some(name),
            _ => None,
        }
    }
}

/// A region in the hugr.
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
    ///
    /// Can be `None` to indicate that the region signature should be inferred.
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

/// Type to indicate whether scopes are open or closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ScopeClosure {
    /// A scope that is open and therefore not isolated from its parent scope.
    #[default]
    Open,
    /// A scope that is closed and therefore isolated from its parent scope.
    Closed,
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

/// A function declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncDecl<'a> {
    /// The name of the function to be declared.
    pub name: &'a str,
    /// The static parameters of the function.
    pub params: &'a [Param<'a>],
    /// The constraints on the static parameters.
    pub constraints: &'a [TermId],
    /// The signature of the function.
    pub signature: TermId,
}

/// An alias declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AliasDecl<'a> {
    /// The name of the alias to be declared.
    pub name: &'a str,
    /// The static parameters of the alias.
    pub params: &'a [Param<'a>],
    /// The type of the alias.
    pub r#type: TermId,
}

/// A term constructor declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstructorDecl<'a> {
    /// The name of the constructor to be declared.
    pub name: &'a str,
    /// The static parameters of the constructor.
    pub params: &'a [Param<'a>],
    /// The constraints on the static parameters.
    pub constraints: &'a [TermId],
    /// The type of the constructed term.
    pub r#type: TermId,
}

/// An operation declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationDecl<'a> {
    /// The name of the operation to be declared.
    pub name: &'a str,
    /// The static parameters of the operation.
    pub params: &'a [Param<'a>],
    /// The constraints on the static parameters.
    pub constraints: &'a [TermId],
    /// The type of the operation. This must be a function type.
    pub r#type: TermId,
}

/// An index of a variable within a node's parameter list.
pub type VarIndex = u16;

/// A term in the compile time meta language.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Term<'a> {
    /// Standin for any term.
    #[default]
    Wildcard,

    /// The type of runtime types.
    ///
    /// `type : static`
    Type,

    /// The type of static types.
    ///
    /// `static : static`
    StaticType,

    /// The type of constraints.
    ///
    /// `constraint : static`
    Constraint,

    /// A local variable.
    Var(VarId),

    /// A symbolic function application.
    ///
    /// The arguments of this application cover only the explicit parameters of the referenced declaration,
    /// leaving out the implicit parameters. Once the type of the declaration is known, the implicit parameters
    /// can be inferred and the term replaced with [`Term::ApplyFull`].
    ///
    /// `(GLOBAL ARG-0 ... ARG-n)`
    Apply {
        /// Reference to the symbol to apply.
        symbol: NodeId,
        /// Arguments to the function, covering only the explicit parameters.
        args: &'a [TermId],
    },

    /// A symbolic function application with all arguments applied.
    ///
    /// The arguments to this application cover both the implicit and explicit parameters of the referenced declaration.
    /// Since this can be tedious to write out, only the explicit parameters can be provided via [`Term::Apply`].
    ///
    /// `(@GLOBAL ARG-0 ... ARG-n)`
    ApplyFull {
        /// Reference to the symbol to apply.
        symbol: NodeId,
        /// Arguments to the function, covering both implicit and explicit parameters.
        args: &'a [TermId],
    },

    /// Type for a constant runtime value.
    ///
    /// `(const T) : static` where `T : type`.
    Const {
        /// The runtime type of the constant value.
        ///
        /// **Type:** `type`
        r#type: TermId,
        /// The extension set required to be present in order to use the constant value.
        ///
        /// **Type:** `ext-set`
        extensions: TermId,
    },

    /// A list. May include individual items or other lists to be spliced in.
    List {
        /// The parts of the list.
        parts: &'a [ListPart],
    },

    /// The type of lists, given a type for the items.
    ///
    /// `(list T) : static` where `T : static`.
    ListType {
        /// The type of the items in the list.
        ///
        /// `item_type : static`
        item_type: TermId,
    },

    /// A literal string.
    ///
    /// `"STRING" : str`
    Str(&'a str),

    /// The type of literal strings.
    ///
    /// `str : static`
    StrType,

    /// A literal natural number.
    ///
    /// `N : nat`
    Nat(u64),

    /// The type of literal natural numbers.
    ///
    /// `nat : static`
    NatType,

    /// Extension set.
    ExtSet {
        /// The parts of the extension set.
        ///
        /// Since extension sets are unordered, the parts may occur in any order.
        parts: &'a [ExtSetPart<'a>],
    },

    /// The type of extension sets.
    ///
    /// `ext-set : static`
    ExtSetType,

    /// An algebraic data type.
    ///
    /// `(adt VARIANTS) : type` where `VARIANTS : (list (list type))`.
    Adt {
        /// List of variants in the algrebaic data type.
        /// Each of the variants is itself a list of runtime types.
        variants: TermId,
    },

    /// The type of functions, given lists of input and output types and an extension set.
    FuncType {
        /// The input types of the function, given as a list of runtime types.
        ///
        /// `inputs : (list type)`
        inputs: TermId,
        /// The output types of the function, given as a list of runtime types.
        ///
        /// `outputs : (list type)`
        outputs: TermId,
        /// The set of extensions that the function requires to be present in
        /// order to be called.
        ///
        /// `extensions : ext-set`
        extensions: TermId,
    },

    /// Control flow.
    ///
    /// `(ctrl VALUES) : ctrl` where `VALUES : (list type)`.
    Control {
        /// List of values.
        values: TermId,
    },

    /// Type of control flow edges.
    ///
    /// `ctrl : static`
    ControlType,

    /// Constraint that requires a runtime type to be copyable and discardable.
    NonLinearConstraint {
        /// The runtime type that must be copyable and discardable.
        term: TermId,
    },

    /// A constant anonymous function.
    ConstFunc {
        /// The body of the constant anonymous function.
        region: RegionId,
    },

    /// A constant value for an algebraic data type.
    ConstAdt {
        /// The tag of the variant.
        tag: u16,
        /// The values of the variant.
        values: TermId,
    },

    /// A literal byte string.
    Bytes {
        /// The data of the byte string.
        data: &'a [u8],
    },

    /// The type of byte strings.
    BytesType,

    /// The type of metadata.
    Meta,
}

/// A part of a list term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListPart {
    /// A single item.
    Item(TermId),
    /// A list to be spliced into the parent list.
    Splice(TermId),
}

/// A part of an extension set term.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtSetPart<'a> {
    /// An extension.
    Extension(&'a str),
    /// An extension set to be spliced into the parent extension set.
    Splice(TermId),
}

/// A parameter to a function or alias.
///
/// Parameter names must be unique within a parameter list.
/// Implicit and explicit parameters share a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Param<'a> {
    /// The name of the parameter.
    pub name: &'a str,
    /// The type of the parameter.
    pub r#type: TermId,
    /// The sort of the parameter (implicit or explicit).
    pub sort: ParamSort,
}

/// The sort of a parameter (implicit or explicit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParamSort {
    /// The parameter is implicit and should be inferred, unless a full application form is used
    /// (see [`Term::ApplyFull`] and [`Operation::CustomFull`]).
    Implicit,
    /// The parameter is explicit and should always be provided.
    Explicit,
}

/// Errors that can occur when traversing and interpreting the model.
#[derive(Debug, Clone, Error)]
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
