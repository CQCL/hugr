//! Version 0 (unstable).
//!
//! **Warning**: This module is still under development and is expected to change.
//! It is included in the library to allow for early experimentation, and for
//! the core and model to converge incrementally.
//!
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
use smol_str::SmolStr;
use thiserror::Error;

pub mod text;

macro_rules! define_index {
    ($(#[$meta:meta])* $vis:vis struct $name:ident;) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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
    /// Index of a node in a hugr graph.
    pub struct NodeId;
}

define_index! {
    /// Index of a link in a hugr graph.
    pub struct LinkId;
}

define_index! {
    /// Index of a region in a hugr graph.
    pub struct RegionId;
}

define_index! {
    /// Index of a term in a hugr graph.
    pub struct TermId;
}

/// A module consisting of a hugr graph together with terms.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Module<'a> {
    /// The id of the root region.
    pub root: RegionId,
    /// Table of [`Node`]s.
    pub nodes: Vec<Node<'a>>,
    /// Table of [`Region`]s.
    pub region: Vec<Region<'a>>,
    /// Table of [`Term`]s.
    pub terms: Vec<Term<'a>>,
}

impl<'a> Module<'a> {
    /// Return the node data for a given node id.
    #[inline]
    pub fn get_node(&self, node_id: NodeId) -> Option<&Node<'a>> {
        self.nodes.get(node_id.0 as usize)
    }

    /// Insert a new node into the module and return its id.
    pub fn insert_node(&mut self, node: Node<'a>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Return the term data for a given term id.
    #[inline]
    pub fn get_term(&self, term_id: TermId) -> Option<&Term<'a>> {
        self.terms.get(term_id.0 as usize)
    }

    /// Insert a new term into the module and return its id.
    pub fn insert_term(&mut self, term: Term<'a>) -> TermId {
        let id = TermId(self.terms.len() as u32);
        self.terms.push(term);
        id
    }

    /// Return the region data for a given region id.
    #[inline]
    pub fn get_region(&self, region_id: RegionId) -> Option<&Region<'a>> {
        self.region.get(region_id.0 as usize)
    }

    /// Insert a new region into the module and return its id.
    pub fn insert_region(&mut self, region: Region<'a>) -> RegionId {
        let id = RegionId(self.region.len() as u32);
        self.region.push(region);
        id
    }
}

/// Nodes in the hugr graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node<'a> {
    /// The operation that the node performs.
    pub operation: Operation<'a>,
    /// The input ports of the node.
    pub inputs: &'a [Port<'a>],
    /// The output ports of the node.
    pub outputs: &'a [Port<'a>],
    /// The parameters of the node.
    pub params: &'a [TermId],
    /// The regions of the node.
    pub regions: &'a [RegionId],
    /// The meta information attached to the node.
    pub meta: &'a [MetaItem<'a>],
}

/// Operations that nodes can perform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation<'a> {
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
    /// The implicit parameters of the operation are left out.
    Custom {
        /// The name of the custom operation.
        name: GlobalRef<'a>,
    },
    /// Custom operation.
    ///
    /// The implicit parameters of the operation are included.
    CustomFull {
        /// The name of the custom operation.
        name: GlobalRef<'a>,
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
    TailLoop {
        // TODO: These can be determined by the port types?
        /// Types of the values that are passed as inputs to the loop, and are returned
        /// by the loop body when the loop is continued.
        ///
        /// **Type**: `(list type)`
        inputs: TermId,
        /// Types of the values that are produced at the end of the loop body when the loop
        /// should be ended.
        ///
        /// **Type**: `(list type)`
        outputs: TermId,
        /// Types of the values that are passed as inputs to the loop, to each iteration and
        /// are then returned at the end of the loop.
        ///
        /// **Type**: `(list type)`
        rest: TermId,
        ///
        ///
        /// **Type**: `ext-set`
        extensions: TermId,
    },

    /// Conditional operation.
    ///
    /// # Port types
    ///
    /// - **Inputs**: `[(adt inputs)]` + `context`
    /// - **Outputs**: `outputs`
    Conditional {
        /// Port types for each case of the conditional.
        ///
        /// **Type**: `(list (list type))`
        cases: TermId,
        /// Port types for additional inputs to the conditional.
        ///
        /// **Type**: `(list type)`
        context: TermId,
        /// Port types for the outputs of each case.
        ///
        /// **Type**: `(list type)`
        outputs: TermId,
        ///
        ///
        /// **Type**: `ext-set`
        extensions: TermId,
    },
}

/// A region in the hugr.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Region<'a> {
    /// The kind of the region. See [`RegionKind`] for details.
    pub kind: RegionKind,
    /// The source ports of the region.
    pub sources: &'a [Port<'a>],
    /// The target ports of the region.
    pub targets: &'a [Port<'a>],
    /// The nodes in the region. The order of the nodes is not significant.
    pub children: &'a [NodeId],
    /// The metadata attached to the region.
    pub meta: &'a [MetaItem<'a>],
}

/// The kind of a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegionKind {
    /// Data flow region.
    DataFlow,
    /// Control flow region.
    ControlFlow,
}

/// A port attached to a [`Node`] or [`Region`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Port<'a> {
    /// The link that the port is connected to.
    pub link: LinkRef<'a>,
    /// The type of the port.
    pub r#type: Option<TermId>,
    /// Metadata attached to the port.
    pub meta: &'a [MetaItem<'a>],
}

/// A function declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncDecl<'a> {
    /// The name of the function to be declared.
    pub name: &'a str,
    /// The static parameters of the function.
    pub params: &'a [Param<'a>],
    /// The type of the function.
    pub func: TermId,
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

/// A metadata item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetaItem<'a> {
    /// Name of the metadata item.
    pub name: &'a str,
    /// Value of the metadata item.
    pub value: TermId,
}

/// A reference to a global variable.
///
/// Global variables are defined in nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GlobalRef<'a> {
    /// Reference to the global that is defined by the given node.
    Direct(NodeId),
    /// Reference to the global with the given name.
    Named(&'a str),
}

impl std::fmt::Display for GlobalRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobalRef::Direct(id) => write!(f, ":{}", id.index()),
            GlobalRef::Named(name) => write!(f, "{}", name),
        }
    }
}

/// A reference to a local variable.
///
/// Local variables are defined as parameters to nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LocalRef<'a> {
    /// Reference to the local variable by its parameter index.
    Index(u16),
    /// Reference to the local variable by its name.
    Named(&'a str),
}

impl std::fmt::Display for LocalRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalRef::Index(index) => write!(f, "?:{}", index),
            LocalRef::Named(name) => write!(f, "?{}", name),
        }
    }
}

/// A reference to a link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LinkRef<'a> {
    /// Reference to the link by its id.
    Id(LinkId),
    /// Reference to the link by its name.
    Named(&'a str),
}

impl std::fmt::Display for LinkRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkRef::Id(id) => write!(f, "%:{})", id.index()),
            LinkRef::Named(name) => write!(f, "%{}", name),
        }
    }
}

/// A term in the compile time meta language.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term<'a> {
    /// Standin for any term.
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
    Var(LocalRef<'a>),

    /// A symbolic function application.
    ///
    /// `(GLOBAL ARG-0 ... ARG-n)`
    Apply {
        // TODO: Should the name be replaced with the id of the node that defines
        // the function to be applied? This could be a type, alias or function.
        /// The name of the term.
        name: GlobalRef<'a>,
        /// Arguments to the function, covering only the explicit parameters.
        args: &'a [TermId],
    },

    /// A symbolic function application with all arguments applied.
    ///
    /// `(@GLOBAL ARG-0 ... ARG-n)`
    ApplyFull {
        /// The name of the function to apply.
        name: GlobalRef<'a>,
        /// Arguments to the function, covering both implicit and explicit parameters.
        args: &'a [TermId],
    },

    /// Quote a runtime type as a static type.
    ///
    /// `(quote T) : static` where `T : type`.
    Quote {
        /// The runtime type to be quoted.
        ///
        /// **Type:** `type`
        r#type: TermId,
    },

    /// A list, with an optional tail.
    ///
    /// - `[ITEM-0 ... ITEM-n] : (list T)` where `T : static`, `ITEM-i : T`.
    /// - `[ITEM-0 ... ITEM-n . TAIL] : (list item-type)` where `T : static`, `ITEM-i : T`, `TAIL : (list T)`.
    List {
        /// The items in the list.
        ///
        /// `item-i : item-type`
        items: &'a [TermId],
        /// The tail of the list.
        ///
        /// `tail : (list item-type)`
        tail: Option<TermId>,
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
    Str(SmolStr),

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
    ///
    /// - `(ext EXT-0 ... EXT-n) : ext-set`
    /// - `(ext EXT-0 ... EXT-n . REST) : ext-set` where `REST : ext-set`.
    ExtSet {
        /// The items in the extension set.
        extensions: &'a [&'a str],
        /// The rest of the extension set.
        rest: Option<TermId>,
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
}

impl<'a> Default for Term<'a> {
    fn default() -> Self {
        Self::Wildcard
    }
}

/// A parameter to a function or alias.
///
/// Parameter names must be unique within a parameter list.
/// Implicit and explicit parameters share a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Param<'a> {
    /// An implicit parameter that should be inferred.
    Implicit {
        /// The name of the parameter.
        name: &'a str,
        /// The type of the parameter.
        ///
        /// This must be a term of type `static`.
        r#type: TermId,
    },
    /// An explicit parameter that should always be provided.
    Explicit {
        /// The name of the parameter.
        name: &'a str,
        /// The type of the parameter.
        ///
        /// This must be a term of type `static`.
        r#type: TermId,
    },
    /// A constraint that should be satisfied by other parameters in a parameter list.
    Constraint {
        /// The constraint to be satisfied.
        ///
        /// This must be a term of type `constraint`.
        constraint: TermId,
    },
}

/// Errors that can occur when traversing and interpreting the model.
#[derive(Debug, Clone, Error)]
pub enum ModelError {
    /// There is a reference to a node that does not exist.
    #[error("node not found: {0:?}")]
    NodeNotFound(NodeId),
    /// There is a reference to a term that does not exist.
    #[error("term not found: {0:?}")]
    TermNotFound(TermId),
    /// There is a reference to a region that does not exist.
    #[error("region not found: {0:?}")]
    RegionNotFound(RegionId),
    /// There is a local reference that does not resolve.
    #[error("local variable invalid: {0:?}")]
    InvalidLocal(String),
    /// There is a global reference that does not resolve to a node
    /// that defines a global variable.
    #[error("global variable invalid: {0:?}")]
    InvalidGlobal(String),
    /// The model contains an operation in a place where it is not allowed.
    #[error("unexpected operation on node: {0:?}")]
    UnexpectedOperation(NodeId),
    /// There is a term that is not well-typed.
    #[error("type error in term: {0:?}")]
    TypeError(TermId),
    /// There is a node whose regions are not well-formed according to the node's operation.
    #[error("node has invalid regions: {0:?}")]
    InvalidRegions(NodeId),
    /// There is a name that is not well-formed.
    #[error("malformed name: {0}")]
    MalformedName(SmolStr),
    /// There is a condition node that lacks a case for a tag or
    /// defines two cases for the same tag.
    #[error("condition node is malformed: {0:?}")]
    MalformedCondition(NodeId),
}
