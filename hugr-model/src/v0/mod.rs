//! Version 0 (unstable).
//!
//! Instead of directly nesting `Node`s or `Term`s, we store them in tables and refer to them
//! by their index in the table. This allows us to attach additional data to nodes and terms
//! without changing the data structure itself. This can be used, for example, to keep track
//! of metadata that has been parsed from its generic reprensentation as a term into a more
//! specific in-memory reprensentation.
//!
//! The tabling is also used for deduplication of terms. In practice, many terms will share
//! the same subterms, and we can save memory and validation time by storing them only once.
//! However we allow non-deduplicated terms for cases in which terms carry additional identity
//! over just their structure. For instance, structurally identical terms could originate
//! from different locations in a text file and therefore should be treated differently when
//! locating type errors.
use smol_str::SmolStr;
use tinyvec::TinyVec;

/// Index of a node in a hugr graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeId(pub u32);

/// Index of a port in a hugr graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PortId(pub u32);

/// Index of a term in a hugr graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TermId(pub u32);

/// An identifier referring to types, terms, or functions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(pub SmolStr);

/// A local variable in terms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermVar(pub SmolStr);

/// The name of an edge.
///
/// This is to be used in the textual representation of the graph
/// to indicate that two ports are connected by assigning them the same
/// edge variable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeVar(pub SmolStr);

/// A module consisting of a hugr graph together with terms.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// Table of [`Node`]s.
    pub nodes: Vec<Node>,
    /// Table of [`Port`]s.
    pub ports: Vec<Port>,
    /// Table of [`Edge`]s.
    pub edges: Vec<Edge>,
    /// Table of [`Term`]s.
    pub terms: Vec<Term>,
}

/// Nodes in the hugr graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// The operation that the node performs.
    pub operation: Operation,
    /// Parameters that are passed to the operation.
    pub params: TinyVec<[TermId; 3]>,
    /// The input ports of the node.
    pub inputs: TinyVec<[PortId; 3]>,
    /// The output ports of the node.
    pub outputs: TinyVec<[PortId; 3]>,
    /// The children of the node.
    pub children: TinyVec<[NodeId; 3]>,
    /// Metadata attached to the node.
    pub meta: Vec<MetaItem>,
}

/// Operations that nodes can perform.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Root node of the hugr graph.
    Module,
    /// Internal inputs of the parent node.
    Input,
    /// Internal outputs of the parent node.
    Output,
    /// Data flow graphs.
    Dfg,
    /// Control flow graphs.
    Cfg,
    /// Basic blocks.
    Block,
    /// The exit node of a control flow graph.
    Exit,
    /// Cases in a conditional node.
    Case,
    /// Function definitions.
    DefineFunc(operation::DefineFunc),
    /// Function declarations.
    DeclareFunc(operation::DeclareFunc),
    /// Function calls.
    CallFunc(operation::CallFunc),
    /// Function loads.
    LoadFunc(operation::LoadFunc),
    /// Custom operations.
    Custom(operation::Custom),
    /// Alias definitions.
    DefineAlias(operation::DefineAlias),
    /// Alias declarations.
    DeclareAlias(operation::DeclareAlias),
}

/// The variants for [`Operation`].
pub mod operation {
    #[allow(unused_imports)]
    use super::Operation;
    use super::{Scheme, Symbol, TermId};

    /// See [`Operation::DefineFunc`].
    #[derive(Debug, Clone)]
    pub struct DefineFunc {
        /// The name of the function to be defined.
        pub name: Symbol,
        /// The type scheme of the function.
        pub r#type: Box<Scheme>,
    }

    /// See [`Operation::DeclareFunc`].
    #[derive(Debug, Clone)]
    pub struct DeclareFunc {
        /// The name of the function to be declared.
        pub name: Symbol,
        /// The type scheme of the function.
        pub r#type: Box<Scheme>,
    }

    /// See [`Operation::CallFunc`].
    #[derive(Debug, Clone)]
    pub struct CallFunc {
        /// The name of the function to be called.
        pub name: Symbol,
    }

    /// See [`Operation::LoadFunc`].
    #[derive(Debug, Clone)]
    pub struct LoadFunc {
        /// The name of the function to be loaded.
        pub name: Symbol,
    }

    /// See [`Operation::Custom`].
    #[derive(Debug, Clone)]
    pub struct Custom {
        /// The name of the custom operation.
        pub name: Symbol,
    }

    /// See [`Operation::DefineAlias`].
    #[derive(Debug, Clone)]
    pub struct DefineAlias {
        /// The name of the alias to be defined.
        pub name: Symbol,
        /// The value of the alias.
        pub value: TermId,
    }

    /// See [`Operation::DeclareAlias`].
    #[derive(Debug, Clone)]
    pub struct DeclareAlias {
        /// The name of the alias to be declared.
        pub name: Symbol,
        /// The type of the alias.
        pub r#type: TermId,
    }
}

/// A metadata item.
#[derive(Debug, Clone)]
pub struct MetaItem {
    /// Name of the metadata item.
    pub name: SmolStr,
    /// Value of the metadata item.
    pub value: Term,
}

/// A port in the graph.
#[derive(Debug, Clone)]
pub struct Port {
    /// The type of the port.
    ///
    /// This must be a term of type `Type`.
    /// If the type is unknown, this will be a wildcard term.
    pub r#type: TermId,

    /// Metadata attached to the port.
    pub meta: Vec<MetaItem>,
}

/// An edge in the graph.
///
/// An edge connects input ports to output ports.
/// A port may only be part of at most one edge.
#[derive(Debug, Clone, Default)]
pub struct Edge {
    /// The input ports of the edge.
    pub inputs: TinyVec<[PortId; 3]>,
    /// The output ports of the edge.
    pub outputs: TinyVec<[PortId; 3]>,
}

/// Parameterized terms.
#[derive(Debug, Clone)]
pub struct Scheme {
    /// The named parameters of the scheme.
    ///
    /// Within any term, the previous terms of the parameter list are available as variables.
    pub params: Vec<SchemeParam>,
    /// Constraints on the parameters of the scheme.
    ///
    /// All parameters are available as variables within the constraints.
    pub constraints: TinyVec<[TermId; 3]>,
    /// The body of the scheme.
    ///
    /// All parameters are available as variables within the body.
    pub body: TermId,
}

/// A named parameter of a scheme.
#[derive(Debug, Clone)]
pub struct SchemeParam {
    /// The name of the parameter.
    pub name: TermVar,
    /// The type of the parameter.
    pub r#type: TermId,
}

/// A term in the compile time meta language.
#[derive(Debug, Clone)]
pub enum Term {
    /// Standin for any term.
    Wildcard,
    /// The type of types.
    Type,
    /// The type of constraints.
    Constraint,
    /// A variable.
    Var(TermVar),
    /// A symbolic function application.
    Named(term::Named),
    /// A list, with an optional tail.
    List(term::List),
    /// The type of lists, given a type for the items.
    ListType(term::ListType),
    /// A string.
    Str(SmolStr),
    /// The type of strings.
    StrType,
    /// A natural number.
    Nat(u64),
    /// The type of natural numbers.
    NatType,
    /// Extension set.
    ExtSet(term::ExtSet),
    /// The type of extension sets.
    ExtSetType,
    /// A tuple of values.
    Tuple(term::Tuple),
    /// A product type, given a list of types for the fields.
    ProductType(term::ProductType),
    /// A variant of a sum type, given a tag and its value.
    Tagged(term::Tagged),
    /// A sum type, given a list of variants.
    SumType(term::SumType),
    /// The type of functions, given lists of input and output types and an extension set.
    FuncType(term::FuncType),
}

/// The variants for [`Term`].
pub mod term {
    use super::{Symbol, TermId};
    use smol_str::SmolStr;
    use tinyvec::TinyVec;

    /// Named terms.
    #[derive(Debug, Clone)]
    pub struct Named {
        /// The name of the term.
        pub name: Symbol,
        /// Arguments to the term.
        pub args: TinyVec<[TermId; 3]>,
    }

    /// A homogeneous list of terms.
    #[derive(Debug, Clone)]
    pub struct List {
        /// The items that are contained in the list.
        pub items: TinyVec<[TermId; 3]>,
        /// Optionally, a term that represents the remainder of the list.
        pub tail: Option<TermId>,
    }

    /// The type of a list of terms.
    #[derive(Debug, Clone)]
    pub struct ListType {
        /// The type of the items contained in the list.
        pub item_type: TermId,
    }

    /// A heterogeneous list of terms.
    #[derive(Debug, Clone)]
    pub struct Tuple {
        /// The items that are contained in the tuple.
        pub items: TinyVec<[TermId; 3]>,
    }

    /// A product type.
    #[derive(Debug, Clone)]
    pub struct ProductType {
        /// The types that are contained in the product type.
        pub types: TermId,
    }

    /// Function type.
    #[derive(Debug, Clone)]
    pub struct FuncType {
        /// The type of the inputs to the function.
        ///
        /// This must be a list of types.
        pub inputs: TermId,
        /// The type of the outputs of the function.
        ///
        /// This must be a list of types.
        pub outputs: TermId,
        /// The extensions that are required to run the function.
        ///
        /// This must be an extension set.
        pub extensions: TermId,
    }

    /// Sum type.
    #[derive(Debug, Clone)]
    pub struct SumType {
        /// The types of the variants in the sum.
        pub types: TermId,
    }

    /// Tagged term.
    #[derive(Debug, Clone)]
    pub struct Tagged {
        /// The tag of the tagged term.
        pub tag: u8,
        /// The term that is tagged.
        pub term: TermId,
    }

    /// Extension set.
    #[derive(Debug, Clone, Default)]
    pub struct ExtSet {
        /// The extensions that are contained in the extension set.
        pub extensions: Vec<SmolStr>,
        /// The rest of the extension set.
        pub rest: Option<TermId>,
    }
}
