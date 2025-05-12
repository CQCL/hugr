//! Abstract syntax tree representation of hugr modules.
//!
//! This module defines the abstract syntax tree for hugr graphs. The AST types
//! implement [`Display`] and [`FromStr`] for pretty printing into and parsing
//! from the s-expression based hugr text form. This is useful for debugging and
//! writing hugr modules by hand. For a more performant serialization format, see
//! [binary] instead.
//!
//! The data types in this module logically mirror those of the [table]
//! representation, but are encoded differently. Instead of using ids, the AST
//! data structure is recursive and uses names for symbols, variables and links.
//! [`Term`]s, [`Node`]s and [`Region`]s can be referred to individually in this
//! form, whereas the table form requires them to be seen in the context of a
//! module. An AST module can be translated into a table module via [`Module::resolve`].
//! This representation makes different efficiency tradeoffs than the table
//! form by using standard heap based data structures instead of a bump
//! allocated arena. This is slower but considerably more ergonomic.
//!
//! [binary]: crate::v0::binary
//! [table]: crate::v0::table
//! [`Display`]: std::fmt::Display
//! [`FromStr`]: std::str::FromStr
use std::sync::Arc;

use bumpalo::Bump;

use super::table::{self};
use super::{LinkName, Literal, RegionKind, SymbolName, VarName};

mod parse;
mod print;
#[cfg(feature = "pyo3")]
mod python;
mod resolve;
mod view;

pub use parse::ParseError;
pub use resolve::ResolveError;

/// A package in the hugr AST.
///
/// See [`table::Package`] for the table representation.
///
/// [`table::Package`]: crate::v0::table::Package
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    /// The sequence of modules in the package.
    pub modules: Vec<Module>,
}

impl Package {
    /// Try to convert this package into the table representation by
    /// independently resolving its modules via [`Module::resolve`].
    pub fn resolve<'a>(
        &'a self,
        bump: &'a Bump,
    ) -> Result<table::Package<'a>, resolve::ResolveError> {
        let modules = self
            .modules
            .iter()
            .map(|module| module.resolve(bump))
            .collect::<Result<_, _>>()?;
        Ok(table::Package { modules })
    }
}

/// A module in the hugr AST.
///
/// See [`table::Module`] for the table representation.
///
/// [`table::Module`]: crate::v0::table::Module
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    /// The root region of the module.
    ///
    /// This must be a region of kind [`RegionKind::Module`].
    pub root: Region,
}

impl Module {
    /// Try to convert this module into the table representation.
    ///
    /// This conversion resolves the names of variables, symbols and links
    /// according to their scoping rules described in the [`scope`] module.
    /// Whenever a symbol is used but not defined in scope, an import node will
    /// be inserted into the module region and all references to that symbol will
    /// refer to the import node. This gives the opportunity to link the missing symbols.
    ///
    /// [`scope`]: crate::v0::scope
    pub fn resolve<'a>(
        &'a self,
        bump: &'a Bump,
    ) -> Result<table::Module<'a>, resolve::ResolveError> {
        let mut ctx = resolve::Context::new(bump);
        ctx.resolve_module(self)?;
        Ok(ctx.finish())
    }
}

/// A node in the hugr AST.
///
/// See [`table::Node`] for the table representation.
///
/// [`table::Node`]: crate::v0::table::Node
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Node {
    /// The operation that the node performs.
    pub operation: Operation,

    /// The names of the links connected to the input ports of the node.
    pub inputs: Box<[LinkName]>,

    /// The names of the links connected to the output ports of the node.
    pub outputs: Box<[LinkName]>,

    /// The regions in the node.
    pub regions: Box<[Region]>,

    /// Metadata attached to the node.
    pub meta: Box<[Term]>,

    /// The input/output signature of the node.
    pub signature: Option<Term>,
}

/// The operation of a [`Node`] in the hugr AST.
///
/// See [`table::Operation`] for the table representation.
///
/// [`table::Operation`]: crate::v0::table::Operation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Operation {
    /// Invalid operation to be used as a placeholder.
    #[default]
    Invalid,
    /// Data flow graphs.
    Dfg,
    /// Control flow graphs.
    Cfg,
    /// Basic blocks in a control flow graph.
    Block,
    /// Function definitions.
    DefineFunc(Box<Symbol>),
    /// Function declarations.
    DeclareFunc(Box<Symbol>),
    /// Custom operations.
    Custom(Term),
    /// Alias definitions.
    DefineAlias(Box<Symbol>, Term),
    /// Alias declarations.
    DeclareAlias(Box<Symbol>),
    /// Tail controlled loops.
    TailLoop,
    /// Conditional operations.
    Conditional,
    /// Constructor declarations.
    DeclareConstructor(Box<Symbol>),
    /// Operation declarations.
    DeclareOperation(Box<Symbol>),
    /// Symbol imports.
    Import(SymbolName),
}

impl Operation {
    /// The name of the symbol introduced by this operation, if any.
    #[must_use]
    pub fn symbol_name(&self) -> Option<&SymbolName> {
        if let Operation::Import(symbol_name) = self {
            Some(symbol_name)
        } else {
            Some(&self.symbol()?.name)
        }
    }

    /// The symbol declared or defined by this operation, if any.
    #[must_use]
    pub fn symbol(&self) -> Option<&Symbol> {
        match self {
            Operation::DefineFunc(symbol)
            | Operation::DeclareFunc(symbol)
            | Operation::DefineAlias(symbol, _)
            | Operation::DeclareAlias(symbol)
            | Operation::DeclareConstructor(symbol)
            | Operation::DeclareOperation(symbol) => Some(symbol),
            _ => None,
        }
    }
}

/// A symbol declaration in the hugr AST.
///
/// See [`table::Symbol`] for the table representation.
///
/// [`table::Symbol`]: crate::v0::table::Symbol
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// The name of the symbol.
    pub name: SymbolName,
    /// The parameters of the symbol.
    pub params: Box<[Param]>,
    /// Constraints that the symbol imposes on the parameters.j
    pub constraints: Box<[Term]>,
    /// The type of the terms produced when the symbol is applied.
    pub signature: Term,
}

/// A parameter of a [`Symbol`] in the hugr AST.
///
/// See [`table::Param`] for the table representation.
///
/// [`table::Param`]: crate::v0::table::Param
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// The name of the parameter.
    pub name: VarName,
    /// The type of the parameter.
    pub r#type: Term,
}

/// A region in the hugr AST.
///
/// See [`table::Region`] for the table representation.
///
/// [`table::Region`]: crate::v0::table::Region
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Region {
    /// The kind of the region. See [`RegionKind`] for details.
    pub kind: RegionKind,
    /// The names of the links connected to the source ports of the region.
    pub sources: Box<[LinkName]>,
    /// The names of the links connected to the target ports of the region.
    pub targets: Box<[LinkName]>,
    /// The nodes in the region. The order of the nodes is not significant.
    pub children: Box<[Node]>,
    /// The metadata attached to the region.
    pub meta: Box<[Term]>,
    /// The source/target signature of the region.
    pub signature: Option<Term>,
}

/// A term in the hugr AST.
///
/// To facilitate sharing where possible, terms in the AST use reference
/// counting. This makes it cheap to clone and share terms.
///
/// See [`table::Term`] for the table representation.
///
/// [`table::Term`]: crate::v0::table::Term
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Term {
    /// Standin for any term.
    #[default]
    Wildcard,
    /// Local variable, identified by its name.
    Var(VarName),
    /// Symbol application.
    Apply(SymbolName, Arc<[Term]>),
    /// List of static data.
    List(Arc<[SeqPart]>),
    /// Static literal value.
    Literal(Literal),
    /// Tuple of static data.
    Tuple(Arc<[SeqPart]>),
    /// Function constant.
    Func(Arc<Region>),
}

impl From<Literal> for Term {
    fn from(value: Literal) -> Self {
        Self::Literal(value)
    }
}

/// A part of a tuple/list [`Term`] in the hugr AST.
///
/// See [`table::SeqPart`] for the table representation.
///
/// [`table::SeqPart`]: crate::v0::table::SeqPart
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqPart {
    /// An individual item in the sequence.
    Item(Term),
    /// A sequence to be spliced into this sequence.
    Splice(Term),
}
