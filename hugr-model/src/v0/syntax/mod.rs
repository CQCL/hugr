use std::sync::Arc;

use ordered_float::OrderedFloat;
use smol_str::SmolStr;

use super::{LinkId, LinkIndex, RegionKind, ScopeClosure};

mod build;
mod parse;
mod print;
mod view;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub root: Region,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub operation: Operation,
    pub inputs: Box<[LinkName]>,
    pub outputs: Box<[LinkName]>,
    pub regions: Box<[Region]>,
    pub meta: Box<[Term]>,
    pub signature: Option<Term>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    Invalid,
    Dfg,
    Cfg,
    Block,
    DefineFunc(Arc<Symbol>),
    DeclareFunc(Arc<Symbol>),
    Custom(Term),
    DefineAlias(Arc<Symbol>, Term),
    DeclareAlias(Arc<Symbol>),
    TailLoop,
    Conditional,
    DeclareConstructor(Arc<Symbol>),
    DeclareOperation(Arc<Symbol>),
    Import(SymbolName),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub name: SymbolName,
    pub params: Arc<[Param]>,
    pub constraints: Arc<[Term]>,
    pub signature: Term,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: VarName,
    pub r#type: Term,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Region {
    pub kind: RegionKind,
    pub sources: Box<[LinkName]>,
    pub targets: Box<[LinkName]>,
    pub children: Box<[Node]>,
    pub meta: Box<[Term]>,
    pub signature: Option<Term>,
    pub scope: ScopeClosure,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Term {
    #[default]
    Wildcard,
    Var(VarName),
    Apply(SymbolName, Arc<[Term]>),
    List(Arc<[SeqPart]>),
    Str(SmolStr),
    Nat(u64),
    Bytes(Arc<[u8]>),
    Float(OrderedFloat<f64>),
    Tuple(Arc<[SeqPart]>),
    Func(Arc<Region>),
    ExtSet,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarName(SmolStr);

impl VarName {
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self(name.into())
    }
}

impl AsRef<str> for VarName {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymbolName(SmolStr);

impl SymbolName {
    pub fn new(name: impl Into<SmolStr>) -> Self {
        Self(name.into())
    }
}

impl AsRef<str> for SymbolName {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LinkName(SmolStr);

impl LinkName {
    pub fn new_index(index: LinkIndex) -> Self {
        Self(format!("{}", index).into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeqPart {
    Item(Term),
    Splice(Term),
}
