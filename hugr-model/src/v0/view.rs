//! Helper functions for viewing parts of a module as specific types.
use ordered_float::OrderedFloat;

use super::{Module, NodeId, Operation, Term, TermId};

/// Trait for viewing parts of a [`Module`] as a specific type.
pub trait View<'a>: Sized {
    /// The identifier that references the structure within the module.
    type Id;

    /// Attempt to view the structure with the given identifier.
    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self>;
}

impl<'a> View<'a> for &'a str {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        match module.get_term(id)? {
            Term::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl<'a> View<'a> for &'a [u8] {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        match module.get_term(id)? {
            Term::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

impl<'a> View<'a> for OrderedFloat<f64> {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        match module.get_term(id)? {
            Term::Float(f) => Some(*f),
            _ => None,
        }
    }
}

impl<'a> View<'a> for f64 {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        match module.get_term(id)? {
            Term::Float(f) => Some(f.into_inner()),
            _ => None,
        }
    }
}

impl<'a> View<'a> for u64 {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        match module.get_term(id)? {
            Term::Nat(nat) => Some(*nat),
            _ => None,
        }
    }
}

/// View for a term that is an application of a named constructor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamedConstructor<'a> {
    /// The name of the applied constructor.
    pub name: &'a str,
    /// The parameters to the constructor.
    pub args: &'a [TermId],
}

impl<'a> View<'a> for NamedConstructor<'a> {
    type Id = TermId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        let Term::Apply(symbol_node, args) = module.get_term(id)? else {
            return None;
        };
        let symbol = module.get_node(*symbol_node)?.operation.symbol()?;
        Some(NamedConstructor { name: symbol, args })
    }
}

/// View for a node with a named custom operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamedOperation<'a> {
    /// The name of the operation.
    pub name: &'a str,
    /// The parameters passed to the operation.
    pub params: &'a [TermId],
}

impl<'a> View<'a> for NamedOperation<'a> {
    type Id = NodeId;

    fn view(module: &'a Module<'a>, id: Self::Id) -> Option<Self> {
        let node = module.get_node(id)?;
        let Operation::Custom(symbol_node) = &node.operation else {
            return None;
        };
        let name = module.get_node(*symbol_node)?.operation.symbol()?;
        let params = node.params;
        Some(NamedOperation { name, params })
    }
}
