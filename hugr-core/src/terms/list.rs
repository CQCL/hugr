use hugr_model::v0::ast;
use itertools::Itertools as _;
use servo_arc::{Arc, ThinArc};
use std::{fmt::Display, hash::Hash};

use super::{SeqPart, Term, Typed, ViewError, views::CoreList};

/// Homogeneous sequences of [`Term`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct List(ThinArc<ListHeader, SeqPart>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct ListHeader {
    list_type: CoreList,
}

impl List {
    pub fn new<I>(parts: I, item_type: Term) -> Self
    where
        I: IntoIterator<Item = SeqPart>,
        I::IntoIter: ExactSizeIterator,
    {
        Self(ThinArc::from_header_and_iter(
            ListHeader {
                list_type: CoreList { item_type },
            },
            parts.into_iter(),
        ))
    }

    pub fn try_new<I, E>(parts: I, item_type: Term) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<SeqPart, E>>,
        I::IntoIter: ExactSizeIterator,
    {
        let parts = parts.into_iter();

        let mut arc = ThinArc::from_header_and_iter(
            ListHeader {
                list_type: CoreList { item_type },
            },
            (0..parts.len()).map(|_| SeqPart::Item(Term::Wildcard)),
        );

        let part_slots = Arc::get_mut(&mut arc).unwrap().slice_mut().iter_mut();

        for (part_slot, part) in part_slots.zip_eq(parts) {
            *part_slot = part?;
        }

        Ok(Self(arc))
    }

    pub fn parts(&self) -> &[SeqPart] {
        self.0.slice()
    }

    pub fn item_type(&self) -> &Term {
        &self.type_().item_type
    }
}

impl Hash for List {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        todo!()
    }
}

impl Typed for List {
    #[allow(refining_impl_trait)]
    fn type_(&self) -> &CoreList {
        &self.0.header.list_type
    }
}

impl From<&List> for ast::Term {
    fn from(value: &List) -> Self {
        ast::Term::List(value.parts().iter().map(ast::SeqPart::from).collect())
    }
}

impl Display for List {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ast::Term::from(self))
    }
}

impl TryFrom<Term> for List {
    type Error = ViewError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            Term::List(list) => Ok(list),
            Term::Var(var) => {
                let type_ = var.type_().clone();
                Ok(List::new([SeqPart::Splice(Term::Var(var))], type_))
            }
            _ => Err(ViewError::Mismatch),
        }
    }
}
