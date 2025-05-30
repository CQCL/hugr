//! Utility types to view and construct terms.
use std::iter::FusedIterator;

use hugr_model::v0::ast;

use super::{Term, TermKind, View, ViewError, core};

/// The signature of a tail loop region.
#[derive(Debug, Clone)]
pub struct LoopRegionSignature {
    /// Prefix of input types.
    pub just_inputs: Term,
    /// Prefix of output types.
    pub just_outputs: Term,
    /// Suffix of types shared between inputs and outputs.
    pub rest: Term,
}

impl View for LoopRegionSignature {
    fn view(term: &Term) -> Result<Self, ViewError> {
        let core::Fn { outputs, .. } = term.view()?;
        let outputs = ListPrefix::view(&outputs)?;
        let [adt] = outputs.prefix;
        let core::Adt { variants } = adt.view()?;
        let variants = ListExact::view(&variants)?;
        let [just_inputs, just_outputs] = variants.items;
        Ok(Self {
            just_inputs,
            just_outputs,
            rest: outputs.rest,
        })
    }
}

impl From<LoopRegionSignature> for Term {
    fn from(value: LoopRegionSignature) -> Self {
        let variants = Term::from(ListExact {
            item_type: Term::default(),
            items: [value.just_inputs.clone(), value.just_outputs.clone()],
        });
        let adt = Term::from(core::Adt { variants });
        let inputs = Term::new(TermKind::ListConcat(
            &Term::default(),
            &[value.just_inputs.clone(), value.rest.clone()],
        ));
        let outputs = Term::from(ListPrefix {
            item_type: Term::default(),
            prefix: [adt],
            rest: value.rest.clone(),
        });
        core::Fn { inputs, outputs }.into()
    }
}

/// Utility view for lists with a known-length prefix.
#[derive(Debug, Clone)]
pub struct ListPrefix<const N: usize> {
    /// The type of the items in the list.
    pub item_type: Term,
    /// The prefix of the list.
    pub prefix: [Term; N],
    /// The remainder of the list.
    pub rest: Term,
}

impl<const N: usize> From<ListPrefix<N>> for Term {
    fn from(value: ListPrefix<N>) -> Self {
        let prefix = Term::from(ListExact {
            item_type: value.item_type.clone(),
            items: value.prefix,
        });
        Term::new(TermKind::ListConcat(
            &value.item_type,
            &[prefix, value.rest],
        ))
    }
}

impl<const N: usize> View for ListPrefix<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        let mut parts = ListIter::new(term);
        let item_type = parts.item_type().clone();
        let mut prefix: [Term; N] = std::array::from_fn(|_| Term::default());
        let mut index = 0;

        for part in &mut parts {
            match part {
                ListPart::Item(term) => {
                    let item = match prefix.get_mut(index) {
                        Some(item) => item,
                        None => break,
                    };
                    *item = term;
                    index += 1;
                }
                ListPart::Splice(term) => match term.get() {
                    TermKind::Wildcard => return Err(ViewError::Uninferred),
                    TermKind::Var(_) => return Err(ViewError::Variable),
                    _ => return Err(ViewError::Mismatch),
                },
            }
        }

        let rest = parts.remaining();

        Ok(Self {
            item_type,
            prefix,
            rest,
        })
    }
}

/// Utility view for closed lists with a known length.
#[derive(Debug, Clone)]
pub struct ListExact<const N: usize> {
    /// The type of the items in the list.
    pub item_type: Term,
    /// The items in the list.
    pub items: [Term; N],
}

impl<const N: usize> From<ListExact<N>> for Term {
    fn from(value: ListExact<N>) -> Self {
        Term::new(TermKind::List(&value.item_type, &value.items))
    }
}

impl<const N: usize> View for ListExact<N> {
    fn view(term: &Term) -> Result<Self, ViewError> {
        let parts = ListIter::new(term);
        let item_type = parts.item_type().clone();
        let mut items: [Term; N] = std::array::from_fn(|_| Term::default());
        let mut index = 0;

        for part in parts {
            match part {
                ListPart::Item(term) => {
                    let item = items.get_mut(index).ok_or(ViewError::Mismatch)?;
                    *item = term;
                    index += 1;
                }
                ListPart::Splice(term) => match term.get() {
                    TermKind::Wildcard => return Err(ViewError::Uninferred),
                    TermKind::Var(_) => return Err(ViewError::Variable),
                    _ => return Err(ViewError::Mismatch),
                },
            }
        }

        Ok(Self { item_type, items })
    }
}

#[derive(Debug, Clone)]
pub enum ListPart {
    Item(Term),
    Splice(Term),
}

impl From<ListPart> for ast::SeqPart {
    fn from(value: ListPart) -> Self {
        match value {
            ListPart::Item(term) => Self::Item(term.into()),
            ListPart::Splice(term) => Self::Splice(term.into()),
        }
    }
}

/// Iterator over the [`ListPart`]s of a list term.
pub struct ListIter {
    items: Vec<Term>,
    stack: Vec<Term>,
    item_type: Term,
}

// TODO: It might be better for performance if `ListIter` allowed for
// some inline capacity in the `items` and `stack` fields to avoid
// allocating in common cases.

impl ListIter {
    /// Create a new iterator from the given term.
    pub fn new(term: &Term) -> Self {
        match term.get() {
            TermKind::List(item_type, items) => Self {
                items: items.iter().rev().cloned().collect(),
                stack: vec![],
                item_type: item_type.clone(),
            },
            TermKind::ListConcat(item_type, lists) => Self {
                items: vec![],
                stack: lists.iter().rev().cloned().collect(),
                item_type: item_type.clone(),
            },
            _ => Self {
                items: vec![],
                stack: vec![term.clone()],
                item_type: Term::default(),
            },
        }
    }

    /// Returns the type of the items in the list.
    pub fn item_type(&self) -> &Term {
        &self.item_type
    }

    /// Create a term for the remaining list.
    pub fn remaining(self) -> Term {
        let item_type = self.item_type.clone();
        list_from_parts(self, item_type)
    }
}

impl Iterator for ListIter {
    type Item = ListPart;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.items.pop() {
                return Some(ListPart::Item(item));
            };

            let list = self.stack.pop()?;

            match list.get() {
                TermKind::List(_, items) => {
                    self.items.extend(items.iter().rev().cloned());
                }
                TermKind::ListConcat(_, lists) => {
                    self.stack.extend(lists.iter().rev().cloned());
                }
                _ => return Some(ListPart::Splice(list)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.items.len();

        let max = if self.stack.is_empty() {
            Some(self.items.len())
        } else {
            None
        };

        (min, max)
    }
}

impl FusedIterator for ListIter {}

/// Build a list term from a sequence of [`ListPart`]s.
pub fn list_from_parts(iter: impl IntoIterator<Item = ListPart>, item_type: Term) -> Term {
    todo!()
}
