//! Definition of dataflow operations with no children.

use std::borrow::Cow;

use super::dataflow::DataflowOpTrait;
use super::{OpTag, impl_op_name};
use crate::types::{EdgeKind, Signature, Type, TypeRow};

/// An operation that creates a tagged sum value from one of its variants.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Tag {
    /// The variant to create.
    pub tag: usize,
    /// The variants of the sum type.
    /// TODO this allows *none* of the variants to contain row variables, but
    /// we could allow variants *other than the tagged one* to contain rowvars.
    pub variants: Vec<TypeRow>,
}

impl Tag {
    /// Create a new Tag operation.
    #[must_use]
    pub fn new(tag: usize, variants: Vec<TypeRow>) -> Self {
        Self { tag, variants }
    }
}

impl_op_name!(Tag);

impl DataflowOpTrait for Tag {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &'static str {
        "Tag Sum operation"
    }

    /// The signature of the operation.
    fn signature(&self) -> Cow<'_, Signature> {
        // TODO: Store a cached signature
        Cow::Owned(Signature::new(
            self.variants
                .get(self.tag)
                .expect("Not a valid tag")
                .clone(),
            vec![Type::new_sum(self.variants.clone())],
        ))
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn substitute(&self, subst: &crate::types::Substitution) -> Self {
        Self {
            variants: self.variants.iter().map(|r| r.substitute(subst)).collect(),
            tag: self.tag,
        }
    }
}
