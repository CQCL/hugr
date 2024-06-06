//! Definition of dataflow operations with no children.

use super::dataflow::DataflowOpTrait;
use super::{impl_op_name, OpTag};

use crate::extension::ExtensionSet;

use crate::{
    extension::ExtensionId,
    types::{EdgeKind, FunctionType, Type, TypeRow},
};

/// A no-op operation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Noop {
    /// The type of edges connecting the Noop.
    pub ty: Type,
}

impl Noop {
    /// Create a new Noop operation.
    pub fn new(ty: Type) -> Self {
        Self { ty }
    }
}

impl Default for Noop {
    fn default() -> Self {
        Self { ty: Type::UNIT }
    }
}

/// An operation that packs all its inputs into a tuple.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct MakeTuple {
    ///Tuple element types.
    pub tys: TypeRow,
}

impl MakeTuple {
    /// Create a new MakeTuple operation.
    pub fn new(tys: TypeRow) -> Self {
        Self { tys }
    }
}

/// An operation that unpacks a tuple into its components.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct UnpackTuple {
    ///Tuple element types.
    pub tys: TypeRow,
}

impl UnpackTuple {
    /// Create a new UnpackTuple operation.
    pub fn new(tys: TypeRow) -> Self {
        Self { tys }
    }
}

/// An operation that creates a tagged sum value from one of its variants.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
pub struct Tag {
    /// The variant to create.
    pub tag: usize,
    /// The variants of the sum type.
    pub variants: Vec<TypeRow>,
}

impl Tag {
    /// Create a new Tag operation.
    pub fn new(tag: usize, variants: Vec<TypeRow>) -> Self {
        Self { tag, variants }
    }
}

/// A node which adds a extension req to the types of the wires it is passed
/// It has no effect on the values passed along the edge
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
#[non_exhaustive]
pub struct Lift {
    /// The types of the edges
    pub type_row: TypeRow,
    /// The extensions which we're adding to the inputs
    pub new_extension: ExtensionId,
}

impl Lift {
    /// Create a new Lift operation.
    pub fn new(type_row: TypeRow, new_extension: ExtensionId) -> Self {
        Self {
            type_row,
            new_extension,
        }
    }
}

impl_op_name!(Noop);
impl_op_name!(MakeTuple);
impl_op_name!(UnpackTuple);
impl_op_name!(Tag);
impl_op_name!(Lift);

impl DataflowOpTrait for Noop {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        "Noop gate"
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        FunctionType::new(vec![self.ty.clone()], vec![self.ty.clone()])
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

impl DataflowOpTrait for MakeTuple {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        "MakeTuple operation"
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        FunctionType::new(self.tys.clone(), vec![Type::new_tuple(self.tys.clone())])
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

impl DataflowOpTrait for UnpackTuple {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        "UnpackTuple operation"
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        FunctionType::new(vec![Type::new_tuple(self.tys.clone())], self.tys.clone())
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

impl DataflowOpTrait for Tag {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        "Tag Sum operation"
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        FunctionType::new(
            self.variants
                .get(self.tag)
                .expect("Not a valid tag")
                .clone(),
            vec![Type::new_sum(self.variants.clone())],
        )
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}

impl DataflowOpTrait for Lift {
    const TAG: OpTag = OpTag::Leaf;

    /// A human-readable description of the operation.
    fn description(&self) -> &str {
        "Add a extension requirement to an edge"
    }

    /// The signature of the operation.
    fn signature(&self) -> FunctionType {
        FunctionType::new(self.type_row.clone(), self.type_row.clone())
            .with_extension_delta(ExtensionSet::singleton(&self.new_extension))
    }

    fn other_input(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }

    fn other_output(&self) -> Option<EdgeKind> {
        Some(EdgeKind::StateOrder)
    }
}
