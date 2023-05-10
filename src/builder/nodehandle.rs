use crate::types::{ClassicType, SimpleType};

use super::Wire;
use core::iter::FusedIterator;
use derive_more::From as DerFrom;
use itertools::Itertools;
use portgraph::NodeIndex;
use smol_str::SmolStr;

#[derive(Debug, Clone)]
pub struct Outputs {
    node: NodeIndex,
    range: std::ops::Range<usize>,
}

impl Iterator for Outputs {
    type Item = Wire;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|offset| Wire(self.node, offset))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.range.nth(n).map(|offset| Wire(self.node, offset))
    }

    #[inline]
    fn count(self) -> usize {
        self.range.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl ExactSizeIterator for Outputs {
    #[inline]
    fn len(&self) -> usize {
        self.range.len()
    }
}

impl DoubleEndedIterator for Outputs {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|offset| Wire(self.node, offset))
    }
}

impl FusedIterator for Outputs {}

pub trait BuildHandle {
    fn node(&self) -> NodeIndex;
    fn num_value_outputs(&self) -> usize {
        0
    }

    #[inline]
    fn outputs(&self) -> Outputs {
        Outputs {
            node: self.node(),
            range: (0..self.num_value_outputs()),
        }
    }

    fn outputs_arr<const N: usize>(&self) -> [Wire; N] {
        self.outputs()
            .collect_vec()
            .try_into()
            .expect(&format!("Incorrect number of wires: {}", N)[..])
    }

    #[inline]
    fn out_wire(&self, offset: usize) -> Wire {
        Wire(self.node(), offset)
    }
}

#[derive(DerFrom, Debug)]
pub struct OpID(NodeIndex, usize);

#[derive(DerFrom, Debug)]
pub struct DfgID(NodeIndex, usize);

#[derive(DerFrom, Debug)]
pub struct CfgID(NodeIndex, usize);

#[derive(DerFrom, Debug, Clone)]
pub struct FuncID(NodeIndex);

#[derive(DerFrom, Debug, Clone)]
pub struct NewTypeID {
    node: NodeIndex,
    name: SmolStr,
    core_type: SimpleType,
}

impl NewTypeID {
    /// Retrieve the NewType
    pub fn get_new_type(&self) -> SimpleType {
        self.core_type.clone().into_new_type(self.name.clone())
    }

    /// Retrieve the underlying core type
    pub fn get_core_type(&self) -> &SimpleType {
        &self.core_type
    }

    /// Retrieve the underlying core type
    pub fn get_name(&self) -> &SmolStr {
        &self.name
    }
}

#[derive(DerFrom, Debug)]
pub struct ConstID(NodeIndex, ClassicType);

impl ConstID {
    pub fn const_type(&self) -> ClassicType {
        self.1.clone()
    }
}

#[derive(DerFrom, Debug)]
pub struct BasicBlockID(NodeIndex);

#[derive(DerFrom, Debug)]
pub struct CaseID(NodeIndex);

#[derive(DerFrom, Debug)]
pub struct TailLoopID(NodeIndex, usize);

#[derive(DerFrom, Debug)]
pub struct ConditionalID(NodeIndex, usize);

impl From<DfgID> for FuncID {
    #[inline]
    fn from(value: DfgID) -> Self {
        Self(value.0)
    }
}

impl From<DfgID> for BasicBlockID {
    #[inline]
    fn from(value: DfgID) -> Self {
        Self(value.0)
    }
}

impl From<DfgID> for CaseID {
    #[inline]
    fn from(value: DfgID) -> Self {
        Self(value.0)
    }
}

impl From<DfgID> for TailLoopID {
    #[inline]
    fn from(value: DfgID) -> Self {
        Self(value.0, value.1)
    }
}

impl BuildHandle for OpID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for ConditionalID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for DfgID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for TailLoopID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for CfgID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for FuncID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }
}

impl BuildHandle for NewTypeID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.node
    }
}

impl BuildHandle for ConstID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }
}

impl BuildHandle for BasicBlockID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }
}
