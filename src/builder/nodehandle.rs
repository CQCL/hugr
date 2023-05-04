use crate::types::{ClassicType, SimpleType};

use super::Wire;
use derive_more::From as DerFrom;
use portgraph::NodeIndex;
use smol_str::SmolStr;

pub trait BuildHandle {
    fn node(&self) -> NodeIndex;
    fn num_value_outputs(&self) -> usize {
        0
    }
    fn outputs(&self) -> Vec<Wire> {
        (0..self.num_value_outputs())
            .map(|offset| self.out_wire(offset))
            .collect()
    }

    fn outputs_arr<const N: usize>(&self) -> [Wire; N] {
        self.outputs()
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
pub struct DeltaID(NodeIndex, usize);

#[derive(DerFrom, Debug)]
pub struct KappaID(NodeIndex, usize);

#[derive(DerFrom, Debug, Clone)]
pub struct FuncID(NodeIndex);

#[derive(DerFrom, Debug, Clone)]
pub struct NewTypeID {
    node: NodeIndex,
    name: SmolStr,
    core_type: SimpleType,
}

impl NewTypeID {
    // Retrieve the NewType
    pub fn get_new_type(&self) -> SimpleType {
        self.core_type.clone().into_new_type(self.name.clone())
    }

    // Retrieve the underlying core type
    pub fn get_core_type(&self) -> &SimpleType {
        &self.core_type
    }

    // Retrieve the underlying core type
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
pub struct BetaID(NodeIndex);

#[derive(DerFrom, Debug)]
pub struct LambdaID(NodeIndex);

#[derive(DerFrom, Debug)]
pub struct ThetaID(NodeIndex, usize);

#[derive(DerFrom, Debug)]
pub struct GammaID(NodeIndex, usize);

impl From<DeltaID> for FuncID {
    #[inline]
    fn from(value: DeltaID) -> Self {
        Self(value.0)
    }
}

impl From<DeltaID> for BetaID {
    #[inline]
    fn from(value: DeltaID) -> Self {
        Self(value.0)
    }
}

impl From<DeltaID> for LambdaID {
    #[inline]
    fn from(value: DeltaID) -> Self {
        Self(value.0)
    }
}

impl From<DeltaID> for ThetaID {
    #[inline]
    fn from(value: DeltaID) -> Self {
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

impl BuildHandle for GammaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for DeltaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for ThetaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn num_value_outputs(&self) -> usize {
        self.1
    }
}

impl BuildHandle for KappaID {
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

impl BuildHandle for BetaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }
}
