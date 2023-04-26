use crate::types::ClassicType;

use super::Wire;
use derive_more::From as DerFrom;
use portgraph::NodeIndex;

pub trait BuildHandle {
    fn node(&self) -> NodeIndex;
    fn sig_out_wires(&self) -> &[Wire];
    #[inline]
    fn out_wire(&self, offset: usize) -> Wire {
        Wire(self.node(), offset)
    }
}

#[derive(DerFrom, Debug)]
pub struct DeltaID(NodeIndex, Vec<Wire>);

#[derive(DerFrom, Debug)]
pub struct KappaID(NodeIndex, Vec<Wire>);

#[derive(DerFrom, Debug, Clone)]
pub struct FuncID(NodeIndex);

#[derive(DerFrom, Debug)]
pub struct ConstID(NodeIndex, ClassicType);

impl ConstID {
    pub fn const_type(&self) -> ClassicType {
        self.1.clone()
    }
}

#[derive(DerFrom, Debug)]
pub struct BetaID(NodeIndex);

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

impl BuildHandle for DeltaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn sig_out_wires(&self) -> &[Wire] {
        &self.1
    }
}

impl BuildHandle for KappaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn sig_out_wires(&self) -> &[Wire] {
        &self.1
    }
}

impl BuildHandle for FuncID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn sig_out_wires(&self) -> &[Wire] {
        &[]
    }
}

impl BuildHandle for ConstID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn sig_out_wires(&self) -> &[Wire] {
        &[]
    }
}

impl BuildHandle for BetaID {
    #[inline]
    fn node(&self) -> NodeIndex {
        self.0
    }

    #[inline]
    fn sig_out_wires(&self) -> &[Wire] {
        &[]
    }
}
