use hugr::{ops::OpType, HugrView};
use inkwell::values::BasicValueEnum;

use crate::fat::FatNode;

use super::func::RowPromise;

/// A type used whenever emission is delegated to a function, for example in
/// [crate::emit::EmitOp].
pub struct EmitOpArgs<'c, OT, H> {
    /// The [HugrView] and [hugr::Node] we are emitting
    pub node: FatNode<'c, OT, H>,
    /// The values that should be used for all Value input ports of the node
    pub inputs: Vec<BasicValueEnum<'c>>,
    /// The results of the node should be put here
    pub outputs: RowPromise<'c>,
}

impl<'c, OT, H> EmitOpArgs<'c, OT, H> {
    /// Get the internal [FatNode]
    pub fn node(&self) -> FatNode<'c, OT, H> {
        self.node.clone()
    }
}

impl<'c, H: HugrView> EmitOpArgs<'c, OpType, H> {
    /// Attempt to specialise the internal [FatNode].
    pub fn try_into_ot<OT: 'c>(self) -> Result<EmitOpArgs<'c, OT, H>, Self>
    where
        &'c OpType: TryInto<&'c OT>,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
        } = self;
        match node.try_into_ot() {
            Some(new_node) => Ok(EmitOpArgs {
                node: new_node,
                inputs,
                outputs,
            }),
            None => Err(EmitOpArgs {
                node,
                inputs,
                outputs,
            }),
        }
    }

    /// Specialise the internal [FatNode].
    ///
    /// Panics if `OT` is not the [HugrView::get_optype] of the internal
    /// [hugr::Node].
    pub fn into_ot<OTInto: PartialEq + 'c>(self, ot: &OTInto) -> EmitOpArgs<'c, OTInto, H>
    where
        for<'a> &'a OpType: TryInto<&'a OTInto>,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
        } = self;
        EmitOpArgs {
            node: node.into_ot(ot),
            inputs,
            outputs,
        }
    }
}
