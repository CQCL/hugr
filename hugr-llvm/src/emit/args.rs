use hugr_core::{HugrView, Node, ops::OpType};
use inkwell::values::BasicValueEnum;

use crate::utils::fat::FatNode;

use super::func::RowPromise;

/// A type used whenever emission is delegated to a function, for example in
/// [`crate::custom::extension_op::ExtensionOpMap::emit_extension_op`].
pub struct EmitOpArgs<'c, 'hugr, OT, H> {
    /// The [`HugrView`] and [`hugr_core::Node`] we are emitting
    pub node: FatNode<'hugr, OT, H>,
    /// The values that should be used for all Value input ports of the node
    pub inputs: Vec<BasicValueEnum<'c>>,
    /// The results of the node should be put here
    pub outputs: RowPromise<'c>,
}

impl<'hugr, OT, H> EmitOpArgs<'_, 'hugr, OT, H> {
    /// Get the internal [`FatNode`]
    #[must_use]
    pub fn node(&self) -> FatNode<'hugr, OT, H> {
        self.node
    }
}

impl<'c, 'hugr, H: HugrView<Node = Node>> EmitOpArgs<'c, 'hugr, OpType, H> {
    /// Attempt to specialise the internal [`FatNode`].
    pub fn try_into_ot<OT>(self) -> Result<EmitOpArgs<'c, 'hugr, OT, H>, Self>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
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

    /// Specialise the internal [`FatNode`].
    ///
    /// Panics if `OT` is not the [`HugrView::get_optype`] of the internal
    /// [`hugr_core::Node`].
    pub fn into_ot<OTInto: PartialEq + 'c>(self, ot: &OTInto) -> EmitOpArgs<'c, 'hugr, OTInto, H>
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
