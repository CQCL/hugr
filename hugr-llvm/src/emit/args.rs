use hugr_core::{ops::OpType, HugrView, Node};
use inkwell::{builder::Builder, values::{BasicValueEnum, PointerValue}};
use anyhow::Result;

use crate::utils::fat::FatNode;

use super::func::{RowMailBox, RowPromise};

/// A type used whenever emission is delegated to a function, for example in
/// [crate::custom::extension_op::ExtensionOpMap::emit_extension_op].
pub struct EmitOpArgs<'c, 'hugr, OT, H> {
    /// The [HugrView] and [hugr_core::Node] we are emitting
    pub node: FatNode<'hugr, OT, H>,
    /// The values that should be used for all Value input ports of the node
    pub inputs: Vec<BasicValueEnum<'c>>,
    /// The results of the node should be put here
    pub outputs: RowPromise<'c>,

    input_mailbox: RowMailBox<'c>,
}

impl<'c, 'hugr, OT, H> EmitOpArgs<'c, 'hugr, OT, H> {
    pub fn try_new(
        builder: &Builder<'c>,
        node: FatNode<'hugr, OT, H>,
        input_mailbox: RowMailBox<'c>,
        outputs: RowPromise<'c>,
    ) -> Result<Self> {
        let inputs = input_mailbox.read_vec(builder, [])?;
        Ok(Self {
            node,
            inputs,
            outputs,
            input_mailbox
        })
    }

    /// Get the internal [FatNode]
    pub fn node(&self) -> FatNode<'hugr, OT, H> {
        self.node
    }

    pub fn input_alloca(&self, i: usize) -> PointerValue {
        self.input_mailbox[i].alloca()
    }

    pub fn output_alloca(&self, i: usize) -> PointerValue {
        self.outputs[i].alloca()
    }
}

impl<'c, 'hugr, H: HugrView<Node = Node>> EmitOpArgs<'c, 'hugr, OpType, H> {
    /// Attempt to specialise the internal [FatNode].
    pub fn try_into_ot<OT>(self) -> Result<EmitOpArgs<'c, 'hugr, OT, H>, Self>
    where
        for<'a> &'a OpType: TryInto<&'a OT>,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
            input_mailbox
        } = self;
        match node.try_into_ot() {
            Some(new_node) => Ok(EmitOpArgs {
                node: new_node,
                inputs,
                outputs,
                input_mailbox
            }),
            None => Err(EmitOpArgs {
                node,
                inputs,
                outputs,
                input_mailbox
            }),
        }
    }

    /// Specialise the internal [FatNode].
    ///
    /// Panics if `OT` is not the [HugrView::get_optype] of the internal
    /// [hugr_core::Node].
    pub fn into_ot<OTInto: PartialEq + 'c>(self, ot: &OTInto) -> EmitOpArgs<'c, 'hugr, OTInto, H>
    where
        for<'a> &'a OpType: TryInto<&'a OTInto>,
    {
        let EmitOpArgs {
            node,
            inputs,
            outputs,
            input_mailbox
        } = self;
        EmitOpArgs {
            node: node.into_ot(ot),
            inputs,
            outputs,
            input_mailbox
        }
    }
}
