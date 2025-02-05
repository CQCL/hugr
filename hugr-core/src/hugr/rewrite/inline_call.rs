//! Rewrite to inline a Call to a FuncDefn by copying the body of the function
//! into a DFG which replaces the Call node.
use thiserror::Error;

use crate::ops::DataflowParent;
use crate::ops::{OpType, DFG};
use crate::{HugrView, Node};

use super::{HugrMut, Rewrite};

/// Rewrite to inline a [Call](OpType::Call) to a known [FuncDefn](OpType::FuncDefn)
pub struct InlineCall(Node);

/// Error in performing [InlineCall] rewrite.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum InlineCallError {
    /// The specified Node was not a [Call](OpType::Call)
    #[error("Node to inline {0} expected to be a Call but actually {1}")]
    NotCallNode(Node, OpType),
    /// The node was a Call, but the target was not a [FuncDefn](OpType::FuncDefn)
    /// - presumably a [FuncDecl](OpType::FuncDecl), if the Hugr is valid.
    #[error("Can only inline Call nodes targetting FuncDefn's not {0}")]
    CallTargetNotFuncDefn(OpType),
}

impl Rewrite for InlineCall {
    type ApplyResult = ();
    type Error = InlineCallError;
    fn verify(&self, h: &impl crate::HugrView) -> Result<(), Self::Error> {
        let call_ty = h.get_optype(self.0);
        if !call_ty.is_call() {
            return Err(InlineCallError::NotCallNode(self.0, call_ty.clone()));
        }
        let func_ty = h.get_optype(h.static_source(self.0).unwrap());
        if !func_ty.is_func_defn() {
            return Err(InlineCallError::CallTargetNotFuncDefn(func_ty.clone()));
        }
        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<(), Self::Error> {
        self.verify(h)?; // Now we know we have a Call to a FuncDefn.
        let orig_func = h.static_source(self.0).unwrap();
        let func_copy = h
            .copy_subtree(orig_func, h.get_parent(self.0).unwrap())
            .new_root;
        let new_op = OpType::from(DFG {
            signature: h
                .get_optype(orig_func)
                .as_func_defn()
                .unwrap()
                .inner_signature()
                .into_owned(),
        });
        let (in_ports, out_ports) = (new_op.input_count(), new_op.output_count());
        h.replace_op(func_copy, new_op).unwrap();
        h.set_num_ports(func_copy, in_ports as _, out_ports as _);
        let new_connections = h
            .all_linked_outputs(self.0)
            .filter(|(n, _)| *n != orig_func)
            .enumerate()
            .map(|(tgt_port, (src, src_port))| (src, src_port, func_copy, tgt_port.into()))
            .chain(h.node_outputs(self.0).flat_map(|src_port| {
                h.linked_inputs(self.0, src_port)
                    .map(move |(tgt, tgt_port)| (func_copy, src_port, tgt, tgt_port))
            }))
            .collect::<Vec<_>>();
        h.remove_node(self.0);
        for (src_node, src_port, tgt_node, tgt_port) in new_connections {
            h.connect(src_node, src_port, tgt_node, tgt_port);
        }
        Ok(())
    }

    /// Failure only occurs if the node is not a Call, or the target not a FuncDefn.
    /// (Any later failure means an invalid Hugr and `panic`.)
    const UNCHANGED_ON_FAILURE: bool = true;

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        Some(self.0).into_iter()
    }
}
