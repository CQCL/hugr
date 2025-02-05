//! Rewrite to inline a Call to a FuncDefn by copying the body of the function
//! into a DFG which replaces the Call node.
use thiserror::Error;

use crate::hugr::views::{DescendantsGraph, ExtractHugr, HierarchyView};
use crate::ops::DataflowParent;
use crate::ops::{handle::FuncID, OpType, DFG};
use crate::{HugrView, Node};

use super::simple_replace::HugrMutInternals;
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
        match h.get_optype(self.0) {
            OpType::Call(_) => Ok(()),
            op => Err(InlineCallError::NotCallNode(self.0, op.clone())),
        }
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<(), Self::Error> {
        self.verify(h)?;
        let orig_func = h.static_source(self.0).unwrap();
        let function = DescendantsGraph::<FuncID<true>>::try_new(&h, orig_func)
            .map_err(|_| InlineCallError::CallTargetNotFuncDefn(h.get_optype(orig_func).clone()))?;
        // Ideally we'd like the following to preserve uses from within "function" of Consts outside
        // the function, but (see https://github.com/CQCL/hugr/discussions/1642) this probably won't happen at the moment - TODO XXX FIXME
        let mut func = function.extract_hugr();
        let recursive_calls = func
            .static_targets(func.root())
            .unwrap()
            .collect::<Vec<_>>();
        let new_op = OpType::from(DFG {
            signature: func
                .root_type()
                .as_func_defn()
                .unwrap()
                .inner_signature()
                .into_owned(),
        });
        let (in_ports, out_ports) = (new_op.input_count(), new_op.output_count());
        func.replace_op(func.root(), new_op).unwrap();
        func.set_num_ports(func.root(), in_ports as _, out_ports as _);
        let func_copy = h.insert_hugr(h.get_parent(self.0).unwrap(), func);
        for (rc, p) in recursive_calls.into_iter() {
            let call_node = func_copy.node_map.get(&rc).unwrap();
            h.disconnect(*call_node, p);
            h.connect(orig_func, 0, *call_node, p);
        }
        let func_copy = func_copy.new_root;
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
