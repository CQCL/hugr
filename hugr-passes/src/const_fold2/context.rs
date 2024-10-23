use std::sync::Arc;

use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::{ExtensionOp, Value};
use hugr_core::{IncomingPort, Node, OutgoingPort};
use itertools::Either;

use super::value_handle::{HashedConst, ValueHandle, ValueKey};
use crate::dataflow::{ConstLoader, PartialValue, TotalContext};

/// A [context](crate::dataflow::DFContext) that uses [ValueHandle]s
/// and performs [ExtensionOp::constant_fold] (using [Value]s for extension-op inputs).
///
/// Just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation.
#[derive(Debug)]
pub struct ConstFoldContext;

impl ConstLoader<ValueHandle> for ConstFoldContext {
    fn value_from_opaque(
            &self,
            node: Node,
            fields: &[usize],
            val: &OpaqueValue
    ) -> Option<ValueHandle> {
        Some(ValueHandle::new_opaque(node, fields, val.clone()))
    }

    fn value_from_const_hugr(&self, node: Node, fields: &[usize], h: &hugr_core::Hugr) -> Option<ValueHandle> {
        Some(ValueHandle(fields.iter().fold(ValueKey::Node(node), |k,i|k.field(*i)), Either::Right(Arc::new(h.clone()))))
    }

    fn value_from_function(&self, _node: Node, _type_args: &[hugr_core::types::TypeArg]) -> Option<ValueHandle> {
        
    }
}


impl TotalContext<ValueHandle> for ConstFoldContext {
    type InterpretableVal = Value;

    fn interpret_leaf_op(
        &self,
        n: Node,
        op: &ExtensionOp,
        ins: &[(IncomingPort, Value)],
    ) -> Vec<(OutgoingPort, PartialValue<ValueHandle>)> {
        let ins = ins.iter().map(|(p, v)| (*p, v.clone())).collect::<Vec<_>>();
        op.constant_fold(&ins).map_or(Vec::new(), |outs| {
            outs.into_iter()
                .map(|(p, v)| (p, ValueHandle::new(ValueKey::Node(n), v)))
                .collect()
        })
    }
}
