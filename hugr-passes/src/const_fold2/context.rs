use hugr_core::ops::{ExtensionOp, Value};
use hugr_core::{IncomingPort, Node, OutgoingPort};

use super::value_handle::{ValueHandle, ValueKey};
use crate::dataflow::{PartialValue, TotalContext};

/// A [context](crate::dataflow::DFContext) that uses [ValueHandle]s
/// and performs [ExtensionOp::constant_fold] (using [Value]s for extension-op inputs).
///
/// Just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation.
#[derive(Debug)]
pub struct ConstFoldContext;

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
