use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use hugr_core::ops::{OpType, Value};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort};

use super::value_handle::{ValueHandle, ValueKey};
use crate::dataflow::TotalContext;

/// A [context](crate::dataflow::DFContext) for doing analysis with [ValueHandle]s.
/// Interprets [LoadConstant](OpType::LoadConstant) nodes,
/// and [ExtensionOp](OpType::ExtensionOp) nodes where the extension does
/// (using [Value]s for extension-op inputs).
///
/// Just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation.
#[derive(Debug)]
pub struct HugrValueContext<H: HugrView>(Arc<H>);

impl<H: HugrView> HugrValueContext<H> {
    /// Creates a new instance, given ownership of the [HugrView]
    pub fn new(hugr: H) -> Self {
        Self(Arc::new(hugr))
    }
}

// Deriving Clone requires H:HugrView to implement Clone,
// but we don't need that as we only clone the Arc.
impl<H: HugrView> Clone for HugrValueContext<H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

// Any value used in an Ascent program must be hashable.
// However, there should only be one DFContext, so its hash is immaterial.
impl<H: HugrView> Hash for HugrValueContext<H> {
    fn hash<I: Hasher>(&self, _state: &mut I) {}
}

impl<H: HugrView> PartialEq for HugrValueContext<H> {
    fn eq(&self, other: &Self) -> bool {
        // Any AscentProgram should have only one DFContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        true
    }
}

impl<H: HugrView> Eq for HugrValueContext<H> {}

impl<H: HugrView> PartialOrd for HugrValueContext<H> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Any AscentProgram should have only one DFContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        Some(std::cmp::Ordering::Equal)
    }
}

impl<H: HugrView> Deref for HugrValueContext<H> {
    type Target = Hugr;

    fn deref(&self) -> &Self::Target {
        self.0.base_hugr()
    }
}

impl<H: HugrView> TotalContext<ValueHandle> for HugrValueContext<H> {
    type InterpretableVal = Value;

    fn interpret_leaf_op(
        &self,
        n: Node,
        ins: &[(IncomingPort, Value)],
    ) -> Vec<(OutgoingPort, ValueHandle)> {
        match self.0.get_optype(n) {
            OpType::LoadConstant(load_op) => {
                assert!(ins.is_empty()); // static edge, so need to find constant
                let const_node = self
                    .0
                    .single_linked_output(n, load_op.constant_port())
                    .unwrap()
                    .0;
                let const_op = self.0.get_optype(const_node).as_const().unwrap();
                vec![(
                    OutgoingPort::from(0),
                    ValueHandle::new(const_node.into(), Arc::new(const_op.value().clone())),
                )]
            }
            OpType::ExtensionOp(op) => {
                let ins = ins.iter().map(|(p, v)| (*p, v.clone())).collect::<Vec<_>>();
                op.constant_fold(&ins).map_or(Vec::new(), |outs| {
                    outs.into_iter()
                        .map(|(p, v)| (p, ValueHandle::new(ValueKey::Node(n), Arc::new(v))))
                        .collect()
                })
            }
            _ => vec![],
        }
    }
}
