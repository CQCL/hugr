use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use hugr_core::ops::{CustomOp, DataflowOpTrait, OpType};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, PortIndex};

use super::{ValueHandle, ValueKey};
use crate::const_fold2::datalog::{DFContext, PartialValue};

/// An implementation of [DFContext] with [ValueHandle]
/// that just stores a Hugr (actually any [HugrView]),
/// (there is )no state for operation-interpretation).
#[derive(Debug)]
pub struct HugrValueContext<H: HugrView>(Arc<H>);

impl<H: HugrView> HugrValueContext<H> {
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

impl<H: HugrView> DFContext<ValueHandle> for HugrValueContext<H> {
    fn hugr(&self) -> &impl HugrView {
        self.0.as_ref()
    }

    fn interpret_leaf_op(
        &self,
        n: Node,
        ins: &[PartialValue<ValueHandle>],
    ) -> Option<Vec<PartialValue<ValueHandle>>> {
        match self.0.get_optype(n) {
            OpType::LoadConstant(load_op) => {
                // ins empty as static edge, we need to find the constant ourselves
                let const_node = self
                    .0
                    .single_linked_output(n, load_op.constant_port())
                    .unwrap()
                    .0;
                let const_op = self.0.get_optype(const_node).as_const().unwrap();
                Some(vec![ValueHandle::new(
                    const_node.into(),
                    Arc::new(const_op.value().clone()),
                )
                .into()])
            }
            OpType::CustomOp(CustomOp::Extension(op)) => {
                let sig = op.signature();
                let known_ins = sig
                    .input_types()
                    .into_iter()
                    .enumerate()
                    .zip(ins.iter())
                    .filter_map(|((i, ty), pv)| {
                        pv.clone()
                            .try_into_value(ty)
                            .map(|v| (IncomingPort::from(i), v))
                            .ok()
                    })
                    .collect::<Vec<_>>();
                let outs = op.constant_fold(&known_ins)?;
                let mut res = vec![PartialValue::bottom(); sig.output_count()];
                for (op, v) in outs {
                    res[op.index()] = ValueHandle::new(ValueKey::Node(n), Arc::new(v)).into()
                }
                Some(res)
            }
            _ => None,
        }
    }
}
