use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use hugr_core::ops::OpType;
use hugr_core::{Hugr, HugrView, Node};

// ALAN Note this probably belongs with ValueHandle, outside datalog
use super::{DFContext, PartialValue};
use crate::const_fold2::value_handle::ValueHandle;

#[derive(Debug)]
pub(super) struct DataflowContext<H: HugrView>(Arc<H>);

impl<H: HugrView> DataflowContext<H> {
    pub fn new(hugr: H) -> Self {
        Self(Arc::new(hugr))
    }
}

// Deriving Clone requires H:HugrView to implement Clone,
// but we don't need that as we only clone the Arc.
impl<H: HugrView> Clone for DataflowContext<H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<H: HugrView> Hash for DataflowContext<H> {
    fn hash<I: Hasher>(&self, _state: &mut I) {}
}

impl<H: HugrView> PartialEq for DataflowContext<H> {
    fn eq(&self, other: &Self) -> bool {
        // Any AscentProgram should have only one DataflowContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        true
    }
}

impl<H: HugrView> Eq for DataflowContext<H> {}

impl<H: HugrView> PartialOrd for DataflowContext<H> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Any AscentProgram should have only one DataflowContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        Some(std::cmp::Ordering::Equal)
    }
}

impl<H: HugrView> Deref for DataflowContext<H> {
    type Target = Hugr;

    fn deref(&self) -> &Self::Target {
        self.0.base_hugr()
    }
}

impl<H: HugrView> DFContext<ValueHandle> for DataflowContext<H> {
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
            _ => None,
        }
    }
}
