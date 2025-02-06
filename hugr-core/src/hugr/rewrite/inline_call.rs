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
        h.disconnect(self.0, h.get_optype(self.0).static_input_port().unwrap());

        let new_op = OpType::from(DFG {
            signature: h
                .get_optype(orig_func)
                .as_func_defn()
                .unwrap()
                .inner_signature()
                .into_owned(),
        });
        let (in_ports, out_ports) = (new_op.input_count(), new_op.output_count());
        h.replace_op(self.0, new_op).unwrap();
        h.set_num_ports(self.0, in_ports as _, out_ports as _);
        h.copy_descendants(orig_func, self.0);

        Ok(())
    }

    /// Failure only occurs if the node is not a Call, or the target not a FuncDefn.
    /// (Any later failure means an invalid Hugr and `panic`.)
    const UNCHANGED_ON_FAILURE: bool = true;

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        Some(self.0).into_iter()
    }
}

#[cfg(test)]
mod test {
    use std::iter::successors;

    use itertools::Itertools;

    use crate::builder::{Container, Dataflow, DataflowSubContainer, HugrBuilder, ModuleBuilder};
    use crate::ops::handle::{FuncID, NodeHandle};
    use crate::ops::Value;
    use crate::std_extensions::arithmetic::{
        int_ops::IntOpDef,
        int_types::{self, ConstInt, INT_TYPES},
    };
    use crate::{types::Signature, Hugr, HugrView, Node};

    use super::{HugrMut, InlineCall};

    fn calls(h: &impl HugrView) -> Vec<Node> {
        h.nodes().filter(|n| h.get_optype(*n).is_call()).collect()
    }

    fn extension_ops(h: &impl HugrView) -> Vec<Node> {
        h.nodes()
            .filter(|n| h.get_optype(*n).is_extension_op())
            .collect()
    }

    #[test]
    fn test_inline() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let cst3 = mb.add_constant(Value::from(ConstInt::new_u(4, 3)?));
        let func = {
            let mut fb = mb.define_function(
                "foo",
                Signature::new_endo(INT_TYPES[4].clone())
                    .with_extension_delta(int_types::EXTENSION_ID),
            )?;
            let c1 = fb.load_const(&cst3);
            let [i] = fb.input_wires_arr();
            let add = fb.add_dataflow_op(IntOpDef::iadd.with_log_width(4), [i, c1])?;
            fb.finish_with_outputs(add.outputs())?
        };
        let mut main = mb.define_function("main", Signature::new_endo(INT_TYPES[4].clone()))?;
        let call1 = main.call(func.handle(), &[], main.input_wires())?;
        let call2 = main.call(func.handle(), &[], call1.outputs())?;
        main.finish_with_outputs(call2.outputs())?;
        let mut hugr = mb.finish_hugr()?;
        let call1 = call1.node();
        let call2 = call2.node();
        assert_eq!(
            hugr.output_neighbours(func.node()).collect_vec(),
            [call1, call2]
        );
        assert_eq!(calls(&hugr), [call1, call2]);
        assert_eq!(extension_ops(&hugr).len(), 1);

        hugr.apply_rewrite(InlineCall(call1.node())).unwrap();
        hugr.validate().unwrap();
        assert_eq!(hugr.output_neighbours(func.node()).collect_vec(), [call2]);
        assert_eq!(calls(&hugr), [call2]);
        assert_eq!(extension_ops(&hugr).len(), 2);

        hugr.apply_rewrite(InlineCall(call2.node())).unwrap();
        hugr.validate().unwrap();
        assert_eq!(hugr.output_neighbours(func.node()).next(), None);
        assert_eq!(calls(&hugr), []);
        assert_eq!(extension_ops(&hugr).len(), 3);

        Ok(())
    }

    #[test]
    fn test_recursion() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let (func, rec_call) = {
            let mut fb = mb.define_function("foo", Signature::new_endo(INT_TYPES[5].clone()))?;
            let cst1 = fb.add_load_value(ConstInt::new_u(5, 1)?);
            let [i] = fb.input_wires_arr();
            let add = fb.add_dataflow_op(IntOpDef::iadd.with_log_width(5), [i, cst1])?;
            let call = fb.call(
                &FuncID::<true>::from(fb.container_node()),
                &[],
                add.outputs(),
            )?;
            (fb.finish_with_outputs(call.outputs())?, call)
        };
        let mut main = mb.define_function("main", Signature::new_endo(INT_TYPES[5].clone()))?;
        let call = main.call(func.handle(), &[], main.input_wires())?;
        let main = main.finish_with_outputs(call.outputs())?;
        let mut hugr = mb.finish_hugr()?;

        let get_nonrec_call = |h: &Hugr| {
            let v = calls(h);
            assert!(v.iter().all(|n| h.static_source(*n) == Some(func.node())));
            assert_eq!(v[0], rec_call.node());
            v.into_iter().skip(1).exactly_one()
        };

        let mut call = call.node();
        for i in 2..10 {
            hugr.apply_rewrite(InlineCall(call))?;
            hugr.validate().unwrap();
            assert_eq!(extension_ops(&hugr).len(), i);
            call = get_nonrec_call(&hugr).unwrap();
            assert_eq!(
                hugr.output_neighbours(func.node()).collect_vec(),
                [rec_call.node(), call.node()]
            );
            let mut ancestors = successors(hugr.get_parent(call), |n| hugr.get_parent(*n));
            for _ in 1..i {
                assert!(hugr.get_optype(ancestors.next().unwrap()).is_dfg());
            }
            assert_eq!(ancestors.next(), Some(main.node()));
            assert_eq!(ancestors.next(), Some(hugr.root()));
            assert_eq!(ancestors.next(), None);
        }
        Ok(())
    }
}
