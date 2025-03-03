//! Rewrite to inline a Call to a FuncDefn by copying the body of the function
//! into a DFG which replaces the Call node.
use thiserror::Error;

use crate::ops::{DataflowParent, OpType, DFG};
use crate::types::Substitution;
use crate::{HugrView, Node};

use super::{HugrMut, Rewrite};

/// Rewrite to inline a [Call](OpType::Call) to a known [FuncDefn](OpType::FuncDefn)
pub struct InlineCall(Node);

/// Error in performing [InlineCall] rewrite.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InlineCallError {
    /// The specified Node was not a [Call](OpType::Call)
    #[error("Node to inline {0} expected to be a Call but actually {1}")]
    NotCallNode(Node, OpType),
    /// The node was a Call, but the target was not a [FuncDefn](OpType::FuncDefn)
    /// - presumably a [FuncDecl](OpType::FuncDecl), if the Hugr is valid.
    #[error("Call targetted node {0} which must be a FuncDefn but was {1}")]
    CallTargetNotFuncDefn(Node, OpType),
}

impl InlineCall {
    /// Create a new instance that will inline the specified node
    /// (i.e. that should be a [Call](OpType::Call))
    pub fn new(node: Node) -> Self {
        Self(node)
    }
}

impl Rewrite for InlineCall {
    type ApplyResult = ();
    type Error = InlineCallError;
    fn verify(&self, h: &impl HugrView<Node = Node>) -> Result<(), Self::Error> {
        let call_ty = h.get_optype(self.0);
        if !call_ty.is_call() {
            return Err(InlineCallError::NotCallNode(self.0, call_ty.clone()));
        }
        let func = h.static_source(self.0).unwrap();
        let func_ty = h.get_optype(func);
        if !func_ty.is_func_defn() {
            return Err(InlineCallError::CallTargetNotFuncDefn(
                func,
                func_ty.clone(),
            ));
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
        let ty_args = h
            .replace_op(self.0, new_op)
            .unwrap()
            .as_call()
            .unwrap()
            .type_args
            .clone();
        h.set_num_ports(self.0, in_ports as _, out_ports as _);

        h.copy_descendants(
            orig_func,
            self.0,
            (!ty_args.is_empty()).then_some(Substitution::new(&ty_args)),
        );
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
    use crate::ops::{Input, Value};
    use crate::std_extensions::arithmetic::{
        int_ops::{self, IntOpDef},
        int_types::{self, ConstInt, INT_TYPES},
    };
    use crate::{types::Signature, HugrView, Node};

    use super::{HugrMut, InlineCall, InlineCallError};

    fn calls(h: &impl HugrView<Node = Node>) -> Vec<Node> {
        h.nodes().filter(|n| h.get_optype(*n).is_call()).collect()
    }

    fn extension_ops(h: &impl HugrView<Node = Node>) -> Vec<Node> {
        h.nodes()
            .filter(|n| h.get_optype(*n).is_extension_op())
            .collect()
    }

    #[test]
    fn test_inline() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let cst3 = mb.add_constant(Value::from(ConstInt::new_u(4, 3)?));
        let sig = Signature::new_endo(INT_TYPES[4].clone())
            .with_extension_delta(int_ops::EXTENSION_ID)
            .with_extension_delta(int_types::EXTENSION_ID);
        let func = {
            let mut fb = mb.define_function("foo", sig.clone())?;
            let c1 = fb.load_const(&cst3);
            let [i] = fb.input_wires_arr();
            let add = fb.add_dataflow_op(IntOpDef::iadd.with_log_width(4), [i, c1])?;
            fb.finish_with_outputs(add.outputs())?
        };
        let mut main = mb.define_function("main", sig)?;
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
        let sig = Signature::new_endo(INT_TYPES[5].clone())
            .with_extension_delta(int_ops::EXTENSION_ID)
            .with_extension_delta(int_types::EXTENSION_ID);
        let (func, rec_call) = {
            let mut fb = mb.define_function("foo", sig.clone())?;
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
        let mut main = mb.define_function("main", sig)?;
        let call = main.call(func.handle(), &[], main.input_wires())?;
        let main = main.finish_with_outputs(call.outputs())?;
        let mut hugr = mb.finish_hugr()?;

        let func = func.node();
        let mut call = call.node();
        for i in 2..10 {
            hugr.apply_rewrite(InlineCall(call))?;
            hugr.validate().unwrap();
            assert_eq!(extension_ops(&hugr).len(), i);
            let v = calls(&hugr);
            assert!(v.iter().all(|n| hugr.static_source(*n) == Some(func)));

            let [rec, nonrec] = v.try_into().expect("Should be two");
            assert_eq!(rec, rec_call.node());
            assert_eq!(hugr.output_neighbours(func).collect_vec(), [rec, nonrec]);
            call = nonrec;

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

    #[test]
    fn test_bad() {
        let mut modb = ModuleBuilder::new();
        let decl = modb
            .declare(
                "UndefinedFunc",
                Signature::new_endo(INT_TYPES[3].clone()).into(),
            )
            .unwrap();
        let mut main = modb
            .define_function("main", Signature::new_endo(INT_TYPES[3].clone()))
            .unwrap();
        let call = main.call(&decl, &[], main.input_wires()).unwrap();
        let main = main.finish_with_outputs(call.outputs()).unwrap();
        let h = modb.finish_hugr().unwrap();
        let mut h2 = h.clone();
        assert_eq!(
            h2.apply_rewrite(InlineCall(call.node())),
            Err(InlineCallError::CallTargetNotFuncDefn(
                decl.node(),
                h.get_optype(decl.node()).clone()
            ))
        );
        assert_eq!(h, h2);
        let [inp, _out, _call] = h
            .children(main.node())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert_eq!(
            h2.apply_rewrite(InlineCall(inp)),
            Err(InlineCallError::NotCallNode(
                inp,
                Input {
                    types: INT_TYPES[3].clone().into()
                }
                .into()
            ))
        )
    }
}
