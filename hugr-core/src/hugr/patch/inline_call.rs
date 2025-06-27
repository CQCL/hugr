//! Rewrite to inline a Call to a `FuncDefn` by copying the body of the function
//! into a DFG which replaces the Call node.
use derive_more::{Display, Error};

use crate::core::HugrNode;
use crate::ops::{DFG, DataflowParent, OpType};
use crate::types::Substitution;
use crate::{Direction, HugrView, Node};

use super::{HugrMut, PatchHugrMut, PatchVerification};

/// Rewrite to inline a [Call](OpType::Call) to a known [`FuncDefn`](OpType::FuncDefn)
pub struct InlineCall<N = Node>(N);

/// Error in performing [`InlineCall`] rewrite.
#[derive(Clone, Debug, Display, Error, PartialEq)]
#[non_exhaustive]
pub enum InlineCallError<N = Node> {
    /// The specified Node was not a [Call](OpType::Call)
    #[display("Node to inline {_0} expected to be a Call but actually {_1}")]
    NotCallNode(N, OpType),
    /// The node was a Call, but the target was not a [`FuncDefn`](OpType::FuncDefn)
    /// - presumably a [`FuncDecl`](OpType::FuncDecl), if the Hugr is valid.
    #[display("Call targetted node {_0} which must be a FuncDefn but was {_1}")]
    CallTargetNotFuncDefn(N, OpType),
}

impl<N> InlineCall<N> {
    /// Create a new instance that will inline the specified node
    /// (i.e. that should be a [Call](OpType::Call))
    pub fn new(node: N) -> Self {
        Self(node)
    }
}

impl<N: HugrNode> PatchVerification for InlineCall<N> {
    type Error = InlineCallError<N>;
    type Node = N;
    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), Self::Error> {
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

    fn invalidated_nodes(&self, _: &impl HugrView<Node = N>) -> impl Iterator<Item = N> {
        Some(self.0).into_iter()
    }
}

impl<N: HugrNode> PatchHugrMut for InlineCall<N> {
    type Outcome = ();
    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<(), Self::Error> {
        self.verify(h)?; // Now we know we have a Call to a FuncDefn.
        let orig_func = h.static_source(self.0).unwrap();

        h.disconnect(self.0, h.get_optype(self.0).static_input_port().unwrap());

        // The order input port gets renumbered because the static input
        // (which comes between the value inports and the order inport) gets removed
        let old_order_in = h.get_optype(self.0).other_input_port().unwrap();
        let order_preds = h.linked_outputs(self.0, old_order_in).collect::<Vec<_>>();
        h.disconnect(self.0, old_order_in); // PortGraph currently does this anyway

        let new_op = OpType::from(DFG {
            signature: h
                .get_optype(orig_func)
                .as_func_defn()
                .unwrap()
                .inner_signature()
                .into_owned(),
        });
        let new_order_in = new_op.other_input_port().unwrap();

        let ty_args = h
            .replace_op(self.0, new_op)
            .as_call()
            .unwrap()
            .type_args
            .clone();

        h.add_ports(self.0, Direction::Incoming, -1);

        // Reconnect order predecessors
        for (src, srcp) in order_preds {
            h.connect(src, srcp, self.0, new_order_in);
        }

        h.copy_descendants(
            orig_func,
            self.0,
            (!ty_args.is_empty()).then_some(Substitution::new(&ty_args)),
        );
        Ok(())
    }

    /// Failure only occurs if the node is not a Call, or the target not a `FuncDefn`.
    /// (Any later failure means an invalid Hugr and `panic`.)
    const UNCHANGED_ON_FAILURE: bool = true;
}

#[cfg(test)]
mod test {
    use std::iter::successors;

    use itertools::Itertools;

    use crate::builder::{
        Container, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder,
        ModuleBuilder,
    };
    use crate::extension::prelude::usize_t;
    use crate::ops::handle::{FuncID, NodeHandle};
    use crate::ops::{Input, OpType, Value};
    use crate::std_extensions::arithmetic::int_types::INT_TYPES;
    use crate::std_extensions::arithmetic::{int_ops::IntOpDef, int_types::ConstInt};

    use crate::types::{PolyFuncType, Signature, Type, TypeBound};
    use crate::{HugrView, Node};

    use super::{HugrMut, InlineCall, InlineCallError};

    fn calls(h: &impl HugrView<Node = Node>) -> Vec<Node> {
        h.entry_descendants()
            .filter(|n| h.get_optype(*n).is_call())
            .collect()
    }

    fn extension_ops(h: &impl HugrView<Node = Node>) -> Vec<Node> {
        h.entry_descendants()
            .filter(|n| h.get_optype(*n).is_extension_op())
            .collect()
    }

    #[test]
    fn test_inline() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let cst3 = mb.add_constant(Value::from(ConstInt::new_u(4, 3)?));
        let sig = Signature::new_endo(INT_TYPES[4].clone());
        let func = {
            let mut fb = mb.define_function("foo", sig.clone())?;
            let c1 = fb.load_const(&cst3);
            let mut inner = fb.dfg_builder(sig.clone(), fb.input_wires())?;
            let [i] = inner.input_wires_arr();
            let add = inner.add_dataflow_op(IntOpDef::iadd.with_log_width(4), [i, c1])?;
            let inner_res = inner.finish_with_outputs(add.outputs())?;
            fb.finish_with_outputs(inner_res.outputs())?
        };
        let mut main = mb.define_function("main", sig)?;
        let call1 = main.call(func.handle(), &[], main.input_wires())?;
        main.add_other_wire(main.input().node(), call1.node());
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

        assert_eq!(
            hugr.linked_outputs(
                call1.node(),
                hugr.get_optype(call1).other_input_port().unwrap()
            )
            .count(),
            1
        );
        hugr.apply_patch(InlineCall(call1.node())).unwrap();
        hugr.validate().unwrap();
        assert_eq!(hugr.output_neighbours(func.node()).collect_vec(), [call2]);
        assert_eq!(calls(&hugr), [call2]);
        assert_eq!(extension_ops(&hugr).len(), 2);
        assert_eq!(
            hugr.linked_outputs(
                call1.node(),
                hugr.get_optype(call1).other_input_port().unwrap()
            )
            .count(),
            1
        );
        hugr.apply_patch(InlineCall(call2.node())).unwrap();
        hugr.validate().unwrap();
        assert_eq!(hugr.output_neighbours(func.node()).next(), None);
        assert_eq!(calls(&hugr), []);
        assert_eq!(extension_ops(&hugr).len(), 3);

        Ok(())
    }

    #[test]
    fn test_recursion() -> Result<(), Box<dyn std::error::Error>> {
        let mut mb = ModuleBuilder::new();
        let sig = Signature::new_endo(INT_TYPES[5].clone());
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
            hugr.apply_patch(InlineCall(call))?;
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
            assert_eq!(ancestors.next(), Some(hugr.entrypoint()));
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
            h2.apply_patch(InlineCall(call.node())),
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
            h2.apply_patch(InlineCall(inp)),
            Err(InlineCallError::NotCallNode(
                inp,
                Input {
                    types: INT_TYPES[3].clone().into()
                }
                .into()
            ))
        );
    }

    #[test]
    fn test_polymorphic() -> Result<(), Box<dyn std::error::Error>> {
        let tuple_ty = Type::new_tuple(vec![usize_t(); 2]);
        let mut fb = FunctionBuilder::new("mkpair", Signature::new(usize_t(), tuple_ty.clone()))?;
        let helper = {
            let mut mb = fb.module_root_builder();
            let fb2 = mb.define_function(
                "id",
                PolyFuncType::new(
                    [TypeBound::Copyable.into()],
                    Signature::new_endo(Type::new_var_use(0, TypeBound::Copyable)),
                ),
            )?;
            let inps = fb2.input_wires();
            fb2.finish_with_outputs(inps)?
        };
        let call1 = fb.call(helper.handle(), &[usize_t().into()], fb.input_wires())?;
        let [call1_out] = call1.outputs_arr();
        let tup = fb.make_tuple([call1_out, call1_out])?;
        let call2 = fb.call(helper.handle(), &[tuple_ty.into()], [tup])?;
        let mut hugr = fb.finish_hugr_with_outputs(call2.outputs()).unwrap();

        assert_eq!(
            hugr.output_neighbours(helper.node()).collect::<Vec<_>>(),
            [call1.node(), call2.node()]
        );
        hugr.apply_patch(InlineCall::new(call1.node()))?;

        assert_eq!(
            hugr.output_neighbours(helper.node()).collect::<Vec<_>>(),
            [call2.node()]
        );
        assert!(hugr.get_optype(call1.node()).is_dfg());
        assert!(matches!(
            hugr.children(call1.node())
                .map(|n| hugr.get_optype(n).clone())
                .collect::<Vec<_>>()[..],
            [OpType::Input(_), OpType::Output(_)]
        ));
        Ok(())
    }
}
