//! Rewrite to peel one iteration of a [TailLoop], creating a [DFG] containing a copy of
//! the loop body, and a [Conditional] containing the original `TailLoop` node.
use derive_more::{Display, Error};

use crate::core::HugrNode;
use crate::ops::{Case, Conditional, DFG, DataflowOpTrait, Input, OpType, Output, TailLoop};
use crate::types::Signature;
use crate::{HugrView, Node, PortIndex};

use super::{HugrMut, PatchHugrMut, PatchVerification};

/// Rewrite peel one iteration of a [TailLoop] to a known [`FuncDefn`](OpType::FuncDefn)
pub struct PeelTailLoop<N = Node>(N);

/// Error in performing [`PeelTailLoop`] rewrite.
#[derive(Clone, Debug, Display, Error, PartialEq)]
#[non_exhaustive]
pub enum PeelTailLoopError<N = Node> {
    /// The specified Node was not a [TailLoop]
    #[display("Node to inline {_0} expected to be a TailLoop but actually {_1}")]
    NotTailLoop(N, OpType),
}

impl<N> PeelTailLoop<N> {
    /// Create a new instance that will inline the specified node
    /// (i.e. that should be a [TailLoop])
    pub fn new(node: N) -> Self {
        Self(node)
    }
}

impl<N: HugrNode> PatchVerification for PeelTailLoop<N> {
    type Error = PeelTailLoopError<N>;
    type Node = N;
    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), Self::Error> {
        let opty = h.get_optype(self.0);
        if !opty.is_tail_loop() {
            return Err(PeelTailLoopError::NotTailLoop(self.0, opty.clone()));
        }
        Ok(())
    }

    fn invalidation_set(&self) -> impl Iterator<Item = N> {
        Some(self.0).into_iter()
    }
}

impl<N: HugrNode> PatchHugrMut for PeelTailLoop<N> {
    type Outcome = ();
    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<(), Self::Error> {
        self.verify(h)?; // Now we know we have a TailLoop.
        let op = h.get_optype(self.0);
        let order_inport = op.other_input_port().unwrap();
        let order_outport = op.other_output_port().unwrap();
        let tl = op.as_tail_loop().unwrap();
        let Signature {
            input: loop_in,
            output: loop_out,
        } = tl.signature().into_owned();
        let sum_rows = Vec::from(tl.control_variants());
        let rest = tl.rest.clone();
        let iter_outputs = tl.body_output_row().into_owned();
        let num_iter_outputs = iter_outputs.len();
        let dfg = h.add_node_before(
            self.0,
            DFG {
                signature: Signature::new(loop_in, iter_outputs),
            },
        );

        h.copy_descendants(self.0, dfg, None);

        let cond_n = h.add_node_after(
            dfg,
            Conditional {
                sum_rows,
                other_inputs: rest,
                outputs: loop_out.clone(),
            },
        );
        debug_assert_eq!(
            h.signature(dfg).unwrap().output_types(),
            h.signature(cond_n).unwrap().input_types()
        );

        for i in 0..num_iter_outputs {
            h.connect(dfg, i, cond_n, i);
        }
        let cond = h.get_optype(cond_n).as_conditional().unwrap();
        let case_in_rows = [0, 1].map(|i| cond.case_input_row(i).unwrap());
        // Stop borrowing `cond` as it borrows `h`
        let cases = case_in_rows.map(|in_row| {
            let n = h.add_node_with_parent(
                cond_n,
                Case {
                    signature: Signature::new(in_row.clone(), loop_out.clone()),
                },
            );
            h.add_node_with_parent(n, Input { types: in_row });
            h.add_node_with_parent(
                n,
                Output {
                    types: loop_out.clone(),
                },
            );
            n
        });

        let [i, o] = h.get_io(cases[TailLoop::BREAK_TAG]).unwrap();
        for p in 0..loop_out.len() {
            h.connect(i, p, o, p);
        }

        h.set_parent(self.0, cases[TailLoop::CONTINUE_TAG]);
        let [i, o] = h.get_io(cases[TailLoop::CONTINUE_TAG]).unwrap();
        // Inputs to original TailLoop are fed to DFG; TailLoop now takes inputs from Case(.Input)
        for inport in h.node_inputs(self.0).collect::<Vec<_>>() {
            for (src_n, src_p) in h.linked_outputs(self.0, inport).collect::<Vec<_>>() {
                h.connect(src_n, src_p, dfg, inport);
                if inport == order_inport {
                    // ALAN is inport the right port here?
                    h.connect(src_n, src_p, cond_n, inport);
                }
            }
            h.disconnect(self.0, inport);
            if inport != order_inport {
                h.connect(i, inport.index(), self.0, inport);
            }
        }
        // Outputs from original TailLoop come from Conditional; TailLoop outputs go to Case(.Output)
        for outport in h.node_outputs(self.0).collect::<Vec<_>>() {
            for (tgt_n, tgt_p) in h.linked_inputs(self.0, outport).collect::<Vec<_>>() {
                h.connect(cond_n, outport, tgt_n, tgt_p);
            }
            h.disconnect(self.0, outport);
            if outport != order_outport {
                h.connect(self.0, outport, o, outport.index());
            }
        }
        Ok(())
    }

    /// Failure only occurs if the node is not a [TailLoop].
    /// (Any later failure means an invalid Hugr and `panic`.)
    const UNCHANGED_ON_FAILURE: bool = true;
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::builder::test::simple_dfg_hugr;
    use crate::builder::{
        Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, HugrBuilder,
    };
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::ops::{OpTag, OpTrait, Tag, TailLoop, handle::NodeHandle};
    use crate::std_extensions::arithmetic::int_types::INT_TYPES;
    use crate::types::{Signature, Type, TypeRow};
    use crate::{HugrView, hugr::HugrMut};

    use super::{PeelTailLoop, PeelTailLoopError};

    #[test]
    fn bad_peel() {
        let backup = simple_dfg_hugr();
        let opty = backup.entrypoint_optype().clone();
        assert!(!opty.is_tail_loop());
        let mut h = backup.clone();
        let r = h.apply_patch(PeelTailLoop::new(h.entrypoint()));
        assert_eq!(
            r,
            Err(PeelTailLoopError::NotTailLoop(backup.entrypoint(), opty))
        );
        assert_eq!(h, backup);
    }

    #[test]
    fn peel_loop_incoming_edges() {
        let i32_t = || INT_TYPES[5].clone();
        let mut fb = FunctionBuilder::new(
            "main",
            Signature::new(vec![bool_t(), usize_t(), i32_t()], usize_t()),
        )
        .unwrap();
        let helper = fb
            .module_root_builder()
            .declare(
                "helper",
                Signature::new(
                    vec![bool_t(), usize_t(), i32_t()],
                    vec![Type::new_sum([vec![bool_t()], vec![]]), usize_t()],
                )
                .into(),
            )
            .unwrap();
        let [b, u, i] = fb.input_wires_arr();
        let (tl, call) = {
            let mut tlb = fb
                .tail_loop_builder([(bool_t(), b)], [(usize_t(), u)], TypeRow::new())
                .unwrap();
            let [b, u] = tlb.input_wires_arr();
            // Static edge from FuncDecl, and 'ext' edge from function Input:
            let c = tlb.call(&helper, &[], [b, u, i]).unwrap();
            let [pred, other] = c.outputs_arr();
            (tlb.finish_with_outputs(pred, [other]).unwrap(), c.node())
        };
        let mut h = fb.finish_hugr_with_outputs(tl.outputs()).unwrap();

        h.apply_patch(PeelTailLoop::new(tl.node())).unwrap();
        h.validate().unwrap();
        assert_eq!(
            h.nodes()
                .filter(|n| h.get_optype(*n).is_tail_loop())
                .collect_vec(),
            [tl.node()]
        );
        use OpTag::*;
        assert_eq!(
            tags(&h, call),
            [FnCall, TailLoop, Case, Conditional, FuncDefn, ModuleRoot]
        );
        let [c1, c2] = h
            .all_linked_inputs(helper.node())
            .map(|(n, _p)| n)
            .collect_array()
            .unwrap();
        assert!([c1, c2].contains(&call));
        let other = if call == c1 { c2 } else { c1 };
        assert_eq!(tags(&h, other), [FnCall, Dfg, FuncDefn, ModuleRoot]);
    }

    fn tags<H: HugrView>(h: &H, n: H::Node) -> Vec<OpTag> {
        let mut v = Vec::new();
        let mut o = Some(n);
        while let Some(n) = o {
            v.push(h.get_optype(n).tag());
            o = h.get_parent(n);
        }
        v
    }

    #[test]
    fn peel_loop_order_output() {
        let i16_t = || INT_TYPES[4].clone();
        let mut fb =
            FunctionBuilder::new("main", Signature::new(vec![i16_t(), bool_t()], i16_t())).unwrap();

        let [i, b] = fb.input_wires_arr();
        let tl = {
            let mut tlb = fb
                .tail_loop_builder([(i16_t(), i), (bool_t(), b)], [], i16_t().into())
                .unwrap();
            let [i, _b] = tlb.input_wires_arr();
            // This loop only goes round once. However, we do not expect this to affect
            // peeling: *dataflow analysis* can tell us that the conditional will always
            // take one Case (that does not contain the TailLoop), we do not do that here.
            let [cont] = tlb
                .add_dataflow_op(
                    Tag::new(
                        TailLoop::BREAK_TAG,
                        tlb.loop_signature().unwrap().control_variants().into(),
                    ),
                    [i],
                )
                .unwrap()
                .outputs_arr();
            tlb.finish_with_outputs(cont, []).unwrap()
        };
        let [i2] = tl.outputs_arr();
        // Create a DFG (no inputs, one output) that reads the result of the TailLoop via an 'ext` edge
        let dfg = fb
            .dfg_builder(Signature::new(vec![], i16_t()), [])
            .unwrap()
            .finish_with_outputs([i2])
            .unwrap();
        let mut h = fb.finish_hugr_with_outputs(dfg.outputs()).unwrap();
        let tl = tl.node();

        h.apply_patch(PeelTailLoop::new(tl)).unwrap();
        h.validate().unwrap();
        {
            use OpTag::*;
            assert_eq!(
                tags(&h, tl),
                [TailLoop, Case, Conditional, FuncDefn, ModuleRoot]
            );
        }
        let [out_n] = h.output_neighbours(tl).collect_array().unwrap();
        assert!(h.get_optype(out_n).is_output());
        assert_eq!(h.get_parent(tl), h.get_parent(out_n));
        let [c] = h
            .nodes()
            .filter(|n| h.get_optype(*n).is_conditional())
            .collect_array()
            .unwrap();
        assert_eq!(
            h.output_neighbours(c).sorted().collect_vec(),
            [dfg.node(), h.get_io(dfg.node()).unwrap()[1]]
        );
    }
}
