//! Rewrite to peel one iteration of a [TailLoop], creating a [DFG] containing a copy of
//! the loop body, and a [Conditional] containing the original `TailLoop` node.
use derive_more::{Display, Error};

use crate::core::HugrNode;
use crate::ops::{
    Case, Conditional, DFG, DataflowOpTrait, Input, OpTrait, OpType, Output, TailLoop,
};
use crate::types::Signature;
use crate::{Direction, HugrView, Node};

use super::{HugrMut, PatchHugrMut, PatchVerification};

/// Rewrite that peels one iteration of a [TailLoop] by turning the
/// iteration test into a [Conditional].
#[derive(Clone, Debug, PartialEq)]
pub struct PeelTailLoop<N = Node>(N);

/// Error in performing [`PeelTailLoop`] rewrite.
#[derive(Clone, Debug, Display, Error, PartialEq)]
#[non_exhaustive]
pub enum PeelTailLoopError<N = Node> {
    /// The specified Node was not a [`TailLoop`]
    #[display("Node to peel {node} expected to be a TailLoop but actually {op}")]
    NotTailLoop {
        /// The node requested to peel
        node: N,
        /// The actual (non-tail-loop) operation
        op: OpType,
    },
}

impl<N> PeelTailLoop<N> {
    /// Create a new instance that will peel the specified [TailLoop] node
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
            return Err(PeelTailLoopError::NotTailLoop {
                node: self.0,
                op: opty.clone(),
            });
        }
        Ok(())
    }

    fn invalidated_nodes(&self, h: &impl HugrView<Node = N>) -> impl Iterator<Item = N> {
        h.get_io(self.0)
            .into_iter()
            .flat_map(|[_, output]| [self.0, output].into_iter())
    }
}

impl<N: HugrNode> PatchHugrMut for PeelTailLoop<N> {
    type Outcome = ();
    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<(), Self::Error> {
        self.verify(h)?; // Now we know we have a TailLoop!
        let loop_ty = h.optype_mut(self.0);
        let signature = loop_ty.dataflow_signature().unwrap().into_owned();
        // Replace the TailLoop with a DFG - this maintains all external connections
        let OpType::TailLoop(tl) = std::mem::replace(loop_ty, DFG { signature }.into()) else {
            panic!("Wasn't a TailLoop ?!")
        };
        let sum_rows = Vec::from(tl.control_variants());
        let rest = tl.rest.clone();
        let Signature {
            input: loop_in,
            output: loop_out,
        } = tl.signature().into_owned();

        // Copy the DFG (ex-TailLoop) children into a new TailLoop *before* we add any more
        let new_loop = h.add_node_after(self.0, tl); // Temporary parent
        h.copy_descendants(self.0, new_loop, None);

        // Add conditional inside DFG.
        let [_, dfg_out] = h.get_io(self.0).unwrap();
        let cond = Conditional {
            sum_rows,
            other_inputs: rest,
            outputs: loop_out.clone(),
        };
        let case_in_rows = [0, 1].map(|i| cond.case_input_row(i).unwrap());
        // This preserves all edges from the end of the loop body to the conditional:
        h.replace_op(dfg_out, cond);
        let cond_n = dfg_out;
        h.add_ports(cond_n, Direction::Outgoing, loop_out.len() as isize + 1);
        let dfg_out = h.add_node_before(
            cond_n,
            Output {
                types: loop_out.clone(),
            },
        );
        for p in 0..loop_out.len() {
            h.connect(cond_n, p, dfg_out, p)
        }

        // Now wire up the internals of the Conditional
        let cases = case_in_rows.map(|in_row| {
            let signature = Signature::new(in_row.clone(), loop_out.clone());
            let n = h.add_node_with_parent(cond_n, Case { signature });
            h.add_node_with_parent(n, Input { types: in_row });
            let types = loop_out.clone();
            h.add_node_with_parent(n, Output { types });
            n
        });

        h.set_parent(new_loop, cases[TailLoop::CONTINUE_TAG]);
        let [ctn_in, ctn_out] = h.get_io(cases[TailLoop::CONTINUE_TAG]).unwrap();
        let [brk_in, brk_out] = h.get_io(cases[TailLoop::BREAK_TAG]).unwrap();
        for p in 0..loop_out.len() {
            h.connect(brk_in, p, brk_out, p);
            h.connect(new_loop, p, ctn_out, p)
        }
        for p in 0..loop_in.len() {
            h.connect(ctn_in, p, new_loop, p);
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
        let op = backup.entrypoint_optype().clone();
        assert!(!op.is_tail_loop());
        let mut h = backup.clone();
        let r = h.apply_patch(PeelTailLoop::new(h.entrypoint()));
        assert_eq!(
            r,
            Err(PeelTailLoopError::NotTailLoop {
                node: backup.entrypoint(),
                op
            })
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
                    vec![Type::new_sum([vec![bool_t(); 2], vec![]]), usize_t()],
                )
                .into(),
            )
            .unwrap();
        let [b, u, i] = fb.input_wires_arr();
        let (tl, call) = {
            let mut tlb = fb
                .tail_loop_builder(
                    [(bool_t(), b), (bool_t(), b)],
                    [(usize_t(), u)],
                    TypeRow::new(),
                )
                .unwrap();
            let [b, _, u] = tlb.input_wires_arr();
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
                .count(),
            1
        );
        use OpTag::*;
        assert_eq!(tags(&h, call), [FnCall, Dfg, FuncDefn, ModuleRoot]);
        let [c1, c2] = h
            .all_linked_inputs(helper.node())
            .map(|(n, _p)| n)
            .collect_array()
            .unwrap();
        assert!([c1, c2].contains(&call));
        let other = if call == c1 { c2 } else { c1 };
        assert_eq!(
            tags(&h, other),
            [
                FnCall,
                TailLoop,
                Case,
                Conditional,
                Dfg,
                FuncDefn,
                ModuleRoot
            ]
        );
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
        let [tl] = h
            .nodes()
            .filter(|n| h.get_optype(*n).is_tail_loop())
            .collect_array()
            .unwrap();
        {
            use OpTag::*;
            assert_eq!(
                tags(&h, tl),
                [TailLoop, Case, Conditional, Dfg, FuncDefn, ModuleRoot]
            );
        }
        let [out_n] = h.output_neighbours(tl).collect_array().unwrap();
        assert!(h.get_optype(out_n).is_output());
        assert_eq!(h.get_parent(tl), h.get_parent(out_n));
    }
}
