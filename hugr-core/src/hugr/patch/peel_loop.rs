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
        let tl = h.get_optype(self.0).as_tail_loop().unwrap();

        let Signature {
            input: loop_in,
            output: loop_out,
        } = tl.signature().into_owned();
        let iter_outputs = tl.body_output_row().into_owned();
        let num_iter_outputs = iter_outputs.len();
        let dfg = h.add_node_before(
            self.0,
            DFG {
                signature: Signature::new(loop_in, iter_outputs.clone()),
            },
        );

        h.copy_descendants(self.0, dfg, None);

        let mut other_inputs = iter_outputs;
        let sum_rows = other_inputs
            .remove(0)
            .as_sum()
            .unwrap()
            .variants()
            .map(|r| r.clone().try_into().unwrap())
            .collect();

        let cond_n = h.add_node_after(
            dfg,
            Conditional {
                sum_rows,
                other_inputs: other_inputs.into(),
                outputs: loop_out.clone(),
            },
        );
        debug_assert_eq!(
            h.signature(dfg).unwrap().output_types(),
            h.signature(cond_n).unwrap().input_types()
        );

        for i in 0..num_iter_outputs {
            h.connect(cond_n, i, dfg, i);
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
            }
            h.disconnect(self.0, inport);
            // Note this also creates an Order edge from Case.Input -> TailLoop if the loop had any Order predecessors
            h.connect(i, inport.index(), self.0, inport);
        }
        // Outputs from original TailLoop come from Conditional; TailLoop outputs go to Case(.Output)
        for outport in h.node_outputs(self.0).collect::<Vec<_>>() {
            for (tgt_n, tgt_p) in h.linked_inputs(self.0, outport).collect::<Vec<_>>() {
                h.connect(cond_n, outport, tgt_n, tgt_p);
            }
            h.disconnect(self.0, outport);
            // Note this also creates an Order edge from TailLoop -> Case.Output if the loop had any Order successors
            h.connect(self.0, outport, o, outport.index());
        }
        Ok(())
    }

    /// Failure only occurs if the node is not a [TailLoop].
    /// (Any later failure means an invalid Hugr and `panic`.)
    const UNCHANGED_ON_FAILURE: bool = true;
}
