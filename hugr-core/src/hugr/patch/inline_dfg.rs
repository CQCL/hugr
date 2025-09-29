//! A rewrite that inlines a DFG node, moving all children
//! of the DFG except Input+Output into the DFG's parent,
//! and deleting the DFG along with its Input + Output

use super::{PatchHugrMut, PatchVerification};
use crate::core::HugrNode;
use crate::ops::handle::{DfgID, NodeHandle};
use crate::{HugrView, IncomingPort, Node, OutgoingPort, PortIndex};

/// Structure identifying an `InlineDFG` rewrite from the spec
pub struct InlineDFG<N = Node>(pub DfgID<N>);

/// Errors from an [`InlineDFG`] rewrite.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InlineDFGError<N = Node> {
    /// Node to inline was not a DFG. (E.g. node has been overwritten since the `DfgID` originated.)
    #[error("{node} was not a DFG")]
    NotDFG {
        /// The node we tried to inline
        node: N,
    },
    /// The DFG node is the hugr entrypoint
    #[error("Cannot inline the entrypoint node, {node}")]
    CantInlineEntrypoint {
        /// The node we tried to inline
        node: N,
    },
}

impl<N: HugrNode> PatchVerification for InlineDFG<N> {
    type Error = InlineDFGError<N>;

    type Node = N;

    fn verify(&self, h: &impl crate::HugrView<Node = N>) -> Result<(), Self::Error> {
        let n = self.0.node();
        if h.get_optype(n).as_dfg().is_none() {
            return Err(InlineDFGError::NotDFG { node: n });
        }
        if n == h.entrypoint() {
            return Err(InlineDFGError::CantInlineEntrypoint { node: n });
        }
        Ok(())
    }

    fn invalidated_nodes(
        &self,
        _: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        [self.0.node()].into_iter()
    }
}

impl<N: HugrNode> PatchHugrMut for InlineDFG<N> {
    /// The removed nodes: the DFG, and its Input and Output children.
    type Outcome = [N; 3];

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(
        self,
        h: &mut impl crate::hugr::HugrMut<Node = N>,
    ) -> Result<Self::Outcome, Self::Error> {
        self.verify(h)?;
        let n = self.0.node();
        let (oth_in, oth_out) = {
            let dfg_ty = h.get_optype(n);
            (
                dfg_ty.other_input_port().unwrap(),
                dfg_ty.other_output_port().unwrap(),
            )
        };
        let parent = h.get_parent(n).unwrap();
        let [input, output] = h.get_io(n).unwrap();
        for ch in h.children(n).skip(2).collect::<Vec<_>>() {
            h.set_parent(ch, parent);
        }
        // DFG Inputs. Deal with Order inputs first
        for (src_n, src_p) in h.linked_outputs(n, oth_in).collect::<Vec<_>>() {
            // Order edge from src_n to DFG => add order edge to each successor of Input node
            debug_assert_eq!(Some(src_p), h.get_optype(src_n).other_output_port());
            for tgt_n in h.output_neighbours(input).collect::<Vec<_>>() {
                h.add_other_edge(src_n, tgt_n);
            }
        }
        // And remaining (Value) inputs
        let input_ord_succs = h
            .linked_inputs(input, h.get_optype(input).other_output_port().unwrap())
            .collect::<Vec<_>>();
        for inp in h.node_inputs(n).collect::<Vec<_>>() {
            if inp == oth_in {
                continue;
            }
            // Hugr is invalid if there is no output linked to the DFG input.
            let (src_n, src_p) = h.single_linked_output(n, inp).unwrap();
            h.disconnect(n, inp); // These disconnects allow permutations to work trivially.
            let outp = OutgoingPort::from(inp.index());
            let targets = h.linked_inputs(input, outp).collect::<Vec<_>>();
            h.disconnect(input, outp);

            for (tgt_n, tgt_p) in targets {
                h.connect(src_n, src_p, tgt_n, tgt_p);
            }
            // Ensure order-successors of Input node execute after any node producing an input
            for (tgt, _) in &input_ord_succs {
                h.add_other_edge(src_n, *tgt);
            }
        }
        // DFG Outputs. Deal with Order outputs first.
        for (tgt_n, tgt_p) in h.linked_inputs(n, oth_out).collect::<Vec<_>>() {
            debug_assert_eq!(Some(tgt_p), h.get_optype(tgt_n).other_input_port());
            for src_n in h.input_neighbours(output).collect::<Vec<_>>() {
                h.add_other_edge(src_n, tgt_n);
            }
        }
        // And remaining (Value) outputs
        let output_ord_preds = h
            .linked_outputs(output, h.get_optype(output).other_input_port().unwrap())
            .collect::<Vec<_>>();
        for outport in h.node_outputs(n).collect::<Vec<_>>() {
            if outport == oth_out {
                continue;
            }
            let inpp = IncomingPort::from(outport.index());
            // Hugr is invalid if the Output node has no corresponding input
            let (src_n, src_p) = h.single_linked_output(output, inpp).unwrap();
            h.disconnect(output, inpp);

            for (tgt_n, tgt_p) in h.linked_inputs(n, outport).collect::<Vec<_>>() {
                h.connect(src_n, src_p, tgt_n, tgt_p);
                // Ensure order-predecessors of Output node execute before any node consuming a DFG output
                for (src, _) in &output_ord_preds {
                    h.add_other_edge(*src, tgt_n);
                }
            }
            h.disconnect(n, outport);
        }
        h.remove_node(input);
        h.remove_node(output);
        assert!(h.children(n).next().is_none());
        h.remove_node(n);
        Ok([n, input, output])
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use rstest::rstest;

    use crate::builder::{
        Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, SubContainer,
        endo_sig, inout_sig,
    };
    use crate::extension::prelude::qb_t;
    use crate::hugr::HugrMut;
    use crate::ops::handle::{DfgID, NodeHandle};
    use crate::ops::{OpType, Value};
    use crate::std_extensions::arithmetic::float_types;
    use crate::std_extensions::arithmetic::int_ops::IntOpDef;
    use crate::std_extensions::arithmetic::int_types::{self, ConstInt};
    use crate::types::Signature;
    use crate::utils::test_quantum_extension;
    use crate::{Direction, HugrView, Port, type_row};
    use crate::{Hugr, Wire};

    use super::InlineDFG;

    fn find_dfgs<H: HugrView>(h: &H) -> Vec<H::Node> {
        h.entry_descendants()
            .filter(|n| h.get_optype(*n).as_dfg().is_some())
            .collect()
    }
    fn extension_ops<H: HugrView>(h: &H) -> Vec<H::Node> {
        h.nodes()
            .filter(|n| matches!(h.get_optype(*n), OpType::ExtensionOp(_)))
            .collect()
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn inline_add_load_const(#[case] nonlocal: bool) -> Result<(), Box<dyn std::error::Error>> {
        use crate::hugr::patch::inline_dfg::InlineDFGError;

        let int_ty = &int_types::INT_TYPES[6];

        let mut outer = DFGBuilder::new(inout_sig(vec![int_ty.clone(); 2], vec![int_ty.clone()]))?;
        let [a, b] = outer.input_wires_arr();
        fn make_const<T: AsMut<Hugr> + AsRef<Hugr>>(
            d: &mut DFGBuilder<T>,
        ) -> Result<Wire, Box<dyn std::error::Error>> {
            let cst = Value::extension(ConstInt::new_u(6, 15)?);
            let c1 = d.add_load_const(cst);

            Ok(c1)
        }
        let c1 = nonlocal.then(|| make_const(&mut outer));
        let inner = {
            let mut inner = outer.dfg_builder_endo([(int_ty.clone(), a)])?;
            let [a] = inner.input_wires_arr();
            let c1 = c1.unwrap_or_else(|| make_const(&mut inner))?;
            let a1 = inner.add_dataflow_op(IntOpDef::iadd.with_log_width(6), [a, c1])?;
            inner.finish_with_outputs(a1.outputs())?
        };
        let [a1] = inner.outputs_arr();

        let a1_sub_b = outer.add_dataflow_op(IntOpDef::isub.with_log_width(6), [a1, b])?;
        let mut outer = outer.finish_hugr_with_outputs(a1_sub_b.outputs())?;

        // Sanity checks
        assert_eq!(
            outer.children(inner.node()).count(),
            if nonlocal { 3 } else { 5 }
        ); // Input, Output, add; + const, load_const
        assert_eq!(find_dfgs(&outer), vec![outer.entrypoint(), inner.node()]);
        let [add, sub] = extension_ops(&outer).try_into().unwrap();
        assert_eq!(
            outer.get_parent(outer.get_parent(add).unwrap()),
            outer.get_parent(sub)
        );
        assert_eq!(outer.entry_descendants().count(), 10); // 6 above + inner DFG + outer (DFG + Input + Output + sub)
        {
            // Check we can't inline the outer DFG
            let mut h = outer.clone();
            assert_eq!(
                h.apply_patch(InlineDFG(DfgID::from(h.entrypoint()))),
                Err(InlineDFGError::CantInlineEntrypoint {
                    node: h.entrypoint()
                })
            );
            assert_eq!(h, outer); // unchanged
        }

        outer.apply_patch(InlineDFG(*inner.handle()))?;
        outer.validate()?;
        assert_eq!(outer.entry_descendants().count(), 7);
        assert_eq!(find_dfgs(&outer), vec![outer.entrypoint()]);
        let [add, sub] = extension_ops(&outer).try_into().unwrap();
        assert_eq!(outer.get_parent(add), Some(outer.entrypoint()));
        assert_eq!(outer.get_parent(sub), Some(outer.entrypoint()));
        assert_eq!(
            outer.node_connections(add, sub).collect::<Vec<_>>().len(),
            1
        );
        Ok(())
    }

    #[test]
    fn permutation() -> Result<(), Box<dyn std::error::Error>> {
        let mut h = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;
        let [p, q] = h.input_wires_arr();
        let [p_h] = h
            .add_dataflow_op(test_quantum_extension::h_gate(), [p])?
            .outputs_arr();
        let swap = {
            let swap = h.dfg_builder(Signature::new_endo(vec![qb_t(), qb_t()]), [p_h, q])?;
            let [a, b] = swap.input_wires_arr();
            swap.finish_with_outputs([b, a])?
        };
        let [q, p] = swap.outputs_arr();
        let cx = h.add_dataflow_op(test_quantum_extension::cx_gate(), [q, p])?;

        let mut h = h.finish_hugr_with_outputs(cx.outputs())?;
        assert_eq!(find_dfgs(&h), vec![h.entrypoint(), swap.node()]);
        assert_eq!(h.entry_descendants().count(), 8); // Dfg+I+O, H, CX, Dfg+I+O
        // No permutation outside the swap DFG:
        assert_eq!(
            h.node_connections(p_h.node(), swap.node())
                .collect::<Vec<_>>(),
            vec![[
                Port::new(Direction::Outgoing, 0),
                Port::new(Direction::Incoming, 0)
            ]]
        );
        assert_eq!(
            h.node_connections(swap.node(), cx.node())
                .collect::<Vec<_>>(),
            vec![
                [
                    Port::new(Direction::Outgoing, 0),
                    Port::new(Direction::Incoming, 0)
                ],
                [
                    Port::new(Direction::Outgoing, 1),
                    Port::new(Direction::Incoming, 1)
                ]
            ]
        );

        h.apply_patch(InlineDFG(*swap.handle()))?;
        assert_eq!(find_dfgs(&h), vec![h.entrypoint()]);
        assert_eq!(h.entry_descendants().count(), 5); // Dfg+I+O
        let mut ops = extension_ops(&h);
        ops.sort_by_key(|n| h.num_outputs(*n)); // Put H before CX
        let [h_gate, cx] = ops.try_into().unwrap();
        // Now permutation exists:
        assert_eq!(
            h.node_connections(h_gate, cx).collect::<Vec<_>>(),
            vec![[
                Port::new(Direction::Outgoing, 0),
                Port::new(Direction::Incoming, 1)
            ]]
        );
        Ok(())
    }

    #[test]
    fn order_edges() -> Result<(), Box<dyn std::error::Error>> {
        /*      -----|-----|-----
         *           |     |
         *          H_a   H_b
         *           |.    /         NB. Order edge H_a to nested DFG
         *           | .  |
         *           |  /-|--------\
         *           |  | | .  Cst | NB. Order edge Input to LCst
         *           |  | |  . |   |
         *           |  | |   LCst |
         *           |  |  \ /     |
         *           |  |  RZ      |
         *           |  |  |       |
         *           |  |  meas    |
         *           |  |  | \     |
         *           |  |  |  if   |
         *           |  |  |  .    | NB. Order edge if to Output
         *           |  \--|-------/
         *           |  .  |
         *           | .   |         NB. Order edge nested DFG to H_a2
         *           H_a2  /
         *             \  /
         *              CX
         */
        // Extension inference here relies on quantum ops not requiring their own test_quantum_extension
        let mut outer = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()]))?;
        let [a, b] = outer.input_wires_arr();
        let h_a = outer.add_dataflow_op(test_quantum_extension::h_gate(), [a])?;
        let h_b = outer.add_dataflow_op(test_quantum_extension::h_gate(), [b])?;
        let mut inner = outer.dfg_builder(endo_sig(qb_t()), h_b.outputs())?;
        let [i] = inner.input_wires_arr();
        let f = inner.add_load_value(float_types::ConstF64::new(1.0));
        inner.add_other_wire(inner.input().node(), f.node());
        let r = inner.add_dataflow_op(test_quantum_extension::rz_f64(), [i, f])?;
        let [m, b] = inner
            .add_dataflow_op(test_quantum_extension::measure(), r.outputs())?
            .outputs_arr();
        // Node using the boolean. Here we just select between two empty computations.
        let mut if_n =
            inner.conditional_builder(([type_row![], type_row![]], b), [], type_row![])?;
        if_n.case_builder(0)?.finish_with_outputs([])?;
        if_n.case_builder(1)?.finish_with_outputs([])?;
        let if_n = if_n.finish_sub_container()?;
        inner.add_other_wire(if_n.node(), inner.output().node());
        let inner = inner.finish_with_outputs([m])?;
        outer.add_other_wire(h_a.node(), inner.node());
        let h_a2 = outer.add_dataflow_op(test_quantum_extension::h_gate(), h_a.outputs())?;
        outer.add_other_wire(inner.node(), h_a2.node());
        let cx = outer.add_dataflow_op(
            test_quantum_extension::cx_gate(),
            h_a2.outputs().chain(inner.outputs()),
        )?;
        let mut outer = outer.finish_hugr_with_outputs(cx.outputs())?;

        outer.apply_patch(InlineDFG(*inner.handle()))?;
        outer.validate()?;
        let order_neighbours = |n, d| {
            let p = outer.get_optype(n).other_port(d).unwrap();
            outer
                .linked_ports(n, p)
                .map(|(n, _)| n)
                .collect::<HashSet<_>>()
        };
        // h_a should have Order edges added to Rz and the F64 load_const
        assert_eq!(
            order_neighbours(h_a.node(), Direction::Outgoing),
            HashSet::from([r.node(), f.node()])
        );
        // Likewise the load_const should have Order edges from the inputs to the inner DFG, i.e. h_a and h_b
        assert_eq!(
            order_neighbours(f.node(), Direction::Incoming),
            HashSet::from([h_a.node(), h_b.node()])
        );
        // h_a2 should have Order edges from the measure and if
        assert_eq!(
            order_neighbours(h_a2.node(), Direction::Incoming),
            HashSet::from([m.node(), if_n.node()])
        );
        // the if should have Order edges to the CX and h_a2
        assert_eq!(
            order_neighbours(if_n.node(), Direction::Outgoing),
            HashSet::from([h_a2.node(), cx.node()])
        );
        Ok(())
    }
}
