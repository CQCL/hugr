//! A rewrite that inlines a DFG node, moving all children
//! of the DFG except Input+Output into the DFG's parent,
//! and deleting the DFG along with its Input + Output

use super::Rewrite;
use crate::ops::handle::{DfgID, NodeHandle};
use crate::ops::OpType;
use crate::{Direction, IncomingPort, Node, PortIndex};

/// Structure identifying an `InlineDFG` rewrite from the spec
pub struct InlineDFG(pub DfgID);

/// Errors from an [InlineDFG] rewrite.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InlineDFGError {
    /// Node to inline was not a DFG. (E.g. node has been overwritten since the DfgID originated.)
    #[error("Node {0} was not a DFG")]
    NotDFG(Node),
    /// DFG has no parent (is the root)
    #[error("Node did not have a parent into which to inline")]
    NoParent,
    /// DFG has other edges (i.e. Order edges) incoming/outgoing.
    /// (We don't support such as the new endpoints for such edges is not clear.)
    #[error("DFG node had non-dataflow edges in direction {0:?}")]
    HasOtherEdges(Direction),
}

impl Rewrite for InlineDFG {
    /// Returns the removed nodes: the DFG, and its Input and Output children,
    type ApplyResult = [Node; 3];
    type Error = InlineDFGError;

    type InvalidationSet<'a> = <[Node; 1] as IntoIterator>::IntoIter;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl crate::HugrView) -> Result<(), Self::Error> {
        let n = self.0.node();
        let op @ OpType::DFG { .. } = h.get_optype(n) else {
            return Err(InlineDFGError::NotDFG(n));
        };
        if h.get_parent(n).is_none() {
            return Err(InlineDFGError::NoParent);
        };

        for d in Direction::BOTH {
            if op
                .other_port(d)
                .is_some_and(|p| h.linked_ports(n, p).next().is_some())
            {
                return Err(InlineDFGError::HasOtherEdges(d));
            };
        }

        Ok(())
    }

    fn apply(self, h: &mut impl crate::hugr::HugrMut) -> Result<Self::ApplyResult, Self::Error> {
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
        for ch in h.children(n).skip(2).collect::<Vec<_>>().into_iter() {
            h.set_parent(ch, parent).unwrap();
        }
        // Inputs. Just skip any port of the Input node that no out-edges from it.
        for outp in h.node_outputs(input).collect::<Vec<_>>() {
            let inport = IncomingPort::from(outp.index());
            if inport == oth_in {
                continue;
            };
            // We don't handle the case where the DFG is missing a value on the corresponding inport.
            // (An invalid Hugr - but we could just skip it, if desired.)
            let (src_n, src_p) = h.single_linked_output(n, inport).unwrap();
            h.disconnect(n, inport).unwrap();
            let targets = h.linked_inputs(input, outp).collect::<Vec<_>>();
            h.disconnect(input, outp).unwrap();
            for (tgt_n, tgt_p) in targets {
                h.connect(src_n, src_p, tgt_n, tgt_p).unwrap();
            }
        }
        // Outputs. Just skip any output of the DFG node that isn't used.
        for outport in h.node_outputs(n).collect::<Vec<_>>() {
            if outport == oth_out {
                continue;
            };
            let inpp = IncomingPort::from(outport.index());
            // Likewise, we don't handle the case where the inner DFG doesn't have
            // an edge to an (input port of) the Output node corresponding to an edge from the DFG
            let (src_n, src_p) = h.single_linked_output(output, inpp).unwrap();
            h.disconnect(output, inpp).unwrap();
            let targets = h.linked_inputs(n, outport).collect::<Vec<_>>();
            h.disconnect(n, outport).unwrap();
            for (tgt_n, tgt_p) in targets {
                h.connect(src_n, src_p, tgt_n, tgt_p).unwrap();
            }
        }
        h.remove_node(input).unwrap();
        h.remove_node(output).unwrap();
        assert!(h.children(n).next().is_none());
        h.remove_node(n).unwrap();
        Ok([n, input, output])
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        // TODO should we return Input + Output as well?
        [self.0.node()].into_iter()
    }
}

#[cfg(test)]
mod test {
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr};

    use crate::extension::{ExtensionRegistry, ExtensionSet, PRELUDE};
    use crate::hugr::rewrite::inline_dfg::InlineDFGError;
    use crate::hugr::HugrMut;
    use crate::ops::handle::{DfgID, NodeHandle};
    use crate::ops::Const;
    use crate::std_extensions::arithmetic::int_ops::{self, IntOpDef};
    use crate::std_extensions::arithmetic::int_types::{self, ConstIntU};
    use crate::types::FunctionType;
    use crate::values::Value;
    use crate::{HugrView, Node};

    use super::InlineDFG;

    #[test]
    fn simple() -> Result<(), Box<dyn std::error::Error>> {
        let delta = ExtensionSet::singleton(&int_ops::EXTENSION_ID);
        let reg = ExtensionRegistry::try_new([
            PRELUDE.to_owned(),
            int_ops::EXTENSION.to_owned(),
            int_types::EXTENSION.to_owned(),
        ])
        .unwrap();
        let int_ty = &int_types::INT_TYPES[6];

        let mut inner = DFGBuilder::new(
            FunctionType::new_endo(vec![int_ty.clone()]).with_extension_delta(&delta),
        )?;
        let [a] = inner.input_wires_arr();
        let const_val = Value::Extension {
            c: (Box::new(ConstIntU::new(6, 15)?),),
        };
        let c1 = inner.add_load_const(Const::new(const_val, int_ty.clone())?)?;
        let a1 = inner.add_dataflow_op(IntOpDef::iadd.with_width(6), [a, c1])?;
        let inner = inner.finish_hugr_with_outputs(a1.outputs(), &reg)?;
        {
            // Check we can't inline that DFG
            let mut h = inner.clone();
            assert_eq!(
                h.apply_rewrite(InlineDFG(DfgID::from(h.root()))),
                Err(InlineDFGError::NoParent)
            );
            assert_eq!(h, inner); // unchanged
        }
        assert_eq!(inner.nodes().len(), 6); // DFG, Input, Output, const, load_const, add
        let mut outer = DFGBuilder::new(
            FunctionType::new(vec![int_ty.clone(); 2], vec![int_ty.clone()])
                .with_extension_delta(&delta),
        )?;
        let [a, b] = outer.input_wires_arr();
        let inner = outer.add_hugr_with_wires(inner, [a])?;
        let [a1] = inner.outputs_arr();
        let a1_sub_b = outer.add_dataflow_op(IntOpDef::isub.with_width(6), [a1, b])?;
        let mut outer = outer.finish_hugr_with_outputs(a1_sub_b.outputs(), &reg)?;

        fn find_dfgs(h: &impl HugrView) -> Vec<Node> {
            h.nodes()
                .filter(|n| h.get_optype(*n).as_dfg().is_some())
                .collect()
        }
        fn extension_ops(h: &impl HugrView) -> Vec<Node> {
            h.nodes()
                .filter(|n| h.get_optype(*n).as_leaf_op().is_some())
                .collect()
        }
        // Sanity checks
        assert_eq!(find_dfgs(&outer), vec![outer.root(), inner.node()]);
        let [add, sub] = extension_ops(&outer).try_into().unwrap();
        assert_eq!(
            outer.get_parent(outer.get_parent(add).unwrap()),
            outer.get_parent(sub)
        );
        assert_eq!(outer.nodes().len(), 10); // 6 above + DFG + Input + Output + sub

        outer.apply_rewrite(InlineDFG(DfgID::from(inner.node())))?;
        outer.validate(&reg)?;
        assert_eq!(outer.nodes().len(), 7);
        assert_eq!(find_dfgs(&outer), vec![outer.root()]);
        let [add, sub] = extension_ops(&outer).try_into().unwrap();
        assert_eq!(outer.get_parent(add), Some(outer.root()));
        assert_eq!(outer.get_parent(sub), Some(outer.root()));
        assert_eq!(
            outer.node_connections(add, sub).collect::<Vec<_>>().len(),
            1
        );
        Ok(())
    }
}
