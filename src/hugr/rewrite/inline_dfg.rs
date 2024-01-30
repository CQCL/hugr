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
#[derive(Clone, Debug, thiserror::Error)]
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
        let parent = h.get_parent(n).unwrap();
        let [input, output] = h.get_io(n).unwrap();
        for ch in h.children(n).skip(2).collect::<Vec<_>>().into_iter() {
            h.set_parent(ch, parent).unwrap();
        }
        // Inputs. Just skip any port of the Input node that no out-edges from it.
        for outp in h.node_outputs(input).collect::<Vec<_>>() {
            let inport = IncomingPort::from(outp.index());
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
