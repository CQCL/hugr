//! Passes to lower operations in a HUGR.

use hugr_core::{
    Hugr, Node,
    hugr::{hugrmut::HugrMut, views::SiblingSubgraph},
    ops::OpType,
};

use itertools::Itertools;
use thiserror::Error;

/// Replace all operations in a HUGR according to a mapping.
/// New operations must match the signature of the old operations.
///
/// Returns a list of the replaced nodes and their old operations.
pub fn replace_many_ops<S: Into<OpType>>(
    hugr: &mut impl HugrMut<Node = Node>,
    mapping: impl Fn(&OpType) -> Option<S>,
) -> Vec<(Node, OpType)> {
    let replacements = hugr
        .entry_descendants()
        .filter_map(|node| {
            let new_op = mapping(hugr.get_optype(node))?;
            Some((node, new_op))
        })
        .collect::<Vec<_>>();

    replacements
        .into_iter()
        .map(|(node, new_op)| {
            let old_op = hugr.replace_op(node, new_op);
            (node, old_op)
        })
        .collect()
}

/// Errors produced by the [`lower_ops`] function.
#[derive(Debug, Error)]
#[error(transparent)]
#[non_exhaustive]
pub enum LowerError {
    /// Invalid subgraph.
    #[error("Subgraph formed by node is invalid: {0}")]
    InvalidSubgraph(#[from] hugr_core::hugr::views::sibling_subgraph::InvalidSubgraph),
    /// Invalid replacement
    #[error("Lowered HUGR not a valid replacement: {0}")]
    InvalidReplacement(#[from] hugr_core::hugr::views::sibling_subgraph::InvalidReplacement),
    /// Rewrite error
    #[error("Rewrite error: {0}")]
    RewriteError(#[from] hugr_core::hugr::SimpleReplacementError),
}

/// Lower operations in a HUGR according to a mapping to a replacement HUGR.
///
/// # Errors
///
/// Returns a [`LowerError`] if the lowered HUGR is invalid or if any rewrite fails.
pub fn lower_ops(
    hugr: &mut impl HugrMut<Node = Node>,
    lowering: impl Fn(&OpType) -> Option<Hugr>,
) -> Result<Vec<(Node, OpType)>, LowerError> {
    let replacements = hugr
        .entry_descendants()
        .filter_map(|node| {
            let hugr = lowering(hugr.get_optype(node))?;
            Some((node, hugr))
        })
        .collect::<Vec<_>>();

    replacements
        .into_iter()
        .map(|(node, replacement)| {
            let subcirc = SiblingSubgraph::from_node(node, hugr);
            let rw = subcirc.create_simple_replacement(hugr, replacement)?;
            let removed_nodes = hugr.apply_patch(rw)?.removed_nodes;
            Ok(removed_nodes
                .into_iter()
                .exactly_one()
                .expect("removed exactly one node"))
        })
        .collect()
}

#[cfg(test)]
mod test {
    use hugr_core::{
        HugrView,
        builder::{DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::{Noop, bool_t},
        std_extensions::logic::LogicOp,
        types::Signature,
    };

    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn noop_hugr() -> Hugr {
        let mut b = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
        let out = b
            .add_dataflow_op(Noop::new(bool_t()), [b.input_wires().next().unwrap()])
            .unwrap()
            .out_wire(0);
        b.finish_hugr_with_outputs([out]).unwrap()
    }

    #[fixture]
    fn identity_hugr() -> Hugr {
        let b = DFGBuilder::new(Signature::new_endo(bool_t())).unwrap();
        let out = b.input_wires().next().unwrap();
        b.finish_hugr_with_outputs([out]).unwrap()
    }

    #[rstest]
    fn test_replace(noop_hugr: Hugr) {
        let mut h = noop_hugr;
        let mut replaced = replace_many_ops(&mut h, |op| {
            let noop = Noop::new(bool_t());
            if op.cast() == Some(noop) {
                Some(LogicOp::Not)
            } else {
                None
            }
        });

        assert_eq!(replaced.len(), 1);
        let (n, op) = replaced.remove(0);
        assert_eq!(op, Noop::new(bool_t()).into());
        assert_eq!(h.get_optype(n), &LogicOp::Not.into());
    }

    #[rstest]
    fn test_lower(noop_hugr: Hugr, identity_hugr: Hugr) {
        let mut h = noop_hugr;

        let lowered = lower_ops(&mut h, |op| {
            let noop = Noop::new(bool_t());
            if op.cast() == Some(noop) {
                Some(identity_hugr.clone())
            } else {
                None
            }
        });

        assert_eq!(lowered.unwrap().len(), 1);
        assert_eq!(h.entry_descendants().count(), 3); // DFG, input, output
    }
}
