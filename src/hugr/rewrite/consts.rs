//! Rewrite operations involving Const and LoadConst operations

use std::iter;

use crate::{
    hugr::{HugrError, HugrMut},
    HugrView, Node,
};
#[rustversion::since(1.75)] // uses impl in return position
use itertools::Itertools;
use thiserror::Error;

use super::Rewrite;

/// Remove a [`crate::ops::LoadConstant`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConstIgnore(pub Node);

/// Error from an [`RemoveConstIgnore`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RemoveConstIgnoreError {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not LoadConst).")]
    InvalidNode(Node),
    /// Node in use.
    #[error("Node: {0:?} has non-zero outgoing connections.")]
    ValueUsed(Node),
    /// Not connected to a Const.
    #[error("Node: {0:?} is not connected to a Const node.")]
    NoConst(Node),
    /// Removal error
    #[error("Removing node caused error: {0:?}.")]
    RemoveFail(#[from] HugrError),
}

#[rustversion::since(1.75)] // uses impl in return position
impl Rewrite for RemoveConstIgnore {
    type Error = RemoveConstIgnoreError;

    // The Const node the LoadConstant was connected to.
    type ApplyResult = Node;

    type InvalidationSet<'a> = iter::Once<Node>;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_load_constant()) {
            return Err(RemoveConstIgnoreError::InvalidNode(node));
        }

        if h.out_value_types(node)
            .next()
            .is_some_and(|(p, _)| h.linked_inputs(node, p).next().is_some())
        {
            return Err(RemoveConstIgnoreError::ValueUsed(node));
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let source = h
            .input_neighbours(node)
            .exactly_one()
            .map_err(|_| RemoveConstIgnoreError::NoConst(node))?;
        h.remove_node(node)?;

        Ok(source)
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        iter::once(self.0)
    }
}

/// Remove a [`crate::ops::Const`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConst(pub Node);

/// Error from an [`RemoveConst`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RemoveConstError {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not Const).")]
    InvalidNode(Node),
    /// Node in use.
    #[error("Node: {0:?} has non-zero outgoing connections.")]
    ValueUsed(Node),
    /// Removal error
    #[error("Removing node caused error: {0:?}.")]
    RemoveFail(#[from] HugrError),
}

impl Rewrite for RemoveConst {
    type Error = RemoveConstError;

    // The parent of the Const node.
    type ApplyResult = Node;

    type InvalidationSet<'a> = iter::Once<Node>;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_const()) {
            return Err(RemoveConstError::InvalidNode(node));
        }

        if h.output_neighbours(node).next().is_some() {
            return Err(RemoveConstError::ValueUsed(node));
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let source = h
            .get_parent(node)
            .expect("Const node without a parent shouldn't happen.");
        h.remove_node(node)?;

        Ok(source)
    }

    fn invalidation_set(&self) -> Self::InvalidationSet<'_> {
        iter::once(self.0)
    }
}

#[rustversion::since(1.75)] // uses impl in return position
#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        builder::{Container, Dataflow, HugrBuilder, ModuleBuilder, SubContainer},
        extension::{
            prelude::{ConstUsize, USIZE_T},
            PRELUDE_REGISTRY,
        },
        hugr::HugrMut,
        ops::{handle::NodeHandle, LeafOp},
        type_row,
        types::FunctionType,
    };
    #[test]
    fn test_const_remove() -> Result<(), Box<dyn std::error::Error>> {
        let mut build = ModuleBuilder::new();
        let con_node = build.add_constant(ConstUsize::new(2))?;

        let mut dfg_build =
            build.define_function("main", FunctionType::new_endo(type_row![]).into())?;
        let load_1 = dfg_build.load_const(&con_node)?;
        let load_2 = dfg_build.load_const(&con_node)?;
        let tup = dfg_build.add_dataflow_op(
            LeafOp::MakeTuple {
                tys: type_row![USIZE_T, USIZE_T],
            },
            [load_1, load_2],
        )?;
        dfg_build.finish_sub_container()?;

        let mut h = build.finish_prelude_hugr()?;
        assert_eq!(h.node_count(), 8);
        let tup_node = tup.node();
        // can't remove invalid node
        assert_eq!(
            h.apply_rewrite(RemoveConst(tup_node)),
            Err(RemoveConstError::InvalidNode(tup_node))
        );

        assert_eq!(
            h.apply_rewrite(RemoveConstIgnore(tup_node)),
            Err(RemoveConstIgnoreError::InvalidNode(tup_node))
        );
        let load_1_node = load_1.node();
        let load_2_node = load_2.node();
        let con_node = con_node.node();

        let remove_1 = RemoveConstIgnore(load_1_node);
        assert_eq!(
            remove_1.invalidation_set().exactly_one().ok(),
            Some(load_1_node)
        );

        let remove_2 = RemoveConstIgnore(load_2_node);

        let remove_con = RemoveConst(con_node);
        assert_eq!(
            remove_con.invalidation_set().exactly_one().ok(),
            Some(con_node)
        );

        // can't remove nodes in use
        assert_eq!(
            h.apply_rewrite(remove_1.clone()),
            Err(RemoveConstIgnoreError::ValueUsed(load_1_node))
        );

        // remove the use
        h.remove_node(tup_node)?;

        // remove first load
        let reported_con_node = h.apply_rewrite(remove_1)?;
        assert_eq!(reported_con_node, con_node);

        // still can't remove const, in use by second load
        assert_eq!(
            h.apply_rewrite(remove_con.clone()),
            Err(RemoveConstError::ValueUsed(con_node))
        );

        // remove second use
        let reported_con_node = h.apply_rewrite(remove_2)?;
        assert_eq!(reported_con_node, con_node);
        // remove const
        assert_eq!(h.apply_rewrite(remove_con)?, h.root());

        assert_eq!(h.node_count(), 4);
        assert!(h.validate(&PRELUDE_REGISTRY).is_ok());
        Ok(())
    }
}
