//! Rewrite operations involving Const and `LoadConst` operations

use std::iter;

use crate::{HugrView, Node, core::HugrNode, hugr::HugrMut};
use itertools::Itertools;
use thiserror::Error;

use super::{PatchHugrMut, PatchVerification};

/// Remove a [`crate::ops::LoadConstant`] node with no consumers.
#[derive(Debug, Clone)]
pub struct RemoveLoadConstant<N = Node>(pub N);

/// Error from an [`RemoveConst`] or [`RemoveLoadConstant`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemoveError<N = Node> {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not correct operation).")]
    InvalidNode(N),
    /// Node in use.
    #[error("Node: {0} has non-zero outgoing connections.")]
    ValueUsed(N),
}

impl<N: HugrNode> PatchVerification for RemoveLoadConstant<N> {
    type Error = RemoveError<N>;
    type Node = N;

    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_load_constant()) {
            return Err(RemoveError::InvalidNode(node));
        }
        let (p, _) = h
            .out_value_types(node)
            .exactly_one()
            .ok()
            .expect("LoadConstant has only one output.");
        if h.linked_inputs(node, p).next().is_some() {
            return Err(RemoveError::ValueUsed(node));
        }

        Ok(())
    }

    fn invalidated_nodes(
        &self,
        _: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        iter::once(self.0)
    }
}

impl<N: HugrNode> PatchHugrMut for RemoveLoadConstant<N> {
    /// The [`Const`](crate::ops::Const) node the [`LoadConstant`](crate::ops::LoadConstant) was
    /// connected to.
    type Outcome = N;

    const UNCHANGED_ON_FAILURE: bool = true;
    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<Self::Outcome, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let source = h
            .input_neighbours(node)
            .exactly_one()
            .ok()
            .expect("Validation should check a Const is connected to LoadConstant.");
        h.remove_node(node);

        Ok(source)
    }
}

/// Remove a [`crate::ops::Const`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConst<N = Node>(pub N);

impl<N: HugrNode> PatchVerification for RemoveConst<N> {
    type Node = N;
    type Error = RemoveError<N>;

    fn verify(&self, h: &impl HugrView<Node = N>) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_const()) {
            return Err(RemoveError::InvalidNode(node));
        }

        if h.output_neighbours(node).next().is_some() {
            return Err(RemoveError::ValueUsed(node));
        }

        Ok(())
    }

    fn invalidated_nodes(
        &self,
        _: &impl HugrView<Node = Self::Node>,
    ) -> impl Iterator<Item = Self::Node> {
        iter::once(self.0)
    }
}

impl<N: HugrNode> PatchHugrMut for RemoveConst<N> {
    // The parent of the Const node.
    type Outcome = N;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn apply_hugr_mut(self, h: &mut impl HugrMut<Node = N>) -> Result<Self::Outcome, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let parent = h
            .get_parent(node)
            .expect("Const node without a parent shouldn't happen.");
        h.remove_node(node);

        Ok(parent)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::{
        builder::{Container, Dataflow, HugrBuilder, ModuleBuilder, SubContainer},
        extension::prelude::ConstUsize,
        ops::{Value, handle::NodeHandle},
        type_row,
        types::Signature,
    };
    #[test]
    fn test_const_remove() -> Result<(), Box<dyn std::error::Error>> {
        let mut build = ModuleBuilder::new();
        let con_node = build.add_constant(Value::extension(ConstUsize::new(2)));

        let mut dfg_build = build.define_function("main", Signature::new_endo(type_row![]))?;
        let load_1 = dfg_build.load_const(&con_node);
        let load_2 = dfg_build.load_const(&con_node);
        let tup = dfg_build.make_tuple([load_1, load_2])?;
        dfg_build.finish_sub_container()?;

        let mut h = build.finish_hugr()?;
        // nodes are Module, Function, Input, Output, Const, LoadConstant*2, MakeTuple
        assert_eq!(h.num_nodes(), 8);
        let tup_node = tup.node();
        // can't remove invalid node
        assert_eq!(
            h.apply_patch(RemoveConst(tup_node)),
            Err(RemoveError::InvalidNode(tup_node))
        );

        assert_eq!(
            h.apply_patch(RemoveLoadConstant(tup_node)),
            Err(RemoveError::InvalidNode(tup_node))
        );
        let load_1_node = load_1.node();
        let load_2_node = load_2.node();
        let con_node = con_node.node();

        let remove_1 = RemoveLoadConstant(load_1_node);
        assert_eq!(
            remove_1.invalidated_nodes(&h).exactly_one().ok(),
            Some(load_1_node)
        );

        let remove_2 = RemoveLoadConstant(load_2_node);

        let remove_con = RemoveConst(con_node);
        assert_eq!(
            remove_con.invalidated_nodes(&h).exactly_one().ok(),
            Some(con_node)
        );

        // can't remove nodes in use
        assert_eq!(
            h.apply_patch(remove_1.clone()),
            Err(RemoveError::ValueUsed(load_1_node))
        );

        // remove the use
        h.remove_node(tup_node);

        // remove first load
        let reported_con_node = h.apply_patch(remove_1)?;
        assert_eq!(reported_con_node, con_node);

        // still can't remove const, in use by second load
        assert_eq!(
            h.apply_patch(remove_con.clone()),
            Err(RemoveError::ValueUsed(con_node))
        );

        // remove second use
        let reported_con_node = h.apply_patch(remove_2)?;
        assert_eq!(reported_con_node, con_node);
        // remove const
        assert_eq!(h.apply_patch(remove_con)?, h.entrypoint());

        assert_eq!(h.num_nodes(), 4);
        assert!(h.validate().is_ok());
        Ok(())
    }
}
