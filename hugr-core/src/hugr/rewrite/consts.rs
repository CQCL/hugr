//! Rewrite operations involving Const and LoadConst operations

use std::iter;

use crate::{hugr::HugrMut, HugrView, Node};
use itertools::Itertools;
use thiserror::Error;

use super::Rewrite;

/// Remove a [`crate::ops::LoadConstant`] node with no consumers.
#[derive(Debug, Clone)]
pub struct RemoveLoadConstant(pub Node);

/// Error from an [`RemoveConst`] or [`RemoveLoadConstant`] operation.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RemoveError {
    /// Invalid node.
    #[error("Node is invalid (either not in HUGR or not correct operation).")]
    InvalidNode(Node),
    /// Node in use.
    #[error("Node: {0:?} has non-zero outgoing connections.")]
    ValueUsed(Node),
}

impl Rewrite for RemoveLoadConstant {
    type Error = RemoveError;

    // The Const node the LoadConstant was connected to.
    type ApplyResult = Node;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
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

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
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

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        iter::once(self.0)
    }
}

/// Remove a [`crate::ops::Const`] node with no outputs.
#[derive(Debug, Clone)]
pub struct RemoveConst(pub Node);

impl Rewrite for RemoveConst {
    type Error = RemoveError;

    // The parent of the Const node.
    type ApplyResult = Node;

    const UNCHANGED_ON_FAILURE: bool = true;

    fn verify(&self, h: &impl HugrView) -> Result<(), Self::Error> {
        let node = self.0;

        if (!h.contains_node(node)) || (!h.get_optype(node).is_const()) {
            return Err(RemoveError::InvalidNode(node));
        }

        if h.output_neighbours(node).next().is_some() {
            return Err(RemoveError::ValueUsed(node));
        }

        Ok(())
    }

    fn apply(self, h: &mut impl HugrMut) -> Result<Self::ApplyResult, Self::Error> {
        self.verify(h)?;
        let node = self.0;
        let parent = h
            .get_parent(node)
            .expect("Const node without a parent shouldn't happen.");
        h.remove_node(node);

        Ok(parent)
    }

    fn invalidation_set(&self) -> impl Iterator<Item = Node> {
        iter::once(self.0)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::extension::prelude::PRELUDE_ID;
    use crate::{
        builder::{Container, Dataflow, HugrBuilder, ModuleBuilder, SubContainer},
        extension::{prelude::ConstUsize, PRELUDE_REGISTRY},
        ops::{handle::NodeHandle, Value},
        type_row,
        types::Signature,
    };
    #[test]
    fn test_const_remove() -> Result<(), Box<dyn std::error::Error>> {
        let mut build = ModuleBuilder::new();
        let con_node = build.add_constant(Value::extension(ConstUsize::new(2)));

        let mut dfg_build = build.define_function(
            "main",
            Signature::new_endo(type_row![]).with_extension_delta(PRELUDE_ID.clone()),
        )?;
        let load_1 = dfg_build.load_const(&con_node);
        let load_2 = dfg_build.load_const(&con_node);
        let tup = dfg_build.make_tuple([load_1, load_2])?;
        dfg_build.finish_sub_container()?;

        let mut h = build.finish_prelude_hugr()?;
        // nodes are Module, Function, Input, Output, Const, LoadConstant*2, MakeTuple
        assert_eq!(h.node_count(), 8);
        let tup_node = tup.node();
        // can't remove invalid node
        assert_eq!(
            h.apply_rewrite(RemoveConst(tup_node)),
            Err(RemoveError::InvalidNode(tup_node))
        );

        assert_eq!(
            h.apply_rewrite(RemoveLoadConstant(tup_node)),
            Err(RemoveError::InvalidNode(tup_node))
        );
        let load_1_node = load_1.node();
        let load_2_node = load_2.node();
        let con_node = con_node.node();

        let remove_1 = RemoveLoadConstant(load_1_node);
        assert_eq!(
            remove_1.invalidation_set().exactly_one().ok(),
            Some(load_1_node)
        );

        let remove_2 = RemoveLoadConstant(load_2_node);

        let remove_con = RemoveConst(con_node);
        assert_eq!(
            remove_con.invalidation_set().exactly_one().ok(),
            Some(con_node)
        );

        // can't remove nodes in use
        assert_eq!(
            h.apply_rewrite(remove_1.clone()),
            Err(RemoveError::ValueUsed(load_1_node))
        );

        // remove the use
        h.remove_node(tup_node);

        // remove first load
        let reported_con_node = h.apply_rewrite(remove_1)?;
        assert_eq!(reported_con_node, con_node);

        // still can't remove const, in use by second load
        assert_eq!(
            h.apply_rewrite(remove_con.clone()),
            Err(RemoveError::ValueUsed(con_node))
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
