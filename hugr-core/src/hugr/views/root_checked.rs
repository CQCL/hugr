use std::borrow::Cow;
use std::marker::PhantomData;

use crate::hugr::internal::{HugrInternals, HugrMutInternals};
use crate::hugr::{HugrError, HugrMut};
use crate::ops::handle::NodeHandle;
use crate::ops::OpTrait;
use crate::{Hugr, Node};

use super::{check_tag, HugrView, RootTagged};

/// A view of the whole Hugr.
/// (Just provides static checking of the type of the root node)
#[derive(Clone)]
pub struct RootChecked<H, Root = Node>(H, PhantomData<Root>);

impl<H: RootTagged, Root: NodeHandle<H::Node>> RootChecked<H, Root> {
    /// Create a hierarchical view of a whole HUGR
    ///
    /// # Errors
    /// Returns [`HugrError::InvalidTag`] if the root isn't a node of the required [`OpTag`]
    ///
    /// [`OpTag`]: crate::ops::OpTag
    pub fn try_new(hugr: H) -> Result<Self, HugrError> {
        if !H::RootHandle::TAG.is_superset(Root::TAG) {
            return Err(HugrError::InvalidTag {
                required: H::RootHandle::TAG,
                actual: Root::TAG,
            });
        }
        check_tag::<Root, _>(&hugr, hugr.root())?;
        Ok(Self(hugr, PhantomData))
    }
}

impl<Root> RootChecked<Hugr, Root> {
    /// Extracts the underlying (owned) Hugr
    pub fn into_hugr(self) -> Hugr {
        self.0
    }
}

impl<Root> RootChecked<&mut Hugr, Root> {
    /// Allows immutably borrowing the underlying mutable reference
    pub fn borrow(&self) -> RootChecked<&Hugr, Root> {
        RootChecked(&*self.0, PhantomData)
    }
}

impl<H: HugrInternals, Root> HugrInternals for RootChecked<H, Root> {
    type Portgraph<'p>
        = H::Portgraph<'p>
    where
        Self: 'p;
    type Node = H::Node;

    super::impls::hugr_internal_methods! {this, &this.0}
}

impl<H: HugrView, Root> HugrView for RootChecked<H, Root> {
    super::impls::hugr_view_methods! {this, &this.0}
}

impl<H: HugrView, Root: NodeHandle<H::Node>> RootTagged for RootChecked<H, Root> {
    type RootHandle = Root;
}

impl<H: AsRef<Hugr>, Root> AsRef<Hugr> for RootChecked<H, Root> {
    fn as_ref(&self) -> &Hugr {
        self.0.as_ref()
    }
}

impl<H: HugrMutInternals, Root: NodeHandle<H::Node>> HugrMutInternals for RootChecked<H, Root> {
    fn replace_op(
        &mut self,
        node: Self::Node,
        op: impl Into<crate::ops::OpType>,
    ) -> Result<crate::ops::OpType, crate::hugr::HugrError> {
        let op = op.into();
        if node == self.root() && !Root::TAG.is_superset(op.tag()) {
            return Err(HugrError::InvalidTag {
                required: Root::TAG,
                actual: op.tag(),
            });
        }
        self.0.replace_op(node, op)
    }

    delegate::delegate! {
        to (&mut self.0) {
            fn set_root(&mut self, root: Self::Node);
            fn set_num_ports(&mut self, node: Self::Node, incoming: usize, outgoing: usize);
            fn add_ports(&mut self, node: Self::Node, direction: crate::Direction, amount: isize) -> std::ops::Range<usize>;
            fn insert_ports(&mut self, node: Self::Node, direction: crate::Direction, index: usize, amount: usize) -> std::ops::Range<usize>;
            fn set_parent(&mut self, node: Self::Node, parent: Self::Node);
            fn move_after_sibling(&mut self, node: Self::Node, after: Self::Node);
            fn move_before_sibling(&mut self, node: Self::Node, before: Self::Node);
            fn optype_mut(&mut self, node: Self::Node) -> &mut crate::ops::OpType;
            fn node_metadata_map_mut(&mut self, node: Self::Node) -> &mut crate::hugr::NodeMetadataMap;
            fn extensions_mut(&mut self) -> &mut crate::extension::ExtensionRegistry;
        }
    }
}

impl<H: HugrMut, Root: NodeHandle<H::Node>> HugrMut for RootChecked<H, Root> {
    super::impls::hugr_mut_methods! {this, &mut this.0}
}

#[cfg(test)]
mod test {
    use super::RootChecked;
    use crate::extension::prelude::MakeTuple;
    use crate::extension::ExtensionSet;
    use crate::hugr::internal::HugrMutInternals;
    use crate::hugr::{HugrError, HugrMut};
    use crate::ops::handle::{BasicBlockID, CfgID, DataflowParentID, DfgID};
    use crate::ops::{DataflowBlock, OpTag, OpType};
    use crate::{ops, type_row, types::Signature, Hugr, HugrView};

    #[test]
    fn root_checked() {
        let root_type: OpType = ops::DFG {
            signature: Signature::new(vec![], vec![]),
        }
        .into();
        let mut h = Hugr::new(root_type.clone());
        let cfg_v = RootChecked::<&Hugr, CfgID>::try_new(&h);
        assert_eq!(
            cfg_v.err(),
            Some(HugrError::InvalidTag {
                required: OpTag::Cfg,
                actual: OpTag::Dfg
            })
        );
        let mut dfg_v = RootChecked::<&mut Hugr, DfgID>::try_new(&mut h).unwrap();
        // That is a HugrMutInternal, so we can try:
        let root = dfg_v.root();
        let bb: OpType = DataflowBlock {
            inputs: type_row![],
            other_outputs: type_row![],
            sum_rows: vec![type_row![]],
            extension_delta: ExtensionSet::new(),
        }
        .into();
        let r = dfg_v.replace_op(root, bb.clone());
        assert_eq!(
            r,
            Err(HugrError::InvalidTag {
                required: OpTag::Dfg,
                actual: ops::OpTag::DataflowBlock
            })
        );
        // That didn't do anything:
        assert_eq!(dfg_v.get_optype(root), &root_type);

        // Make a RootChecked that allows any DataflowParent
        // We won't be able to do this by widening the bound:
        assert_eq!(
            RootChecked::<_, DataflowParentID>::try_new(dfg_v).err(),
            Some(HugrError::InvalidTag {
                required: OpTag::Dfg,
                actual: OpTag::DataflowParent
            })
        );

        let mut dfp_v = RootChecked::<&mut Hugr, DataflowParentID>::try_new(&mut h).unwrap();
        let r = dfp_v.replace_op(root, bb.clone());
        assert_eq!(r, Ok(root_type));
        assert_eq!(dfp_v.get_optype(root), &bb);
        // Just check we can create a nested instance (narrowing the bound)
        let mut bb_v = RootChecked::<_, BasicBlockID>::try_new(dfp_v).unwrap();

        // And it's a HugrMut:
        let nodetype = MakeTuple(type_row![]);
        bb_v.add_node_with_parent(bb_v.root(), nodetype);
    }
}
