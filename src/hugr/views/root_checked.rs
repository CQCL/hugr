use std::marker::PhantomData;

use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::{Hugr, Node};

use super::{check_tag, RootTagged};

/// A view of the whole Hugr.
/// (Just provides static checking of the type of the root node)
pub struct RootChecked<H, Root = Node>(H, PhantomData<Root>);

impl<H: RootTagged, Root: NodeHandle> RootChecked<H, Root> {
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
        check_tag::<Root>(&hugr, hugr.root())?;
        Ok(Self(hugr, PhantomData))
    }
}

impl<Root> RootChecked<Hugr, Root> {
    /// Extracts the underlying (owned) Hugr
    pub fn into_hugr(self) -> Hugr {
        self.0
    }
}

impl<H: AsRef<Hugr>, Root: NodeHandle> RootTagged for RootChecked<H, Root> {
    type RootHandle = Root;
}

impl<H: AsRef<Hugr>, Root> AsRef<Hugr> for RootChecked<H, Root> {
    fn as_ref(&self) -> &Hugr {
        self.0.as_ref()
    }
}

impl<H: AsMut<Hugr> + AsRef<Hugr>, Root: NodeHandle> AsMut<Hugr> for RootChecked<H, Root> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Hugr {
        self.0.as_mut()
    }
}

#[cfg(test)]
mod test {
    use super::RootChecked;
    use crate::extension::ExtensionSet;
    use crate::hugr::hugrmut::sealed::HugrMutInternals;
    use crate::hugr::{HugrError, HugrMut, NodeType};
    use crate::ops::handle::{CfgID, DataflowParentID, DfgID};
    use crate::ops::{BasicBlock, LeafOp, OpTag};
    use crate::{ops, type_row, types::FunctionType, Hugr, HugrView};

    #[test]
    fn root_checked() {
        let root_type = NodeType::pure(ops::DFG {
            signature: FunctionType::new(vec![], vec![]),
        });
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
        let bb = NodeType::pure(BasicBlock::DFB {
            inputs: type_row![],
            other_outputs: type_row![],
            predicate_variants: vec![type_row![]],
            extension_delta: ExtensionSet::new(),
        });
        let r = dfg_v.replace_op(root, bb.clone());
        assert_eq!(
            r,
            Err(HugrError::InvalidTag {
                required: OpTag::Dfg,
                actual: ops::OpTag::BasicBlock
            })
        );
        // That didn't do anything:
        assert_eq!(dfg_v.get_nodetype(root), &root_type);

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
        assert_eq!(dfp_v.get_nodetype(root), &bb);

        // And it's a HugrMut:
        let nodetype = NodeType::pure(LeafOp::MakeTuple { tys: type_row![] });
        dfp_v.add_node_with_parent(dfp_v.root(), nodetype).unwrap();
    }
}
