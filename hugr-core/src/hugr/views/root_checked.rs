use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::hugr::internal::{HugrInternals, HugrMutInternals};
use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::Hugr;

use super::HugrView;

/// A container for a Hugr providing a static assurance that the root node is of
/// a specific type.
#[derive(Clone)]
pub struct RootChecked<H, Handle>(H, PhantomData<Handle>);

impl<H: HugrView, Handle: NodeHandle> RootChecked<H, Handle> {
    /// Create a hierarchical view of a whole HUGR
    ///
    /// # Errors
    /// Returns [`HugrError::InvalidTag`] if the root isn't a node of the required [`OpTag`]
    ///
    /// [`OpTag`]: crate::ops::OpTag
    pub fn try_new(hugr: H) -> Result<Self, HugrError> {
        if !Handle::TAG.is_superset(hugr.root_tag()) {
            return Err(HugrError::InvalidTag {
                required: Handle::TAG,
                actual: hugr.root_tag(),
            });
        }
        Ok(Self(hugr, PhantomData))
    }
}

impl<H, Handle> RootChecked<H, Handle> {
    /// Extracts the underlying (owned) Hugr
    pub fn into_hugr(self) -> H {
        self.0
    }
}

impl<H: HugrView, Handle: NodeHandle> TryFrom<H> for RootChecked<H, Handle> {
    type Error = HugrError;

    fn try_from(hugr: H) -> Result<Self, Self::Error> {
        Self::try_new(hugr)
    }
}

impl<H, Handle> Borrow<H> for RootChecked<H, Handle> {
    fn borrow(&self) -> &H {
        &self.0
    }
}

impl<H: AsRef<Hugr>, Handle> AsRef<Hugr> for RootChecked<H, Handle> {
    fn as_ref(&self) -> &Hugr {
        self.0.as_ref()
    }
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
