use std::marker::PhantomData;

use crate::hugr::HugrError;
use crate::ops::handle::NodeHandle;
use crate::ops::{OpTag, OpTrait};
use crate::{Hugr, Node};

use super::HugrView;

mod dfg;
pub use dfg::InvalidSignature;

/// A wrapper over a Hugr that ensures the entrypoint optype is of the required
/// [`OpTag`].
#[derive(Clone)]
pub struct RootChecked<H, Handle = Node>(H, PhantomData<Handle>);

impl<H: HugrView, Handle: NodeHandle<H::Node>> RootChecked<H, Handle> {
    /// A tag that can contain the operation of the hugr entrypoint node.
    const TAG: OpTag = Handle::TAG;

    /// Returns the most specific tag that can be applied to the entrypoint node.
    pub fn tag(&self) -> OpTag {
        let tag = self.0.get_optype(self.0.entrypoint()).tag();
        debug_assert!(Self::TAG.is_superset(tag));
        tag
    }

    /// Create a hierarchical view of a whole HUGR
    ///
    /// # Errors
    /// Returns [`HugrError::InvalidTag`] if the entrypoint isn't a node of the required [`OpTag`]
    ///
    /// [`OpTag`]: crate::ops::OpTag
    pub fn try_new(hugr: H) -> Result<Self, HugrError> {
        Self::check(&hugr)?;
        Ok(Self(hugr, PhantomData))
    }

    /// Check if a Hugr is valid for the given [`OpTag`].
    ///
    /// To check arbitrary nodes, use [`check_tag`].
    pub fn check(hugr: &H) -> Result<(), HugrError> {
        check_tag::<Handle, _>(hugr, hugr.entrypoint())?;
        Ok(())
    }

    /// Returns a reference to the underlying Hugr.
    pub fn hugr(&self) -> &H {
        &self.0
    }

    /// Extracts the underlying Hugr
    pub fn into_hugr(self) -> H {
        self.0
    }

    /// Returns a wrapper over a reference to the underlying Hugr.
    pub fn as_ref(&self) -> RootChecked<&H, Handle> {
        RootChecked(&self.0, PhantomData)
    }
}

impl<H: AsRef<Hugr>, Handle> AsRef<Hugr> for RootChecked<H, Handle> {
    fn as_ref(&self) -> &Hugr {
        self.0.as_ref()
    }
}

/// A trait for types that can be checked for a specific [`OpTag`] at their entrypoint node.
///
/// This is used mainly specifying function inputs that may either be a [`HugrView`] or an already checked [`RootChecked`].
pub trait RootCheckable<H: HugrView, Handle: NodeHandle<H::Node>>: Sized {
    /// Wrap the Hugr in a [`RootChecked`] if it is valid for the required [`OpTag`].
    ///
    /// If `Self` is already a [`RootChecked`], it is a no-op.
    fn try_into_checked(self) -> Result<RootChecked<H, Handle>, HugrError>;
}
impl<H: HugrView, Handle: NodeHandle<H::Node>> RootCheckable<H, Handle> for H {
    fn try_into_checked(self) -> Result<RootChecked<H, Handle>, HugrError> {
        RootChecked::try_new(self)
    }
}
impl<H: HugrView, Handle: NodeHandle<H::Node>> RootCheckable<H, Handle> for RootChecked<H, Handle> {
    fn try_into_checked(self) -> Result<RootChecked<H, Handle>, HugrError> {
        Ok(self)
    }
}

/// Check that the node in a HUGR can be represented by the required tag.
pub fn check_tag<Required: NodeHandle<N>, N>(
    hugr: &impl HugrView<Node = N>,
    node: N,
) -> Result<(), HugrError> {
    let actual = hugr.get_optype(node).tag();
    let required = Required::TAG;
    if !required.is_superset(actual) {
        return Err(HugrError::InvalidTag { required, actual });
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::RootChecked;
    use crate::hugr::HugrError;
    use crate::ops::handle::{CfgID, DfgID};
    use crate::ops::{OpTag, OpType};
    use crate::{Hugr, ops, types::Signature};

    #[test]
    fn root_checked() {
        let root_type: OpType = ops::DFG {
            signature: Signature::new(vec![], vec![]),
        }
        .into();
        let mut h = Hugr::new_with_entrypoint(root_type.clone()).unwrap();
        let cfg_v = RootChecked::<_, CfgID>::check(&h);
        assert_eq!(
            cfg_v.err(),
            Some(HugrError::InvalidTag {
                required: OpTag::Cfg,
                actual: OpTag::Dfg
            })
        );
        // This should succeed
        let dfg_v = RootChecked::<&mut Hugr, DfgID>::try_new(&mut h).unwrap();
        assert!(OpTag::Dfg.is_superset(dfg_v.tag()));
        assert_eq!(dfg_v.as_ref().tag(), dfg_v.tag());
    }
}
