//! Algorithms using the Hugr.

pub mod const_fold;
mod half_node;
pub mod merge_bbs;
pub mod nest_cfgs;

#[derive(Debug, Clone, Copy, Ord, Eq, PartialOrd, PartialEq)]
/// A type for algorithms to take as configuration, specifying how much
/// verification they should do. Algorithms that accept this configuration
/// should at least verify that input HUGRs are valid, and that output HUGRs are
/// valid.
///
/// The default level is `None` because verification can be expensive.
pub enum VerifyLevel {
    /// Do no verification.
    None,
    /// Verify using [HugrView::validate_no_extensions]. This is useful when you
    /// do not expect valid Extension annotations on Nodes.
    ///
    /// [HugrView::validate_no_extensions]: crate::HugrView::validate_no_extensions
    WithoutExtensions,
    /// Verify using [HugrView::validate].
    ///
    /// [HugrView::validate]: crate::HugrView::validate
    WithExtensions,
}

impl Default for VerifyLevel {
    fn default() -> Self {
        if cfg!(test) {
            Self::WithoutExtensions
        } else {
            Self::None
        }
    }
}
