//! Compilation passes acting on the HUGR program representation.

pub mod const_fold;
pub mod const_fold2;
mod half_node;
pub mod merge_bbs;
pub mod nest_cfgs;
pub mod validation;
