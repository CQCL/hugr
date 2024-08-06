use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use hugr_core::hugr::internal::HugrInternals;
use hugr_core::ops::Value;
use hugr_core::partial_value::{ValueHandle, ValueKey};
use hugr_core::{Hugr, HugrView, Node};

use super::DFContext;

#[derive(Debug)]
pub(super) struct DataflowContext<H: HugrView>(Arc<H>);

impl<H: HugrView> DataflowContext<H> {
    pub fn new(hugr: H) -> Self {
        Self(Arc::new(hugr))
    }
}

// Deriving Clone requires H:HugrView to implement Clone,
// but we don't need that as we only clone the Arc.
impl<H: HugrView> Clone for DataflowContext<H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<H: HugrView> Hash for DataflowContext<H> {
    fn hash<I: Hasher>(&self, state: &mut I) {}
}

impl<H: HugrView> PartialEq for DataflowContext<H> {
    fn eq(&self, other: &Self) -> bool {
        // Any AscentProgram should have only one DataflowContext
        assert_eq!(self as *const _, other as *const _);
        true
    }
}

impl<H: HugrView> Eq for DataflowContext<H> {}

impl<H: HugrView> PartialOrd for DataflowContext<H> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Any AscentProgram should have only one DataflowContext
        assert_eq!(self as *const _, other as *const _);
        Some(std::cmp::Ordering::Equal)
    }
}

impl<H: HugrView> Deref for DataflowContext<H> {
    type Target = Hugr;

    fn deref(&self) -> &Self::Target {
        self.0.base_hugr()
    }
}

impl<H: HugrView> AsRef<Hugr> for DataflowContext<H> {
    fn as_ref(&self) -> &Hugr {
        self.base_hugr()
    }
}

impl<H: HugrView> DFContext for DataflowContext<H> {}
