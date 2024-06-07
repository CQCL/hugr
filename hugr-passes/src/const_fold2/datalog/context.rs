use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use ascent::Lattice;

use either::Either;
use hugr_core::ops::{OpTag, OpTrait, Value};
use hugr_core::partial_value::{ValueHandle, ValueKey};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

#[derive(Clone)]
pub struct ValueCache(HashMap<ValueKey, Arc<Value>>);

impl ValueCache {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&mut self, key: ValueKey, value: &Value) -> ValueHandle {
        let v = self
            .0
            .entry(key.clone())
            .or_insert_with(|| value.clone().into())
            .clone();
        ValueHandle::new(key, v)
    }
}

static mut CONTEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn next_context_id() -> usize {
    unsafe { CONTEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst) }
}

pub struct DataflowContext<'a, H> {
    id: usize,
    hugr: &'a H,
    cache: RefCell<ValueCache>,
}

impl<'a, H> DataflowContext<'a, H> {
    pub fn new(hugr: &'a H) -> Arc<Self> {
        Arc::new(Self {
            id: next_context_id(),
            hugr,
            cache: ValueCache::new().into(),
        })
    }

    pub fn node_value_handle(&self, node: Node, value: &Value) -> ValueHandle {
        self.cache.borrow_mut().get(node.into(), value)
    }

    pub fn hugr(&self) -> &'a H {
        self.hugr
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

impl<H> std::fmt::Debug for DataflowContext<'_, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DataflowContext({})", self.id)
    }
}

impl<H> Hash for DataflowContext<'_, H> {
    fn hash<I: Hasher>(&self, state: &mut I) {
        self.id.hash(state);
    }
}

impl<H> PartialEq<usize> for DataflowContext<'_, H> {
    fn eq(&self, other: &usize) -> bool {
        &self.id == other
    }
}

impl<H> PartialEq for DataflowContext<'_, H> {
    fn eq(&self, other: &Self) -> bool {
        self == &other.id
    }
}

impl<H> Eq for DataflowContext<'_, H> {}

impl<H> PartialOrd for DataflowContext<'_, H> {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&_other.id)
    }
}

impl<'a, H> Deref for DataflowContext<'a, H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        self.hugr
    }
}
