use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use hugr_core::ops::Value;
use hugr_core::partial_value::{ValueHandle, ValueKey};
use hugr_core::{HugrView, Node};

#[derive(Clone)]
pub struct ValueCache(HashMap<ValueKey, Arc<Value>>);

impl ValueCache {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::new_bare()))
    }

    fn new_bare() -> Self {
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
    cache: Arc<Mutex<ValueCache>>,
}

impl<'a, H> DataflowContext<'a, H> {
    fn new(hugr: &'a H, cache: Arc<Mutex<ValueCache>>) -> Self {
        Self {
            id: next_context_id(),
            hugr,
            cache,
        }
    }

    pub fn get_value_handle(&self, key: impl Into<ValueKey>, value: &Value) -> ValueHandle {
        let mut guard = self.cache.lock().unwrap();
        guard.get(key.into(), value)
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

pub struct ArcDataflowContext<'a, H>(Arc<DataflowContext<'a, H>>);

impl<'a, H> ArcDataflowContext<'a, H> {
    pub fn new(h: &'a H, cache: Arc<Mutex<ValueCache>>) -> Self {
        Self(Arc::new(DataflowContext::new(h, cache)))
    }
}

impl<'a, H> Clone for ArcDataflowContext<'a, H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, H> Hash for ArcDataflowContext<'a, H> {
    fn hash<HA: Hasher>(&self, state: &mut HA) {
        self.0.hash(state);
    }
}

impl<'a, H> PartialEq for ArcDataflowContext<'a, H> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<'a, H> Eq for ArcDataflowContext<'a, H> {}

impl<'a, H> Deref for ArcDataflowContext<'a, H> {
    type Target = DataflowContext<'a, H>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait DFContext: Clone + Eq + Hash {
    type H: HugrView;
    fn hugr(&self) -> &Self::H;
    fn node_value_handle(&self, const_node: Node, value: &Value) -> ValueHandle;
}

impl<'a, H: HugrView> DFContext for ArcDataflowContext<'a, H> {
    type H = H;
    fn hugr(&self) -> &Self::H {
        self.0.hugr
    }

    fn node_value_handle(&self, const_node: Node, value: &Value) -> ValueHandle {
        self.0.get_value_handle(const_node, value)
    }
}
