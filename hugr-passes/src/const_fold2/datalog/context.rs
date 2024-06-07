use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use ascent::Lattice;

use either::Either;
use hugr_core::ops::{OpTag, OpTrait, Value};
use hugr_core::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

#[derive(Clone, Debug)]
pub struct ValueHandle(Vec<usize>, Node, Arc<Value>);

impl ValueHandle {
    pub fn value(&self) -> &Value {
        self.2.as_ref()
    }

    pub fn tag(&self) -> usize {
        match self.value() {
            Value::Sum { tag, .. } => *tag,
            Value::Tuple {  .. } => 0,
            _ => panic!("ValueHandle::tag called on non-Sum, non-Tuple value"),
        }
    }

    pub fn index(self: &ValueHandle, i: usize) -> ValueHandle {
        let vs = match self.value() {
            Value::Sum { values, .. } => values,
            Value::Tuple { vs, .. } => vs,
            _ => panic!("ValueHandle::index called on non-Sum, non-Tuple value"),
        };
        assert!(i < vs.len());
        let v = vs[i].clone().into();
        let mut is = self.0.clone();
        is.push(i);
        Self(is, self.1, v)
    }
}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        (&self.0, self.1) == (&other.0, other.1)
    }
}

impl Eq for ValueHandle {}

impl Hash for ValueHandle {
    fn hash<I: Hasher>(&self, state: &mut I) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl Deref for ValueHandle {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

#[derive(Clone)]
pub struct ValueCache(HashMap<Node, Arc<Value>>);

impl ValueCache {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&mut self, node: Node, value: &Value) -> ValueHandle {
        let v = self.0.entry(node).or_insert_with(|| value.clone().into()).clone();
        ValueHandle(vec![], node, v)
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

    pub fn value_handle(&self, node: Node, value: &Value) -> ValueHandle {
        self.cache.borrow_mut().get(node, value)
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

impl<'a,H> Deref for DataflowContext<'a,H> {
    type Target = H;

    fn deref(&self) -> &Self::Target {
        self.hugr
    }
}
