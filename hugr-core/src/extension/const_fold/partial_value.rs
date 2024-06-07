use std::sync::Arc;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::collections::HashMap;

use itertools::{zip_eq, Itertools as _};

use crate::ops::{OpTag, OpTrait, Value};
use crate::types::{Type, TypeEnum};
use crate::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

#[derive(Clone, Debug)]
pub struct ValueHandle(Vec<usize>, Node, Arc<Value>);

impl ValueHandle {
    pub fn new(node: Node, value: Arc<Value>) -> Self {
        Self(vec![], node, value)
    }

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

/// TODO this is dodgy
/// we do not hash or compare the value, just the key
/// this means two handles with different keys, but with the same value, will
/// not compare equal.
impl Deref for ValueHandle {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

/// TODO shouldn't be pub
#[derive(PartialEq, Clone, Eq, Debug)]
pub struct HashableHashMap<K: Hash + std::cmp::Eq, V>(HashMap<K, V>);

impl<K: Hash + std::cmp::Eq, V: Hash> Hash for HashableHashMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.keys().for_each(|k| k.hash(state));
        self.0.values().for_each(|v| v.hash(state));
    }
}

#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue {
    Bottom,
    Value(ValueHandle),
    PartialSum(HashableHashMap<usize, Vec<PartialValue>>),
    Top,
}

impl From<ValueHandle> for PartialValue {
    fn from(v: ValueHandle) -> Self {
        match v.value() {
            Value::Tuple { vs } => {
                let vec = (0..vs.len()).map(|i| PartialValue::from(v.index(i)).into()).collect();
                Self::PartialSum(HashableHashMap([(0, vec)].into_iter().collect()))
            }
            Value::Sum { tag, values, .. } => {
                let vec = (0..values.len()).map(|i| PartialValue::from(v.index(i)).into()).collect();
                Self::PartialSum(HashableHashMap([(*tag, vec)].into_iter().collect()))
            }
            _ => Self::Value(v)
        }
    }
}

impl PartialValue {
    const BOTTOM: Self = Self::Bottom;
    const BOTTOM_REF: &'static Self = &Self::BOTTOM;

    fn initialised(&self) -> bool {
        !self.is_top()
    }

    fn is_top(&self) -> bool { self == &PartialValue::Top }


    /// TODO docs
    /// just delegate to variant_field_value
    pub fn tuple_field_value(&self, idx: usize) -> Self {
        self.variant_field_value(0,idx)
    }

    /// TODO docs
    pub fn variant_field_value(&self, variant: usize, idx: usize) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::PartialSum(HashableHashMap(hm)) => {
                if let Some(row) = hm.get(&variant) {
                    assert!(row.len() > idx);
                    row[idx].clone()
                } else {
                    // We must return top. if self were to gain this variant, we would return the element of that variant.
                    // We must ensure that the value return now is <= that future value
                    Self::Top
                }
            },
            Self::Value(v) if v.tag() == variant => {
                Self::Value(v.index(idx))
            },
            _ => Self::Top
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<Value,Self> {
        let r = match self {
            Self::Value(v) => v.value().clone(),
            Self::PartialSum(HashableHashMap(hm)) => {
                let err = |hm| Err(Self::PartialSum(HashableHashMap(hm)));
                let Ok((k,v)) = hm.iter().exactly_one() else {
                    return err(hm);
                };
                let TypeEnum::Sum(st) = typ.as_type_enum() else {
                    return err(hm);
                };
                let Some(r) = st.get_variant(*k) else {
                    return err(hm);
                };
                if v.len() != r.len() {
                    return err(hm);
                }

                let Ok(vs) = zip_eq(v.into_iter(), r.into_iter()).map(|(v,t)| v.clone().try_into_value(t)).collect::<Result<Vec<_>, _>>() else {
                    return err(hm);
                };

                Value::sum(*k, vs, st.clone()).map_err(|_| Self::PartialSum(HashableHashMap(hm)))?
            },
            x => Err(x)?

        };
        assert_eq!(typ, &r.get_type());
        Ok(r)
    }

    fn join_value_handle(&mut self, vh: ValueHandle) -> bool {
        let mut new_self = self;
        match &mut new_self {
            Self::Bottom => {
                false
            },
            s@Self::Value(_) => {
                let Self::Value(v) = *s else { unreachable!() };
                if v == &vh {
                    false
                } else {
                    **s = Self::Bottom;
                    true
                }
            },
            s@Self::PartialSum(_) => {
                match vh.into() {
                    Self::Value(_) => {
                        **s = Self::Bottom;
                        true
                    }
                    other => s.join_mut(other)
                }
            },
            s@Self::Top => {
                **s = vh.into();
                true
            }
        }
    }

    pub fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    pub fn join_mut(&mut self, other: Self) -> bool {
        // println!("join {self:?}\n{:?}", &other);
        let mut s = self;
        let changed = match (&mut s, other) {
            (Self::Bottom, _) => false,
            (s, other@Self::Bottom) => {
                **s = other;
                true
            },
            (_, Self::Top) => false,
            (s@Self::Top, other) => {
                **s = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) if h1 == &h2 || h1.value() == h2.value() => {
                false
            }
            (s@Self::PartialSum(_), Self::PartialSum(HashableHashMap(hm2))) => {
                let mut changed = false;
                let Self::PartialSum(HashableHashMap(hm1)) = *s else { unreachable!() };
                for (k, v) in hm2 {
                    if let Some(row) = hm1.get_mut(&k) {
                        for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                            changed |= lhs.join_mut(rhs);
                        }
                    } else {
                        hm1.insert(k, v);
                        changed = true;
                    }
                }
                changed
            }
            (s@Self::Value(_), other@Self::PartialSum(_)) => {
                let mut old_self = other;
                std::mem::swap(*s, &mut old_self);
                let Self::Value(h) = old_self else { unreachable!() };
                s.join_value_handle(h)
            }
            (s@Self::PartialSum(_), Self::Value(h)) => {
                s.join_value_handle(h)
            }
            (s,_) => {
                **s = Self::Bottom;
                false
            }
        };
        // if changed {
            // println!("join new self: {:?}", s);
        // }
        changed
    }

    pub fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other) ;
        self
    }

    pub fn meet_mut(&mut self, _other: Self) -> bool {
        todo!()
    }

    pub fn top() -> Self {
        Self::Top
    }

    pub fn bottom() -> Self {
        Self::Bottom
    }

    pub fn variant(tag: usize, values: impl IntoIterator<Item=Self>) -> Self {
        Self::PartialSum(HashableHashMap([(tag, values.into_iter().collect())].into_iter().collect()))
    }
}

impl PartialOrd for PartialValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // TODO we can do better
        (self == other).then_some(std::cmp::Ordering::Equal)
    }
}
