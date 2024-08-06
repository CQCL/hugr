use std::any::Any;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::Downcast;
use itertools::Either;

use crate::ops::Value;
use crate::std_extensions::arithmetic::int_types::ConstInt;
use crate::Node;

pub trait ValueName: std::fmt::Debug + Downcast + Any {
    fn hash(&self) -> u64;
    fn eq(&self, other: &dyn ValueName) -> bool;
}

fn hash_hash(x: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

fn value_name_eq<T: Eq + Downcast + Any>(x: &T, other: &dyn ValueName) -> bool {
    if let Some(other) = other.as_any().downcast_ref::<T>() {
        x == other
    } else {
        false
    }
}

impl ValueName for String {
    fn hash(&self) -> u64 {
        hash_hash(self)
    }

    fn eq(&self, other: &dyn ValueName) -> bool {
        value_name_eq(self, other)
    }
}

impl ValueName for ConstInt {
    fn hash(&self) -> u64 {
        hash_hash(self)
    }

    fn eq(&self, other: &dyn ValueName) -> bool {
        value_name_eq(self, other)
    }
}

#[derive(Clone, Debug)]
pub struct ValueKey(Vec<usize>, Either<Node, Arc<dyn ValueName>>);

impl PartialEq for ValueKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            && match (&self.1, &other.1) {
                (Either::Left(ref n1), Either::Left(ref n2)) => n1 == n2,
                (Either::Right(ref v1), Either::Right(ref v2)) => v1.eq(v2.as_ref()),
                _ => false,
            }
    }
}

impl Eq for ValueKey {}

impl Hash for ValueKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        match &self.1 {
            Either::Left(n) => (0, n).hash(state),
            Either::Right(v) => (1, v.hash()).hash(state),
        }
    }
}

impl From<Node> for ValueKey {
    fn from(n: Node) -> Self {
        Self(vec![], Either::Left(n))
    }
}

impl ValueKey {
    pub fn new(k: impl ValueName) -> Self {
        Self(vec![], Either::Right(Arc::new(k)))
    }

    pub fn index(self, i: usize) -> Self {
        let mut is = self.0;
        is.push(i);
        Self(is, self.1)
    }
}

#[derive(Clone, Debug)]
pub struct ValueHandle(ValueKey, Arc<Value>);

impl ValueHandle {
    pub fn new(key: ValueKey, value: Arc<Value>) -> Self {
        Self(key, value)
    }

    pub fn value(&self) -> &Value {
        self.1.as_ref()
    }

    pub fn is_compound(&self) -> bool {
        match self.value() {
            Value::Sum { .. } | Value::Tuple { .. } => true,
            _ => false,
        }
    }

    pub fn num_fields(&self) -> usize {
        assert!(
            self.is_compound(),
            "ValueHandle::num_fields called on non-Sum, non-Tuple value: {:#?}",
            self
        );
        match self.value() {
            Value::Sum { values, .. } => values.len(),
            Value::Tuple { vs } => vs.len(),
            _ => unreachable!(),
        }
    }

    pub fn tag(&self) -> usize {
        assert!(
            self.is_compound(),
            "ValueHandle::tag called on non-Sum, non-Tuple value: {:#?}",
            self
        );
        match self.value() {
            Value::Sum { tag, .. } => *tag,
            Value::Tuple { .. } => 0,
            _ => unreachable!(),
        }
    }

    pub fn index(self: &ValueHandle, i: usize) -> ValueHandle {
        assert!(
            i < self.num_fields(),
            "ValueHandle::index called with out-of-bounds index {}: {:#?}",
            i,
            &self
        );
        let vs = match self.value() {
            Value::Sum { values, .. } => values,
            Value::Tuple { vs, .. } => vs,
            _ => unreachable!(),
        };
        let v = vs[i].clone().into();
        Self(self.0.clone().index(i), v)
    }
}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        // If the keys are equal, we return true since the values must have the
        // same provenance, and so be equal. If the keys are different but the
        // values are equal, we could return true if we didn't impl Eq, but
        // since we do impl Eq, the Hash contract prohibits us from having equal
        // values with different hashes.
        let r = self.0 == other.0;
        if r {
            debug_assert_eq!(self.get_type(), other.get_type());
        }
        r
    }
}

impl Eq for ValueHandle {}

impl Hash for ValueHandle {
    fn hash<I: Hasher>(&self, state: &mut I) {
        self.0.hash(state);
    }
}

/// TODO this is perhaps dodgy
/// we do not hash or compare the value, just the key
/// this means two handles with different keys, but with the same value, will
/// not compare equal.
impl Deref for ValueHandle {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        self.value()
    }
}

#[cfg(test)]
mod test {
    use crate::{ops::constant::CustomConst as _, types::SumType};

    use super::*;

    #[test]
    fn value_key_eq() {
        let k1 = ValueKey::new("foo".to_string());
        let k2 = ValueKey::new("foo".to_string());
        let k3 = ValueKey::new("bar".to_string());

        assert_eq!(k1, k2);
        assert_ne!(k1, k3);

        let k4: ValueKey = From::<Node>::from(portgraph::NodeIndex::new(1).into());
        let k5 = From::<Node>::from(portgraph::NodeIndex::new(1).into());
        let k6 = From::<Node>::from(portgraph::NodeIndex::new(2).into());

        assert_eq!(&k4, &k5);
        assert_ne!(&k4, &k6);

        let k7 = k5.clone().index(3);
        let k4 = k4.index(3);

        assert_eq!(&k4, &k7);

        let k5 = k5.index(2);

        assert_ne!(&k5, &k7);
    }

    #[test]
    fn value_handle_eq() {
        let k_i = ConstInt::new_u(4, 2).unwrap();
        let subject_val = Arc::new(
            Value::sum(
                0,
                [k_i.clone().into()],
                SumType::new([vec![k_i.get_type()], vec![]]),
            )
            .unwrap(),
        );

        let k1 = ValueKey::new("foo".to_string());
        let v1 = ValueHandle::new(k1.clone(), subject_val.clone());
        let v2 = ValueHandle::new(k1.clone(), Value::extension(k_i).into());

        // we do not compare the value, just the key
        assert_ne!(v1.index(0), v2);
        assert_eq!(v1.index(0).value(), v2.value());
    }
}
