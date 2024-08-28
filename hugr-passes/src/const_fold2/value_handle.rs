use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::ops::Value;
use hugr_core::types::Type;
use hugr_core::Node;

use super::partial_value::{AbstractValue, PartialSum, PartialValue};

#[derive(Clone, Debug)]
pub struct HashedConst {
    hash: u64,
    val: Arc<dyn CustomConst>,
}

impl PartialEq for HashedConst {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.val.equal_consts(other.val.as_ref())
    }
}

impl Eq for HashedConst {}

impl Hash for HashedConst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueKey {
    Field(usize, Box<ValueKey>),
    Const(HashedConst),
    Node(Node),
}

impl From<Node> for ValueKey {
    fn from(n: Node) -> Self {
        Self::Node(n)
    }
}

impl From<HashedConst> for ValueKey {
    fn from(value: HashedConst) -> Self {
        Self::Const(value)
    }
}

impl ValueKey {
    pub fn new(n: Node, k: impl CustomConst) -> Self {
        Self::try_new(k).unwrap_or(Self::Node(n))
    }

    pub fn try_new(cst: impl CustomConst) -> Option<Self> {
        let mut hasher = DefaultHasher::new();
        cst.maybe_hash(&mut hasher).then(|| {
            Self::Const(HashedConst {
                hash: hasher.finish(),
                val: Arc::new(cst),
            })
        })
    }

    pub fn field(self, i: usize) -> Self {
        Self::Field(i, Box::new(self))
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

    pub fn get_type(&self) -> Type {
        self.1.get_type()
    }
}

impl AbstractValue for ValueHandle {
    fn as_sum(&self) -> Option<(usize, impl Iterator<Item = Self> + '_)> {
        match self.value() {
            Value::Sum(Sum { tag, values, .. }) => Some((
                *tag,
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| Self(self.0.clone().field(i), Arc::new(v.clone()))),
            )),
            _ => None,
        }
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

impl From<ValueHandle> for Value {
    fn from(value: ValueHandle) -> Self {
        (*value.1).clone()
    }
}

#[cfg(test)]
mod test {
    use hugr_core::{
        extension::prelude::ConstString,
        ops::constant::CustomConst as _,
        std_extensions::{
            arithmetic::{
                float_types::{ConstF64, FLOAT64_TYPE},
                int_types::{ConstInt, INT_TYPES},
            },
            collections::ListValue,
        },
        types::SumType,
    };

    use super::*;

    #[test]
    fn value_key_eq() {
        let n = Node::from(portgraph::NodeIndex::new(0));
        let n2: Node = portgraph::NodeIndex::new(1).into();
        let k1 = ValueKey::new(n, ConstString::new("foo".to_string()));
        let k2 = ValueKey::new(n2, ConstString::new("foo".to_string()));
        let k3 = ValueKey::new(n, ConstString::new("bar".to_string()));

        assert_eq!(k1, k2); // Node ignored
        assert_ne!(k1, k3);

        assert_eq!(ValueKey::from(n), ValueKey::from(n));
        let f = ConstF64::new(3.141);
        assert_eq!(ValueKey::new(n, f.clone()), ValueKey::from(n));

        assert_ne!(ValueKey::new(n, f.clone()), ValueKey::new(n2, f)); // Node taken into account
        let k4 = ValueKey::from(n);
        let k5 = ValueKey::from(n);
        let k6: ValueKey = ValueKey::from(n2);

        assert_eq!(&k4, &k5);
        assert_ne!(&k4, &k6);

        let k7 = k5.clone().field(3);
        let k4 = k4.field(3);

        assert_eq!(&k4, &k7);

        let k5 = k5.field(2);

        assert_ne!(&k5, &k7);
    }

    #[test]
    fn value_key_list() {
        let v1 = ConstInt::new_u(3, 3).unwrap();
        let v2 = ConstInt::new_u(4, 3).unwrap();
        let v3 = ConstF64::new(3.141);

        let n = Node::from(portgraph::NodeIndex::new(0));
        let n2: Node = portgraph::NodeIndex::new(1).into();

        let lst = ListValue::new(INT_TYPES[0].clone(), [v1.into(), v2.into()]);
        assert_eq!(ValueKey::new(n, lst.clone()), ValueKey::new(n2, lst));

        let lst = ListValue::new(FLOAT64_TYPE, [v3.into()]);
        assert_ne!(
            ValueKey::new(n, lst.clone()),
            ValueKey::new(n2, lst.clone())
        );
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

        let k1 = ValueKey::try_new(ConstString::new("foo".to_string())).unwrap();
        let v1 = ValueHandle::new(k1.clone(), subject_val.clone());
        let v2 = ValueHandle::new(k1.clone(), Value::extension(k_i).into());

        let fields = v1.as_sum().unwrap().1.collect::<Vec<_>>();
        // we do not compare the value, just the key
        assert_ne!(fields[0], v2);
        assert_eq!(fields[0].value(), v2.value());
    }
}
