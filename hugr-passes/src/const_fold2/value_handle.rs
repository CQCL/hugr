use std::collections::hash_map::DefaultHasher; // Moves into std::hash in Rust 1.76.
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::Value;
use hugr_core::types::TypeArg;
use hugr_core::{Hugr, Node};
use itertools::Either;

use crate::dataflow::AbstractValue;

#[derive(Clone, Debug)]
pub struct HashedConst {
    hash: u64,
    pub(super) val: Arc<OpaqueValue>,
}

impl HashedConst {
    pub(super) fn try_new(val: Arc<OpaqueValue>) -> Option<Self> {
        let mut hasher = DefaultHasher::new();
        val.value().try_hash(&mut hasher).then(|| HashedConst {
            hash: hasher.finish(),
            val,
        })
    }
}

impl PartialEq for HashedConst {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.val.value().equal_consts(other.val.value())
    }
}

impl Eq for HashedConst {}

impl Hash for HashedConst {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NodePart {
    Field(usize, Box<NodePart>),
    Node(Node),
}

impl NodePart {
    fn new(node: Node, fields: &[usize]) -> Self {
        fields
            .iter()
            .fold(Self::Node(node), |k, i| Self::Field(*i, Box::new(k)))
    }
}

#[derive(Clone, Debug)]
pub enum ValueHandle {
    Hashable(HashedConst),
    Unhashable(NodePart, Either<Arc<OpaqueValue>, Arc<Hugr>>),
    Function(Node, Vec<TypeArg>),
}

impl ValueHandle {
    pub fn new_opaque(node: Node, fields: &[usize], val: OpaqueValue) -> Self {
        let arc = Arc::new(val);
        HashedConst::try_new(arc.clone()).map_or(
            Self::Unhashable(NodePart::new(node, fields), Either::Left(arc)),
            Self::Hashable,
        )
    }

    pub fn new_const_hugr(node: Node, fields: &[usize], val: Box<Hugr>) -> Self {
        Self::Unhashable(NodePart::new(node, fields), Either::Right(Arc::from(val)))
    }

    pub fn new_function(node: Node, type_args: impl IntoIterator<Item = TypeArg>) -> Self {
        Self::Function(node, type_args.into_iter().collect())
    }
}

impl AbstractValue for ValueHandle {}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Function(n1, args1), Self::Function(n2, args2)) => n1 == n2 && args1 == args2,
            (Self::Hashable(h1), Self::Hashable(h2)) => h1 == h2,
            (Self::Unhashable(k1, _), Self::Unhashable(k2, _)) => {
                // If the keys are equal, we return true since the values must have the
                // same provenance, and so be equal. If the keys are different but the
                // values are equal, we could return true if we didn't impl Eq, but
                // since we do impl Eq, the Hash contract prohibits us from having equal
                // values with different hashes.
                k1 == k2
            }
            _ => false,
        }
    }
}

impl Eq for ValueHandle {}

impl Hash for ValueHandle {
    fn hash<I: Hasher>(&self, state: &mut I) {
        match self {
            ValueHandle::Hashable(hc) => hc.hash(state),
            ValueHandle::Unhashable(key, _) => key.hash(state),
            ValueHandle::Function(node, vec) => {
                node.hash(state);
                vec.hash(state);
            }
        }
    }
}

// Unfortunately we need From<ValueHandle> for Value to be able to pass
// Value's into interpret_leaf_op. So that probably doesn't make sense...
impl TryFrom<ValueHandle> for Value {
    type Error = String;
    fn try_from(value: ValueHandle) -> Result<Self, Self::Error> {
        Ok(match value {
            ValueHandle::Hashable(HashedConst { val, .. })
            | ValueHandle::Unhashable(_, Either::Left(val)) => Value::Extension {
                e: Arc::unwrap_or_clone(val),
            },
            ValueHandle::Unhashable(_, Either::Right(hugr)) => {
                Value::function(Arc::unwrap_or_clone(hugr)).map_err(|e| e.to_string())?
            }
            ValueHandle::Function(node, _type_args) => {
                return Err(format!(
                    "Function defined externally ({}) cannot be turned into Value",
                    node
                ))
            }
        })
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
    use itertools::Itertools;

    use super::*;

    #[test]
    fn value_key_eq() {
        let n = Node::from(portgraph::NodeIndex::new(0));
        let n2: Node = portgraph::NodeIndex::new(1).into();
        let k1 = ValueHandle::new_opaque(n, &[], ConstString::new("foo".to_string()).into());
        let k2 = ValueHandle::new_opaque(n2, &[], ConstString::new("foo".to_string()).into());
        let k3 = ValueHandle::new_opaque(n, &[], ConstString::new("bar".to_string()).into());

        assert_eq!(k1, k2); // Node ignored as constant is hashable
        assert_ne!(k1, k3);

        // Hashable vs Unhashable is not equal (even with same key):
        let f = ConstF64::new(std::f64::consts::PI);
        assert_ne!(ValueHandle::new_opaque(n, &[], f.into()), k1);
        assert_ne!(k1, ValueHandle::new_opaque(n, &[], f.into()));

        // Unhashable vals are compared only by key, not content
        let f2 = ConstF64::new(std::f64::consts::E);
        assert_eq!(
            ValueKey::new_opaque(n, &[], f),
            ValueKey::new_opaque(n, &[], f2)
        );
        assert_ne!(
            ValueKey::new_opaque(n, &[], f),
            ValueKey::new_opaque(n2, &[], f)
        );

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
        let v3 = ConstF64::new(std::f64::consts::PI);

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
        let st = SumType::new([vec![k_i.get_type()], vec![]]);
        let subject_val = Value::sum(0, [k_i.clone().into()], st).unwrap();

        let k1 = ValueKey::try_new(ConstString::new("foo".to_string())).unwrap();
        let PartialValue::PartialSum(ps1) = ValueHandle::new(k1.clone(), subject_val.clone())
        else {
            panic!()
        };
        let (_tag, fields) = ps1.0.into_iter().exactly_one().unwrap();
        let PartialValue::Value(vh1) = fields.into_iter().exactly_one().unwrap() else {
            panic!()
        };

        let PartialValue::Value(v2) = ValueHandle::new(k1.clone(), Value::extension(k_i).into())
        else {
            panic!()
        };

        // we do not compare the value, just the key
        assert_ne!(vh1, v2);
        assert_eq!(vh1.1, v2.1);
    }
}
