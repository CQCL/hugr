//! Total equality (and hence [AbstractValue] support for [Value]s
//! (by adding a source-Node and part unhashable constants)
use std::collections::hash_map::DefaultHasher; // Moves into std::hash in Rust 1.76.
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hugr_core::ops::constant::OpaqueValue;
use hugr_core::ops::Value;
use hugr_core::{Hugr, Node};
use itertools::Either;

use crate::dataflow::AbstractValue;

/// A custom constant that has been successfully hashed via [TryHash](hugr_core::ops::constant::TryHash)
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

/// A [Node] (expected to be a [Const]) and, for Sum constants, optionally,
/// indices of elements (nested arbitrarily deeply) within that.
///
/// [Const]: hugr_core::ops::Const
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodePart {
    /// The specified-index'th field of the [Sum](Value::Sum) constant identified by the RHS
    Field(usize, Box<NodePart>),
    /// The entire value produced by the node
    Node(Node),
}

impl NodePart {
    fn new(node: Node, fields: &[usize]) -> Self {
        fields
            .iter()
            .fold(Self::Node(node), |k, i| Self::Field(*i, Box::new(k)))
    }
}

/// An [Eq]-able and [Hash]-able leaf (non-[Sum](Value::Sum)) Value
#[derive(Clone, Debug)]
pub enum ValueHandle {
    /// A [Value::Extension] that has been hashed
    Hashable(HashedConst),
    /// Either a [Value::Extension] that can't be hashed, or a [Value::Function].
    Unhashable(NodePart, Either<Arc<OpaqueValue>, Arc<Hugr>>),
}

impl ValueHandle {
    /// Makes a new instance from an [OpaqueValue] given the node and (for a [Sum](Value::Sum))
    /// field indices within that (used only if the custom constant is not hashable).
    pub fn new_opaque(node: Node, fields: &[usize], val: OpaqueValue) -> Self {
        let arc = Arc::new(val);
        HashedConst::try_new(arc.clone()).map_or(
            Self::Unhashable(NodePart::new(node, fields), Either::Left(arc)),
            Self::Hashable,
        )
    }

    /// New instance for a [Value::Function] found within a node
    pub fn new_const_hugr(node: Node, fields: &[usize], val: Box<Hugr>) -> Self {
        Self::Unhashable(NodePart::new(node, fields), Either::Right(Arc::from(val)))
    }
}

impl AbstractValue for ValueHandle {}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
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
        }
    }
}

// Unfortunately we need From<ValueHandle> for Value to be able to pass
// Value's into interpret_leaf_op. So that probably doesn't make sense...
impl From<ValueHandle> for Value {
    fn from(value: ValueHandle) -> Self {
        match value {
            ValueHandle::Hashable(HashedConst { val, .. })
            | ValueHandle::Unhashable(_, Either::Left(val)) => Value::Extension {
                e: Arc::try_unwrap(val).unwrap_or_else(|a| a.as_ref().clone()),
            },
            ValueHandle::Unhashable(_, Either::Right(hugr)) => {
                Value::function(Arc::try_unwrap(hugr).unwrap_or_else(|a| a.as_ref().clone()))
                    .map_err(|e| e.to_string())
                    .unwrap()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::{ConstString, USIZE_T},
        std_extensions::{
            arithmetic::{
                float_types::{ConstF64, FLOAT64_TYPE},
                int_types::{ConstInt, INT_TYPES},
            },
            collections::ListValue,
        },
    };

    use super::*;

    #[test]
    fn value_key_eq() {
        let n = Node::from(portgraph::NodeIndex::new(0));
        let n2: Node = portgraph::NodeIndex::new(1).into();
        let h1 = ValueHandle::new_opaque(n, &[], ConstString::new("foo".to_string()).into());
        let h2 = ValueHandle::new_opaque(n2, &[], ConstString::new("foo".to_string()).into());
        let h3 = ValueHandle::new_opaque(n, &[], ConstString::new("bar".to_string()).into());

        assert_eq!(h1, h2); // Node ignored as constant is hashable
        assert_ne!(h1, h3);

        // Hashable vs Unhashable is not equal (even with same key):
        let f = ConstF64::new(std::f64::consts::PI);
        let h4 = ValueHandle::new_opaque(n, &[], f.clone().into());
        assert_ne!(h4, h1);
        assert_ne!(h1, h4);

        // Unhashable vals are compared only by key, not content
        let f2 = ConstF64::new(std::f64::consts::E);
        assert_eq!(h4, ValueHandle::new_opaque(n, &[], f2.clone().into()));
        assert_ne!(h4, ValueHandle::new_opaque(n, &[5], f2.into()));

        let h = Box::new(make_hugr(1));
        let h5 = ValueHandle::new_const_hugr(n, &[], h.clone());
        assert_eq!(
            h5,
            ValueHandle::new_const_hugr(n, &[], Box::new(make_hugr(2)))
        );
        assert_ne!(h5, ValueHandle::new_const_hugr(n2, &[], h));
    }

    fn make_hugr(num_wires: usize) -> Hugr {
        let d = DFGBuilder::new(endo_sig(vec![USIZE_T; num_wires])).unwrap();
        let inputs = d.input_wires();
        d.finish_prelude_hugr_with_outputs(inputs).unwrap()
    }

    #[test]
    fn value_key_list() {
        let v1 = ConstInt::new_u(3, 3).unwrap();
        let v2 = ConstInt::new_u(4, 3).unwrap();
        let v3 = ConstF64::new(std::f64::consts::PI);

        let n = Node::from(portgraph::NodeIndex::new(0));

        let lst = ListValue::new(INT_TYPES[0].clone(), [v1.into(), v2.into()]);
        assert_eq!(
            ValueHandle::new_opaque(n, &[], lst.clone().into()),
            ValueHandle::new_opaque(n, &[1], lst.into())
        );

        let lst = ListValue::new(FLOAT64_TYPE, [v3.into()]);
        assert_ne!(
            ValueHandle::new_opaque(n, &[], lst.clone().into()),
            ValueHandle::new_opaque(n, &[3], lst.into())
        );
    }
}
