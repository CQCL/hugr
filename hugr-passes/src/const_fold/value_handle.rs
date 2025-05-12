//! Total equality (and hence [`AbstractValue`] support for [Value]s
//! (by adding a source-Node and part unhashable constants)
use std::collections::hash_map::DefaultHasher; // Moves into std::hash in Rust 1.76.
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hugr_core::core::HugrNode;
use hugr_core::ops::Value;
use hugr_core::ops::constant::OpaqueValue;
use hugr_core::types::ConstTypeError;
use hugr_core::{Hugr, Node};
use itertools::Either;

use crate::dataflow::{AbstractValue, AsConcrete, ConstLocation, LoadedFunction, Sum};

/// A custom constant that has been successfully hashed via [`TryHash`](hugr_core::ops::constant::TryHash)
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

/// An [Eq]-able and [Hash]-able leaf (non-[Sum](Value::Sum)) Value
#[derive(Clone, Debug)]
pub enum ValueHandle<N = Node> {
    /// A [`Value::Extension`] that has been hashed
    Hashable(HashedConst),
    /// Either a [`Value::Extension`] that can't be hashed, or a [`Value::Function`].
    Unhashable {
        /// The node (i.e. a [Const](hugr_core::ops::Const)) containing the constant
        node: N,
        /// Indices within [`Value::Sum`]s containing the unhashable [`Self::Unhashable::leaf`]
        fields: Vec<usize>,
        /// The unhashable [`Value::Extension`] or [`Value::Function`]
        leaf: Either<Arc<OpaqueValue>, Arc<Hugr>>,
    },
}

fn node_and_fields<N: HugrNode>(loc: &ConstLocation<N>) -> (N, Vec<usize>) {
    match loc {
        ConstLocation::Node(n) => (*n, vec![]),
        ConstLocation::Field(idx, elem) => {
            let (n, mut f) = node_and_fields(elem);
            f.push(*idx);
            (n, f)
        }
    }
}

impl<N: HugrNode> ValueHandle<N> {
    /// Makes a new instance from an [`OpaqueValue`] given the node and (for a [Sum](Value::Sum))
    /// field indices within that (used only if the custom constant is not hashable).
    pub fn new_opaque<'a>(loc: impl Into<ConstLocation<'a, N>>, val: OpaqueValue) -> Self
    where
        N: 'a,
    {
        let arc = Arc::new(val);
        let (node, fields) = node_and_fields(&loc.into());
        HashedConst::try_new(arc.clone()).map_or(
            Self::Unhashable {
                node,
                fields,
                leaf: Either::Left(arc),
            },
            Self::Hashable,
        )
    }

    /// New instance for a [`Value::Function`] found within a node
    pub fn new_const_hugr<'a>(loc: impl Into<ConstLocation<'a, N>>, val: Box<Hugr>) -> Self
    where
        N: 'a,
    {
        let (node, fields) = node_and_fields(&loc.into());
        Self::Unhashable {
            node,
            fields,
            leaf: Either::Right(Arc::from(val)),
        }
    }
}

impl<N: HugrNode> AbstractValue for ValueHandle<N> {}

impl<N: HugrNode> PartialEq for ValueHandle<N> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Hashable(h1), Self::Hashable(h2)) => h1 == h2,
            (
                Self::Unhashable {
                    node: n1,
                    fields: f1,
                    leaf: _,
                },
                Self::Unhashable {
                    node: n2,
                    fields: f2,
                    leaf: _,
                },
            ) => {
                // If the keys are equal, we return true since the values must have the
                // same provenance, and so be equal. If the keys are different but the
                // values are equal, we could return true if we didn't impl Eq, but
                // since we do impl Eq, the Hash contract prohibits us from having equal
                // values with different hashes.
                n1 == n2 && f1 == f2
            }
            _ => false,
        }
    }
}

impl<N: HugrNode> Eq for ValueHandle<N> {}

impl<N: HugrNode> Hash for ValueHandle<N> {
    fn hash<I: Hasher>(&self, state: &mut I) {
        match self {
            ValueHandle::Hashable(hc) => hc.hash(state),
            ValueHandle::Unhashable {
                node,
                fields,
                leaf: _,
            } => {
                node.hash(state);
                fields.hash(state);
            }
        }
    }
}

// Unfortunately we need From<ValueHandle> for Value to be able to pass
// Value's into interpret_leaf_op. So that probably doesn't make sense...
impl<N: HugrNode> AsConcrete<ValueHandle<N>, N> for Value {
    type ValErr = Infallible;
    type SumErr = ConstTypeError;

    fn from_value(value: ValueHandle<N>) -> Result<Self, Infallible> {
        Ok(match value {
            ValueHandle::Hashable(HashedConst { val, .. })
            | ValueHandle::Unhashable {
                leaf: Either::Left(val),
                ..
            } => Value::Extension {
                e: Arc::try_unwrap(val).unwrap_or_else(|a| a.as_ref().clone()),
            },
            ValueHandle::Unhashable {
                leaf: Either::Right(hugr),
                ..
            } => Value::function(Arc::try_unwrap(hugr).unwrap_or_else(|a| a.as_ref().clone()))
                .map_err(|e| e.to_string())
                .unwrap(),
        })
    }

    fn from_sum(value: Sum<Self>) -> Result<Self, Self::SumErr> {
        Self::sum(value.tag, value.values, value.st)
    }

    fn from_func(func: LoadedFunction<N>) -> Result<Self, crate::dataflow::LoadedFunction<N>> {
        Err(func)
    }
}

#[cfg(test)]
mod test {
    use hugr_core::{
        builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig},
        extension::prelude::{ConstString, usize_t},
        std_extensions::{
            arithmetic::{
                float_types::{ConstF64, float64_type},
                int_types::{ConstInt, INT_TYPES},
            },
            collections::list::ListValue,
        },
    };

    use super::*;

    #[test]
    fn value_key_eq() {
        let n = Node::from(portgraph::NodeIndex::new(0));
        let n2: Node = portgraph::NodeIndex::new(1).into();
        let h1 = ValueHandle::new_opaque(n, ConstString::new("foo".to_string()).into());
        let h2 = ValueHandle::new_opaque(n2, ConstString::new("foo".to_string()).into());
        let h3 = ValueHandle::new_opaque(n, ConstString::new("bar".to_string()).into());

        assert_eq!(h1, h2); // Node ignored as constant is hashable
        assert_ne!(h1, h3);

        // Hashable vs Unhashable is not equal (even with same key):
        let f = ConstF64::new(std::f64::consts::PI);
        let h4 = ValueHandle::new_opaque(n, f.clone().into());
        assert_ne!(h4, h1);
        assert_ne!(h1, h4);

        // Unhashable vals are compared only by key, not content
        let f2 = ConstF64::new(std::f64::consts::E);
        assert_eq!(h4, ValueHandle::new_opaque(n, f2.clone().into()));
        assert_ne!(
            h4,
            ValueHandle::new_opaque(ConstLocation::Field(5, &n.into()), f2.into())
        );

        let h = Box::new(make_hugr(1));
        let h5 = ValueHandle::new_const_hugr(n, h.clone());
        assert_eq!(h5, ValueHandle::new_const_hugr(n, Box::new(make_hugr(2))));
        assert_ne!(h5, ValueHandle::new_const_hugr(n2, h));
    }

    fn make_hugr(num_wires: usize) -> Hugr {
        let d = DFGBuilder::new(endo_sig(vec![usize_t(); num_wires])).unwrap();
        let inputs = d.input_wires();
        d.finish_hugr_with_outputs(inputs).unwrap()
    }

    #[test]
    fn value_key_list() {
        let v1 = ConstInt::new_u(3, 3).unwrap();
        let v2 = ConstInt::new_u(4, 3).unwrap();
        let v3 = ConstF64::new(std::f64::consts::PI);

        let n = Node::from(portgraph::NodeIndex::new(0));

        let lst = ListValue::new(INT_TYPES[0].clone(), [v1.into(), v2.into()]);
        assert_eq!(
            ValueHandle::new_opaque(n, lst.clone().into()),
            ValueHandle::new_opaque(ConstLocation::Field(1, &n.into()), lst.into())
        );

        let lst = ListValue::new(float64_type(), [v3.into()]);
        assert_ne!(
            ValueHandle::new_opaque(n, lst.clone().into()),
            ValueHandle::new_opaque(ConstLocation::Field(3, &n.into()), lst.into())
        );
    }
}
