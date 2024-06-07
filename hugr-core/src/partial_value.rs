#![allow(missing_docs)]
use std::any::Any;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use downcast_rs::Downcast;
use itertools::{zip_eq, Either, Itertools as _};

use crate::ops::{OpTag, OpTrait, Value};
use crate::types::{Type, TypeEnum};
use crate::{Hugr, HugrView, IncomingPort, Node, OutgoingPort, PortIndex as _, Wire};

pub trait ValueName: std::fmt::Debug + Downcast + Any {
    fn hash(&self) -> u64;
    fn eq(&self, other: &dyn ValueName) -> bool;
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
            Either::Left(n) => n.hash(state),
            Either::Right(v) => state.write_u64(v.hash()),
        }
    }
}

impl ValueName for String {
    fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        <Self as Hash>::hash(self, &mut hasher);
        hasher.finish()
    }

    fn eq(&self, other: &dyn ValueName) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<Self>() {
            self == other
        } else {
            false
        }
    }
}

impl From<Node> for ValueKey {
    fn from(n: Node) -> Self {
        Self(vec![], Either::Left(n))
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

    pub fn tag(&self) -> usize {
        match self.value() {
            Value::Sum { tag, .. } => *tag,
            Value::Tuple { .. } => 0,
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
        let mut is = self.0 .0.clone();
        is.push(i);
        Self(ValueKey(is, self.0 .1.clone()), v)
    }
}

impl PartialEq for ValueHandle {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
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

/// TODO shouldn't be pub
#[derive(PartialEq, Clone, Eq)]
pub struct HashableHashMap<K: Hash + std::cmp::Eq, V>(HashMap<K, V>);

impl<K: Hash + std::cmp::Eq + std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug
    for HashableHashMap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

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
                let vec = (0..vs.len())
                    .map(|i| PartialValue::from(v.index(i)).into())
                    .collect();
                Self::PartialSum(HashableHashMap([(0, vec)].into_iter().collect()))
            }
            Value::Sum { tag, values, .. } => {
                let vec = (0..values.len())
                    .map(|i| PartialValue::from(v.index(i)).into())
                    .collect();
                Self::PartialSum(HashableHashMap([(*tag, vec)].into_iter().collect()))
            }
            _ => Self::Value(v),
        }
    }
}

impl PartialValue {
    const BOTTOM: Self = Self::Bottom;
    const BOTTOM_REF: &'static Self = &Self::BOTTOM;

    fn initialised(&self) -> bool {
        !self.is_top()
    }

    fn is_top(&self) -> bool {
        self == &PartialValue::Top
    }

    fn assert_invariants(&self) {
        match self {
            Self::PartialSum(HashableHashMap(hm)) => {
                assert_ne!(hm.len(), 0);
                for pv in hm.values().flat_map(|x| x.iter()) {
                    pv.assert_invariants();
                }
            }
            Self::Value(v) => {
                assert!(matches!(v.clone().into(), Self::Value(_)))
            }
            _ => {}
        }
    }

    /// TODO docs
    /// just delegate to variant_field_value
    pub fn tuple_field_value(&self, idx: usize) -> Self {
        self.variant_field_value(0, idx)
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
            }
            Self::Value(v) if v.tag() == variant => Self::Value(v.index(idx)),
            _ => Self::Top,
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let r = match self {
            Self::Value(v) => v.value().clone(),
            Self::PartialSum(HashableHashMap(hm)) => {
                let err = |hm| Err(Self::PartialSum(HashableHashMap(hm)));
                let Ok((k, v)) = hm.iter().exactly_one() else {
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

                let Ok(vs) = zip_eq(v.into_iter(), r.into_iter())
                    .map(|(v, t)| v.clone().try_into_value(t))
                    .collect::<Result<Vec<_>, _>>()
                else {
                    return err(hm);
                };

                Value::sum(*k, vs, st.clone()).map_err(|_| Self::PartialSum(HashableHashMap(hm)))?
            }
            x => Err(x)?,
        };
        assert_eq!(typ, &r.get_type());
        Ok(r)
    }

    fn join_mut_value_handle(&mut self, vh: ValueHandle) -> bool {
        self.assert_invariants();
        let mut new_self = self;
        match &mut new_self {
            Self::Top => false,
            new_self @ Self::Value(_) => {
                let Self::Value(v) = *new_self else {
                    unreachable!()
                };
                if v == &vh {
                    false
                } else {
                    **new_self = Self::Top;
                    true
                }
            }
            s @ Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    **s = Self::Top;
                    true
                }
                other => s.join_mut(other),
            },
            new_self @ Self::Bottom => {
                **new_self = vh.into();
                true
            }
        }
    }

    fn meet_mut_value_handle(&mut self, vh: ValueHandle) -> bool {
        self.assert_invariants();
        let mut new_self = self;
        match &mut new_self {
            Self::Bottom => false,
            new_self @ Self::Value(_) => {
                let Self::Value(v) = *new_self else {
                    unreachable!()
                };
                if v == &vh {
                    false
                } else {
                    **new_self = Self::Bottom;
                    true
                }
            }
            new_self @ Self::PartialSum(_) => match vh.into() {
                Self::Value(_) => {
                    **new_self = Self::Bottom;
                    true
                }
                other => new_self.join_mut(other),
            },
            new_self @ Self::Top => {
                **new_self = vh.into();
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
        let mut new_self = self;
        let changed = match (&mut new_self, other) {
            (Self::Top, _) => false,
            (new_self, other @ Self::Top) => {
                **new_self = other;
                true
            }
            (_, Self::Bottom) => false,
            (new_self @ Self::Bottom, other) => {
                **new_self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) if h1 == &h2 || h1.value() == h2.value() => false,
            (new_self @ Self::PartialSum(_), Self::PartialSum(HashableHashMap(hm2))) => {
                let mut changed = false;
                let Self::PartialSum(HashableHashMap(hm1)) = *new_self else {
                    unreachable!()
                };
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
            (new_self @ Self::Value(_), other @ Self::PartialSum(_)) => {
                let mut old_self = other;
                std::mem::swap(*new_self, &mut old_self);
                let Self::Value(h) = old_self else {
                    unreachable!()
                };
                new_self.join_mut_value_handle(h)
            }
            (new_self @ Self::PartialSum(_), Self::Value(h)) => new_self.join_mut_value_handle(h),
            (new_self, _) => {
                **new_self = Self::Top;
                false
            }
        };
        // if changed {
        // println!("join new self: {:?}", s);
        // }
        changed
    }

    pub fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
    }

    pub fn meet_mut(&mut self, other: Self) -> bool {
        let mut new_self = self;
        let changed = match (&mut new_self, other) {
            (Self::Bottom, _) => false,
            (new_self, other @ Self::Bottom) => {
                **new_self = other;
                true
            }
            (_, Self::Top) => false,
            (new_self @ Self::Top, other) => {
                **new_self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) if h1 == &h2 || h1.value() == h2.value() => false,
            (new_self @ Self::PartialSum(_), Self::PartialSum(HashableHashMap(hm2))) => {
                let mut changed = false;
                let Self::PartialSum(HashableHashMap(hm1)) = *new_self else {
                    unreachable!()
                };
                let mut keys_to_remove = vec![];
                for k in hm1.keys() {
                    if !hm2.contains_key(k) {
                        keys_to_remove.push(*k);
                    }
                }
                for (k, v) in hm2 {
                    if let Some(row) = hm1.get_mut(&k) {
                        for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                            changed |= lhs.meet_mut(rhs);
                        }
                    } else {
                        keys_to_remove.push(k);
                    }
                }
                for k in keys_to_remove {
                    hm1.remove(&k);
                    changed = true;
                }
                changed
            }
            (new_self @ Self::Value(_), other @ Self::PartialSum(_)) => {
                let mut old_self = other;
                std::mem::swap(*new_self, &mut old_self);
                let Self::Value(h) = old_self else {
                    unreachable!()
                };
                new_self.meet_mut_value_handle(h)
            }
            (s @ Self::PartialSum(_), Self::Value(h)) => s.meet_mut_value_handle(h),
            (new_self, _) => {
                **new_self = Self::Bottom;
                false
            }
        };
        // if changed {
        // println!("join new self: {:?}", s);
        // }
        changed
    }

    pub fn top() -> Self {
        Self::Top
    }

    pub fn bottom() -> Self {
        Self::Bottom
    }

    pub fn variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        Self::PartialSum(HashableHashMap(
            [(tag, values.into_iter().collect())].into_iter().collect(),
        ))
    }

    pub fn unit() -> Self {
        Self::variant(0, [])
    }
}

impl PartialOrd for PartialValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Bottom, Self::Bottom) => Some(Ordering::Equal),
            (Self::Top, Self::Top) => Some(Ordering::Equal),
            (Self::Bottom, _) => Some(Ordering::Less),
            (_, Self::Bottom) => Some(Ordering::Greater),
            (Self::Top, _) => Some(Ordering::Greater),
            (_, Self::Top) => Some(Ordering::Less),
            (Self::Value(v1), Self::Value(v2)) => (v1 == v2).then_some(Ordering::Equal),
            (Self::PartialSum(HashableHashMap(hm1)), Self::PartialSum(HashableHashMap(hm2))) => {
                let max_key = hm1.keys().chain(hm2.keys()).copied().max().unwrap();
                let (mut keys1, mut keys2) = (vec![0; max_key + 1], vec![0; max_key + 1]);
                for k in hm1.keys() {
                    keys1[*k] = 1;
                }

                for k in hm2.keys() {
                    keys2[*k] = 1;
                }

                if let Some(ord) = keys1.partial_cmp(&keys2) {
                    if ord != Ordering::Equal {
                        return Some(ord);
                    }
                } else {
                    return None;
                }
                for (k, lhs) in hm1 {
                    let Some(rhs) = hm2.get(&k) else {
                        unreachable!()
                    };
                    match lhs.partial_cmp(rhs) {
                        Some(Ordering::Equal) => continue,
                        x => {
                            return x;
                        }
                    }
                }
                Some(Ordering::Equal)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use itertools::Itertools as _;
    use lazy_static::lazy_static;
    use proptest::prelude::*;

    use super::{PartialValue, ValueHandle};
    impl Arbitrary for ValueHandle {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
            // prop_oneof![

            // ]
            todo!()
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    struct UnarySumType(usize, Vec<Vec<Arc<UnarySumType>>>);

    lazy_static! {
        static ref UNARY_SUM_TYPE_LEAF: UnarySumType = UnarySumType::new([]);
    }

    impl UnarySumType {
        pub fn new(vs: impl IntoIterator<Item = Vec<Arc<Self>>>) -> Self {
            let vec = vs.into_iter().collect_vec();
            let depth: usize = vec
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| x.0 + 1)
                .max()
                .unwrap_or(0);
            Self(depth, vec.into()).into()
        }

        fn is_leaf(&self) -> bool {
            self.0 == 0
        }

        fn assert_invariants(&self) {
            if self.is_leaf() {
                assert!(self.1.iter().all(Vec::is_empty));
            } else {
                for v in self.1.iter().flat_map(|x| x.iter()) {
                    assert!(v.0 < self.0);
                    v.assert_invariants()
                }
            }
        }

        fn select(self) -> impl Strategy<Value = Option<(usize, Vec<Arc<Self>>)>> {
            if self.is_leaf() {
                Just(None).boxed()
            } else {
                any::<prop::sample::Index>()
                    .prop_map(move |i| {
                        let index = i.index(self.1.len());
                        Some((index, self.1[index].clone()))
                    })
                    .boxed()
            }
        }
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct UnarySumTypeParams {
        depth: usize,
        branch_width: usize,
    }

    impl UnarySumTypeParams {
        pub fn descend(mut self, d: usize) -> Self {
            assert!(d < self.depth);
            self.depth = d;
            self
        }
    }

    impl Default for UnarySumTypeParams {
        fn default() -> Self {
            Self {
                depth: 3,
                branch_width: 3,
            }
        }
    }

    impl Arbitrary for UnarySumType {
        type Parameters = UnarySumTypeParams;
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(
            params @ UnarySumTypeParams {
                depth,
                branch_width,
            }: Self::Parameters,
        ) -> Self::Strategy {
            if depth == 0 {
                Just(UNARY_SUM_TYPE_LEAF.clone()).boxed()
            } else {
                (0..depth)
                    .prop_flat_map(move |d| {
                        prop::collection::vec(
                            prop::collection::vec(
                                any_with::<Self>(params.clone().descend(d)).prop_map_into(),
                                0..branch_width,
                            ),
                            1..=branch_width,
                        )
                        .prop_map(UnarySumType::new)
                    })
                    .boxed()
            }
        }
    }

    proptest! {
        #[test]
        fn unary_sum_type_valid(ust: UnarySumType) {
            ust.assert_invariants();
        }
    }

    fn any_partial_value_of_type(ust: UnarySumType) -> impl Strategy<Value = PartialValue> {
        ust.select().prop_flat_map(|x| {
            if let Some((index, usts)) = x {
                let pvs = usts
                    .into_iter()
                    .map(|x| any_partial_value_of_type(Arc::<UnarySumType>::unwrap_or_clone(x)))
                    .collect_vec();
                pvs.prop_map(move |pvs| PartialValue::variant(index, pvs))
                    .boxed()
            } else {
                Just(PartialValue::unit()).boxed()
            }
        })
    }

    fn any_partial_value_with(
        params: <UnarySumType as Arbitrary>::Parameters,
    ) -> impl Strategy<Value = PartialValue> {
        any_with::<UnarySumType>(params).prop_flat_map(any_partial_value_of_type)
    }

    fn any_partial_value() -> impl Strategy<Value = PartialValue> {
        any_partial_value_with(Default::default())
    }

    fn any_partial_values<const N: usize>() -> impl Strategy<Value = [PartialValue; N]> {
        any::<UnarySumType>().prop_flat_map(|ust| {
            TryInto::<[_; N]>::try_into(
                (0..N)
                    .map(|_| any_partial_value_of_type(ust.clone()))
                    .collect_vec(),
            )
            .unwrap()
        })
    }

    proptest! {
        // todo: ValidHandle is valid
        // todo: ValidHandle eq is an equivalence relation

        // todo: PartialValue PartialOrd is transitive
        // todo: PartialValue eq is an equivalence relation
        #[test]
        fn partial_value_valid(pv in any_partial_value()) {
            pv.assert_invariants();
        }

        #[test]
        fn bounded_lattice(v in any_partial_value()) {
            prop_assert!(&v <= &PartialValue::Top);
            prop_assert!(&v >= &PartialValue::Bottom);
        }

        #[test]
        fn lattice_changed(v1 in any_partial_value()) {
            let mut subject = v1.clone();
            assert!(!subject.join_mut(v1.clone()));
            assert!(!subject.meet_mut(v1.clone()));
        }

        #[test]
        fn lattice([v1,v2] in any_partial_values()) {
            let meet = v1.clone().meet(v2.clone());
            prop_assert!(&meet <= &v1, "meet not less <=: {:#?}", &meet);
            prop_assert!(&meet <= &v2, "meet not less <=: {:#?}", &meet);

            let join = v1.clone().join(v2.clone());
            prop_assert!(&join >= &v1, "join not >=: {:#?}", &join);
            prop_assert!(&join >= &v2, "join not >=: {:#?}", &join);
        }
    }
}
