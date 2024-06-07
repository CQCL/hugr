#![allow(missing_docs)]
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use itertools::{zip_eq, Itertools as _};

use crate::ops::Value;
use crate::types::{Type, TypeEnum};

mod value_handle;

pub use value_handle::{ValueKey, ValueHandle};


/// TODO shouldn't be pub
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum(HashMap<usize, Vec<PartialValue>>);

impl PartialSum {
    pub fn variant(tag: usize, values: impl IntoIterator<Item = PartialValue>) -> Self {
        Self([(tag, values.into_iter().collect())].into_iter().collect())
    }

    pub fn num_variants(&self) -> usize {
        self.0.len()
    }

    fn assert_variants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

    pub fn variant_field_value(&self, variant: usize, idx: usize) -> PartialValue {
        if let Some(row) = self.0.get(&variant) {
            assert!(row.len() > idx);
            row[idx].clone()
        } else {
            // We must return top. if self were to gain this variant, we would return the element of that variant.
            // We must ensure that the value return now is <= that future value
            PartialValue::top()
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let Ok((k, v)) = self.0.iter().exactly_one() else {
            Err(self)?
        };

        let TypeEnum::Sum(st) = typ.as_type_enum() else {
            Err(self)?
        };
        let Some(r) = st.get_variant(*k) else {
            Err(self)?
        };
        if v.len() != r.len() {
            return Err(self)
        }
        match zip_eq(v.into_iter(), r.into_iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>,_>>() {
            Ok(vs) => {
                Value::sum(*k, vs, st.clone()).map_err(|_| self)
            }
            Err(_) => Err(self)
        }
    }

    // unsafe because we panic if any common rows have different lengths
    fn join_mut_unsafe(&mut self, other: Self) -> bool {
        let mut changed = false;

        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.join_mut(rhs);
                }
            } else {
                self.0.insert(k, v);
                changed = true;
            }
        }
        changed
    }

    // unsafe because we panic if any common rows have different lengths
    fn meet_mut_unsafe(&mut self, other: Self) -> bool {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for k in self.0.keys() {
            if !other.0.contains_key(k) {
                keys_to_remove.push(*k);
            }
        }
        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.meet_mut(rhs);
                }
            } else {
                keys_to_remove.push(k);
            }
        }
        for k in keys_to_remove {
            self.0.remove(&k);
            changed = true;
        }
        changed
    }
}

impl PartialOrd for PartialSum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let max_key = self.0.keys().chain(other.0.keys()).copied().max().unwrap();
        let (mut keys1, mut keys2) = (vec![0; max_key + 1], vec![0; max_key + 1]);
        for k in self.0.keys() {
            keys1[*k] = 1;
        }

        for k in other.0.keys() {
            keys2[*k] = 1;
        }

        if let Some(ord) = keys1.partial_cmp(&keys2) {
            if ord != Ordering::Equal {
                return Some(ord);
            }
        } else {
            return None;
        }
        for (k, lhs) in &self.0 {
            let Some(rhs) = other.0.get(&k) else {
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
}

impl std::fmt::Debug for PartialSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Hash for PartialSum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl TryFrom<ValueHandle> for PartialSum {
    type Error = ValueHandle;

    fn try_from(value: ValueHandle) -> Result<Self, Self::Error> {
        match value.value() {
            Value::Tuple { vs } => {
                let vec = (0..vs.len())
                    .map(|i| PartialValue::from(value.index(i)).into())
                    .collect();
                return Ok(Self([(0, vec)].into_iter().collect()));
            }
            Value::Sum { tag, values, .. } => {
                let vec = (0..values.len())
                    .map(|i| PartialValue::from(value.index(i)).into())
                    .collect();
                return Ok(Self([(*tag, vec)].into_iter().collect()));
            }
            _ => ()
        };
        Err(value)
    }
}

#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue {
    Bottom,
    Value(ValueHandle),
    PartialSum(PartialSum),
    Top,
}

impl From<ValueHandle> for PartialValue {
    fn from(v: ValueHandle) -> Self {
        TryInto::<PartialSum>::try_into(v).map_or_else(Self::Value, Self::PartialSum)
    }
}

impl From<PartialSum> for PartialValue {
    fn from(v: PartialSum) -> Self {
        Self::PartialSum(v)
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
            Self::PartialSum(ps) => {
                ps.assert_variants();
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
            Self::PartialSum(ps) => {
                ps.variant_field_value(variant, idx)
            }
            Self::Value(v) if v.tag() == variant => Self::Value(v.index(idx)),
            _ => Self::Top,
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<Value, Self> {
        let r = match self {
            Self::Value(v) => Ok(v.value().clone()),
            Self::PartialSum(ps) => ps.try_into_value(typ).map_err(Self::PartialSum),
            x => Err(x),
        }?;
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

    fn value_handles_equal(&self, rhs: &ValueHandle) -> bool {
        let Self::Value(lhs) = self else { unreachable!() };
        lhs == rhs
            // The following is a good idea if ValueHandle gains an Eq
            // instance and so does not do this check:
            // || lhs.value() == rhs.value()
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
            (new_self @ Self::Value(_), Self::Value(h2)) => {
                if new_self.value_handles_equal(&h2) {
                    false
                } else {
                    **new_self = Self::Top;
                    true
                }
            }
            (new_self @ Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = *new_self else {
                    unreachable!()
                };

                ps1.join_mut_unsafe(ps2)
            }
            (new_self @ Self::Value(_), other) => {
                let mut old_self = other;
                std::mem::swap(*new_self, &mut old_self);
                let Self::Value(h) = old_self else {
                    unreachable!()
                };
                new_self.join_mut_value_handle(h)
            }
            (new_self, Self::Value(h)) => new_self.join_mut_value_handle(h),
            // (new_self, _) => {
            //     **new_self = Self::Top;
            //     false
            // }
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
            (new_self @ Self::Value(_), Self::Value(h2)) => {
                if new_self.value_handles_equal(&h2) {
                    false
                } else {
                    **new_self = Self::Bottom;
                    true
                }
            }
            (new_self @ Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = *new_self else {
                    unreachable!()
                };
                ps1.meet_mut_unsafe(ps2)
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
            // (new_self, _) => {
            //     **new_self = Self::Bottom;
            //     false
            // }
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
        PartialSum::variant(tag, values).into()
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
            (Self::Value(_), Self::Value(v2)) => self.value_handles_equal(v2).then_some(Ordering::Equal),
            (Self::PartialSum(ps1), Self::PartialSum(ps2)) => ps1.partial_cmp(ps2),
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

    use crate::{std_extensions::arithmetic::int_types::{self, INT_TYPES, LOG_WIDTH_BOUND}, types::{CustomType, Type, TypeEnum}};

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
    enum TestSumLeafType {
        Int(Type),
        Unit,
    }

    impl Arbitrary for TestSumLeafType {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_params: Self::Parameters,) -> Self::Strategy {
            let int_strat = (0..LOG_WIDTH_BOUND).prop_map(|i| Self::Int(INT_TYPES[i as usize].clone()));
            prop_oneof![
                Just(TestSumLeafType::Unit),
                int_strat
            ].boxed()
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestSumType {
        Branch(usize, Vec<Vec<Arc<TestSumType>>>),
        Leaf(TestSumLeafType)
    }

    impl TestSumType {
        const UNIT: TestSumLeafType = TestSumLeafType::Unit;

        pub fn leaf(v: Type) -> Self {
            TestSumType::Leaf(TestSumLeafType::Int(v))
        }

        pub fn branch(vs: impl IntoIterator<Item = Vec<Arc<Self>>>) -> Self {
            let vec = vs.into_iter().collect_vec();
            let depth: usize = vec
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| x.depth() + 1)
                .max()
                .unwrap_or(0);
            Self::Branch(depth, vec.into()).into()
        }

        fn depth(&self) -> usize {
            match self {
                TestSumType::Branch(x, _) => *x,
                TestSumType::Leaf(_) => 0,
            }
        }

        fn is_leaf(&self) -> bool {
            self.depth() == 0
        }

        fn assert_invariants(&self) {
            match self {
                TestSumType::Branch(d, sop) => {
                    assert!(!sop.is_empty(), "No variants");
                    for v in sop.iter().flat_map(|x| x.iter()) {
                        assert!(v.depth() < *d);
                        v.assert_invariants();
                    }
                }
                TestSumType::Leaf(TestSumLeafType::Int(t)) => {
                    if let TypeEnum::Extension(ct) = t.as_type_enum() {
                        assert_eq!("int", ct.name());
                        assert_eq!(&int_types::EXTENSION_ID, ct.extension());
                    } else {
                        panic!("Expected int type, got {:#?}", t);
                    }
                },
                _ => ()
            }
        }

        fn select(self) -> impl Strategy<Value = Option<(usize, Vec<Arc<Self>>)>> {
            match self {
                TestSumType::Branch(_, sop) => {
                    any::<prop::sample::Index>()
                        .prop_map(move |i| {
                            let index = i.index(sop.len());
                            Some((index, sop[index].clone()))
                        })
                        .boxed()
                }
                TestSumType::Leaf(_) => Just(None).boxed()

            }
        }

        // fn type_check(&self, pv: PartialValue) -> bool {
        //     match (self,pv) {
        //         (_, PartialValue::Bottom) | PartialValue::Top => true,
        //         (_, PartialValue::Value(_)) => todo!(),
        //         (TestSumType::Branch(_, sop), PartialValue::PartialSum(ps)) => {
        //             for (k,v) in ps.0 {
        //                 if k >= sop.len() {
        //                     return false
        //                 }

        //             }
        //         }
        //         (TestSumType::Branch(_, _), PartialValue::Top) => todo!(),
        //         (TestSumType::Leaf(_), PartialValue::Bottom) => todo!(),
        //         (TestSumType::Leaf(_), PartialValue::Value(_)) => todo!(),
        //         (TestSumType::Leaf(_), PartialValue::PartialSum(_)) => todo!(),
        //         (TestSumType::Leaf(_), PartialValue::Top) => todo!(),
        //     }

        // }
    }

    impl From<TestSumLeafType> for TestSumType {
        fn from(value: TestSumLeafType) -> Self {
            Self::Leaf(value)
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

    impl Arbitrary for TestSumType {
        type Parameters = UnarySumTypeParams;
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(
            params @ UnarySumTypeParams {
                depth,
                branch_width,
            }: Self::Parameters,
        ) -> Self::Strategy {
            if depth == 0 {
                any::<TestSumLeafType>().prop_map_into().boxed()
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
                        .prop_map(TestSumType::branch)
                    })
                    .boxed()
            }
        }
    }

    proptest! {
        #[test]
        fn unary_sum_type_valid(ust: TestSumType) {
            ust.assert_invariants();
        }
    }

    fn any_partial_value_of_type(ust: TestSumType) -> impl Strategy<Value = PartialValue> {
        ust.select().prop_flat_map(|x| {
            if let Some((index, usts)) = x {
                let pvs = usts
                    .into_iter()
                    .map(|x| any_partial_value_of_type(Arc::<TestSumType>::unwrap_or_clone(x)))
                    .collect_vec();
                pvs.prop_map(move |pvs| PartialValue::variant(index, pvs))
                    .boxed()
            } else {
                Just(PartialValue::unit()).boxed()
            }
        })
    }

    fn any_partial_value_with(
        params: <TestSumType as Arbitrary>::Parameters,
    ) -> impl Strategy<Value = PartialValue> {
        any_with::<TestSumType>(params).prop_flat_map(any_partial_value_of_type)
    }

    fn any_partial_value() -> impl Strategy<Value = PartialValue> {
        any_partial_value_with(Default::default())
    }

    fn any_partial_values<const N: usize>() -> impl Strategy<Value = [PartialValue; N]> {
        any::<TestSumType>().prop_flat_map(|ust| {
            TryInto::<[_; N]>::try_into(
                (0..N)
                    .map(|_| any_partial_value_of_type(ust.clone()))
                    .collect_vec(),
            )
            .unwrap()
        })
    }

    fn any_typed_partial_value() -> impl Strategy<Value = (TestSumType, PartialValue)> {
        any::<TestSumType>().prop_flat_map(|t| {
            any_partial_value_of_type(t.clone()).prop_map(move |v| (t.clone(),v))
        })
    }

    proptest! {
        // #[test]
        // fn partial_value_type((tst, pv) in any_typed_partial_value()) {
        //     prop_assert!(tst.type_check(pv))
        // }

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
