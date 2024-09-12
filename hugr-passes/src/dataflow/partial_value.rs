use ascent::lattice::BoundedLattice;
use ascent::Lattice;
use hugr_core::types::{SumType, Type, TypeEnum, TypeRow};
use itertools::{zip_eq, Itertools};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Aka, deconstructible into Sum (TryIntoSum ?)
pub trait AbstractValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// We write this way to optimize query/inspection (is-it-a-sum),
    /// at the cost of requiring more cloning during actual conversion
    /// (inside the lazy Iterator, or for the error case, as Self remains)
    fn as_sum(&self) -> Option<(usize, impl Iterator<Item = Self> + '_)>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueOrSum<V> {
    Value(V),
    Sum {
        tag: usize,
        items: Vec<Self>,
        st: SumType,
    },
}

// TODO ALAN inline into PartialValue? Has to be public as it's in a pub enum
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum<V>(pub HashMap<usize, Vec<PartialValue<V>>>);

impl<V> PartialSum<V> {
    pub fn unit() -> Self {
        Self::variant(0, [])
    }
    pub fn variant(tag: usize, values: impl IntoIterator<Item = PartialValue<V>>) -> Self {
        Self(HashMap::from([(tag, Vec::from_iter(values))]))
    }

    pub fn num_variants(&self) -> usize {
        self.0.len()
    }
}

impl<V: AbstractValue> PartialSum<V> {
    fn assert_invariants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

    // Err with key if any common rows have different lengths (self may have been mutated)
    fn try_join_mut(&mut self, other: Self) -> Result<bool, usize> {
        let mut changed = false;

        for (k, v) in other.0 {
            if let Some(row) = self.0.get_mut(&k) {
                if v.len() != row.len() {
                    // Better to check first and avoid mutation, but fine here
                    return Err(k);
                }
                for (lhs, rhs) in zip_eq(row.iter_mut(), v.into_iter()) {
                    changed |= lhs.join_mut(rhs);
                }
            } else {
                self.0.insert(k, v);
                changed = true;
            }
        }
        Ok(changed)
    }

    // Error with key if any common rows have different lengths ( => Bottom)
    fn try_meet_mut(&mut self, other: Self) -> Result<bool, usize> {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for (k, v) in self.0.iter() {
            match other.0.get(k) {
                None => keys_to_remove.push(*k),
                Some(o_v) => {
                    if v.len() != o_v.len() {
                        return Err(*k);
                    }
                }
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
        Ok(changed)
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }

    pub fn try_into_value(self, typ: &Type) -> Result<ValueOrSum<V>, Self> {
        let Ok((k, v)) = self.0.iter().exactly_one() else {
            Err(self)?
        };

        let TypeEnum::Sum(st) = typ.as_type_enum() else {
            Err(self)?
        };
        let Some(r) = st.get_variant(*k) else {
            Err(self)?
        };
        let Ok(r) = TypeRow::try_from(r.clone()) else {
            Err(self)?
        };
        if v.len() != r.len() {
            return Err(self);
        }
        match zip_eq(v, r.into_iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(vs) => Ok(ValueOrSum::Sum {
                tag: *k,
                items: vs,
                st: st.clone(),
            }),
            Err(_) => Err(self),
        }
    }
}

impl<V: Clone> PartialSum<V> {
    pub fn variant_values(&self, variant: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let row = self.0.get(&variant)?;
        assert!(row.len() == len);
        Some(row.clone())
    }
}

impl<V: PartialEq> PartialOrd for PartialSum<V> {
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
            let Some(rhs) = other.0.get(k) else {
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

impl<V: std::fmt::Debug> std::fmt::Debug for PartialSum<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: Hash> Hash for PartialSum<V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.hash(state);
        }
    }
}

/// We really must prevent people from constructing PartialValue::Value of
/// any `value` where `value.as_sum().is_some()``
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue<V> {
    Bottom,
    Value(V),
    PartialSum(PartialSum<V>),
    Top,
}

impl<V: AbstractValue> From<V> for PartialValue<V> {
    fn from(v: V) -> Self {
        v.as_sum()
            .map(|(tag, values)| Self::variant(tag, values.map(Self::from)))
            .unwrap_or(Self::Value(v))
    }
}

impl<V> From<PartialSum<V>> for PartialValue<V> {
    fn from(v: PartialSum<V>) -> Self {
        Self::PartialSum(v)
    }
}

impl<V: AbstractValue> PartialValue<V> {
    // const BOTTOM: Self = Self::Bottom;
    // const BOTTOM_REF: &'static Self = &Self::BOTTOM;
    fn assert_invariants(&self) {
        match self {
            Self::PartialSum(ps) => {
                ps.assert_invariants();
            }
            Self::Value(v) => {
                assert!(v.as_sum().is_none())
            }
            _ => {}
        }
    }

    pub fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    pub fn variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        PartialSum::variant(tag, values).into()
    }

    pub fn unit() -> Self {
        Self::variant(0, [])
    }

    pub fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let vals = match self {
            PartialValue::Bottom => return None,
            PartialValue::Value(v) => {
                assert!(v.as_sum().is_none());
                return None;
            }
            PartialValue::PartialSum(ps) => ps.variant_values(tag, len)?,
            PartialValue::Top => vec![PartialValue::Top; len],
        };
        assert_eq!(vals.len(), len);
        Some(vals)
    }

    pub fn supports_tag(&self, tag: usize) -> bool {
        match self {
            PartialValue::Bottom => false,
            PartialValue::Value(v) => {
                assert!(v.as_sum().is_none());
                false
            }
            PartialValue::PartialSum(ps) => ps.supports_tag(tag),
            PartialValue::Top => true,
        }
    }

    pub fn try_into_value(self, typ: &Type) -> Result<ValueOrSum<V>, Self> {
        match self {
            Self::Value(v) => Ok(ValueOrSum::Value(v.clone())),
            Self::PartialSum(ps) => ps.try_into_value(typ).map_err(Self::PartialSum),
            x => Err(x),
        }
    }
}

impl<V: AbstractValue> Lattice for PartialValue<V> {
    fn join_mut(&mut self, other: Self) -> bool {
        self.assert_invariants();
        // println!("join {self:?}\n{:?}", &other);
        match (&*self, other) {
            (Self::Top, _) => false,
            (_, other @ Self::Top) => {
                *self = other;
                true
            }
            (_, Self::Bottom) => false,
            (Self::Bottom, other) => {
                *self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    *self = Self::Top;
                    true
                }
            }
            (Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = self else {
                    unreachable!()
                };
                match ps1.try_join_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        *self = Self::Top;
                        true
                    }
                }
            }
            (Self::Value(ref v), Self::PartialSum(_))
            | (Self::PartialSum(_), Self::Value(ref v)) => {
                assert!(v.as_sum().is_none());
                *self = Self::Top;
                true
            }
        }
    }

    fn meet(mut self, other: Self) -> Self {
        self.meet_mut(other);
        self
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        self.assert_invariants();
        match (&*self, other) {
            (Self::Bottom, _) => false,
            (_, other @ Self::Bottom) => {
                *self = other;
                true
            }
            (_, Self::Top) => false,
            (Self::Top, other) => {
                *self = other;
                true
            }
            (Self::Value(h1), Self::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    *self = Self::Bottom;
                    true
                }
            }
            (Self::PartialSum(_), Self::PartialSum(ps2)) => {
                let Self::PartialSum(ps1) = self else {
                    unreachable!()
                };
                match ps1.try_meet_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        *self = Self::Bottom;
                        true
                    }
                }
            }
            (Self::Value(ref v), Self::PartialSum(_))
            | (Self::PartialSum(_), Self::Value(ref v)) => {
                assert!(v.as_sum().is_none());
                *self = Self::Bottom;
                true
            }
        }
    }
}

impl<V: AbstractValue> BoundedLattice for PartialValue<V> {
    fn top() -> Self {
        Self::Top
    }

    fn bottom() -> Self {
        Self::Bottom
    }
}

impl<V: PartialEq> PartialOrd for PartialValue<V> {
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
            (Self::PartialSum(ps1), Self::PartialSum(ps2)) => ps1.partial_cmp(ps2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use ascent::{lattice::BoundedLattice, Lattice};
    use itertools::{zip_eq, Either, Itertools as _};
    use proptest::prelude::*;

    use hugr_core::{
        std_extensions::arithmetic::int_types::{self, ConstInt, INT_TYPES, LOG_WIDTH_BOUND},
        types::{Type, TypeArg, TypeEnum},
    };

    use super::{PartialSum, PartialValue};
    use crate::const_fold2::value_handle::{ValueHandle, ValueKey};

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

    impl TestSumLeafType {
        fn assert_valid(&self) {
            if let Self::Int(t) = self {
                if let TypeEnum::Extension(ct) = t.as_type_enum() {
                    assert_eq!("int", ct.name());
                    assert_eq!(&int_types::EXTENSION_ID, ct.extension());
                } else {
                    panic!("Expected int type, got {:#?}", t);
                }
            }
        }

        fn get_type(&self) -> Type {
            match self {
                Self::Int(t) => t.clone(),
                Self::Unit => Type::UNIT,
            }
        }

        fn type_check(&self, ps: &PartialSum<ValueHandle>) -> bool {
            match self {
                Self::Int(_) => false,
                Self::Unit => {
                    if let Ok((0, v)) = ps.0.iter().exactly_one() {
                        v.is_empty()
                    } else {
                        false
                    }
                }
            }
        }

        fn partial_value_strategy(self) -> impl Strategy<Value = PartialValue<ValueHandle>> {
            match self {
                Self::Int(t) => {
                    let TypeEnum::Extension(ct) = t.as_type_enum() else {
                        unreachable!()
                    };
                    // TODO this should be get_log_width, but that's not pub
                    let TypeArg::BoundedNat { n: lw } = ct.args()[0] else {
                        panic!()
                    };
                    (0u64..(1 << (2u64.pow(lw as u32) - 1)))
                        .prop_map(move |x| {
                            let ki = ConstInt::new_u(lw as u8, x).unwrap();
                            let k = ValueKey::try_new(ki.clone()).unwrap();
                            ValueHandle::new(k, Arc::new(ki.into())).into()
                        })
                        .boxed()
                }
                Self::Unit => Just(PartialSum::unit().into()).boxed(),
            }
        }
    }

    impl Arbitrary for TestSumLeafType {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_params: Self::Parameters) -> Self::Strategy {
            let int_strat =
                (0..LOG_WIDTH_BOUND).prop_map(|i| Self::Int(INT_TYPES[i as usize].clone()));
            prop_oneof![Just(TestSumLeafType::Unit), int_strat].boxed()
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestSumType {
        Branch(usize, Vec<Vec<Arc<TestSumType>>>),
        Leaf(TestSumLeafType),
    }

    impl TestSumType {
        #[allow(unused)] // ALAN ?
        fn leaf(v: Type) -> Self {
            TestSumType::Leaf(TestSumLeafType::Int(v))
        }

        fn branch(vs: impl IntoIterator<Item = Vec<Arc<Self>>>) -> Self {
            let vec = vs.into_iter().collect_vec();
            let depth: usize = vec
                .iter()
                .flat_map(|x| x.iter())
                .map(|x| x.depth() + 1)
                .max()
                .unwrap_or(0);
            Self::Branch(depth, vec)
        }

        fn depth(&self) -> usize {
            match self {
                TestSumType::Branch(x, _) => *x,
                TestSumType::Leaf(_) => 0,
            }
        }

        #[allow(unused)] // ALAN ?
        fn is_leaf(&self) -> bool {
            self.depth() == 0
        }

        fn assert_valid(&self) {
            match self {
                TestSumType::Branch(d, sop) => {
                    assert!(!sop.is_empty(), "No variants");
                    for v in sop.iter().flat_map(|x| x.iter()) {
                        assert!(v.depth() < *d);
                        v.assert_valid();
                    }
                }
                TestSumType::Leaf(l) => {
                    l.assert_valid();
                }
            }
        }

        fn select(self) -> impl Strategy<Value = Either<TestSumLeafType, (usize, Vec<Arc<Self>>)>> {
            match self {
                TestSumType::Branch(_, sop) => any::<prop::sample::Index>()
                    .prop_map(move |i| {
                        let index = i.index(sop.len());
                        Either::Right((index, sop[index].clone()))
                    })
                    .boxed(),
                TestSumType::Leaf(l) => Just(Either::Left(l)).boxed(),
            }
        }

        fn get_type(&self) -> Type {
            match self {
                TestSumType::Branch(_, sop) => Type::new_sum(
                    sop.iter()
                        .map(|row| row.iter().map(|x| x.get_type()).collect_vec()),
                ),
                TestSumType::Leaf(l) => l.get_type(),
            }
        }

        fn type_check(&self, pv: &PartialValue<ValueHandle>) -> bool {
            match (self, pv) {
                (_, PartialValue::Bottom) | (_, PartialValue::Top) => true,
                (_, PartialValue::Value(v)) => self.get_type() == v.get_type(),
                (TestSumType::Branch(_, sop), PartialValue::PartialSum(ps)) => {
                    for (k, v) in &ps.0 {
                        if *k >= sop.len() {
                            return false;
                        }
                        let prod = &sop[*k];
                        if prod.len() != v.len() {
                            return false;
                        }
                        if !zip_eq(prod, v).all(|(lhs, rhs)| lhs.type_check(rhs)) {
                            return false;
                        }
                    }
                    true
                }
                (Self::Leaf(l), PartialValue::PartialSum(ps)) => l.type_check(ps),
            }
        }
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
            ust.assert_valid();
        }
    }

    fn any_partial_value_of_type(
        ust: TestSumType,
    ) -> impl Strategy<Value = PartialValue<ValueHandle>> {
        ust.select().prop_flat_map(|x| match x {
            Either::Left(l) => l.partial_value_strategy().boxed(),
            Either::Right((index, usts)) => {
                let pvs = usts
                    .into_iter()
                    .map(|x| {
                        any_partial_value_of_type(
                            Arc::<TestSumType>::try_unwrap(x)
                                .unwrap_or_else(|x| x.as_ref().clone()),
                        )
                    })
                    .collect_vec();
                pvs.prop_map(move |pvs| PartialValue::variant(index, pvs))
                    .boxed()
            }
        })
    }

    fn any_partial_value_with(
        params: <TestSumType as Arbitrary>::Parameters,
    ) -> impl Strategy<Value = PartialValue<ValueHandle>> {
        any_with::<TestSumType>(params).prop_flat_map(any_partial_value_of_type)
    }

    fn any_partial_value() -> impl Strategy<Value = PartialValue<ValueHandle>> {
        any_partial_value_with(Default::default())
    }

    fn any_partial_values<const N: usize>() -> impl Strategy<Value = [PartialValue<ValueHandle>; N]>
    {
        any::<TestSumType>().prop_flat_map(|ust| {
            TryInto::<[_; N]>::try_into(
                (0..N)
                    .map(|_| any_partial_value_of_type(ust.clone()))
                    .collect_vec(),
            )
            .unwrap()
        })
    }

    fn any_typed_partial_value() -> impl Strategy<Value = (TestSumType, PartialValue<ValueHandle>)>
    {
        any::<TestSumType>().prop_flat_map(|t| {
            any_partial_value_of_type(t.clone()).prop_map(move |v| (t.clone(), v))
        })
    }

    proptest! {
        #[test]
        fn partial_value_type((tst, pv) in any_typed_partial_value()) {
            prop_assert!(tst.type_check(&pv))
        }

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
            prop_assert!(v <= PartialValue::top());
            prop_assert!(v >= PartialValue::bottom());
        }

        #[test]
        fn meet_join_self_noop(v1 in any_partial_value()) {
            let mut subject = v1.clone();

            assert_eq!(v1.clone(), v1.clone().join(v1.clone()));
            assert!(!subject.join_mut(v1.clone()));
            assert_eq!(subject, v1);

            assert_eq!(v1.clone(), v1.clone().meet(v1.clone()));
            assert!(!subject.meet_mut(v1.clone()));
            assert_eq!(subject, v1);
        }

        #[test]
        fn lattice([v1,v2] in any_partial_values()) {
            let meet = v1.clone().meet(v2.clone());
            prop_assert!(meet <= v1, "meet not less <=: {:#?}", &meet);
            prop_assert!(meet <= v2, "meet not less <=: {:#?}", &meet);

            let join = v1.clone().join(v2.clone());
            prop_assert!(join >= v1, "join not >=: {:#?}", &join);
            prop_assert!(join >= v2, "join not >=: {:#?}", &join);
        }
    }
}
