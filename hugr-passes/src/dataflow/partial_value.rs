use ascent::lattice::BoundedLattice;
use ascent::Lattice;
use hugr_core::ops::constant::SumOf;
use hugr_core::types::{Type, TypeEnum, TypeRow};
use itertools::{zip_eq, Itertools};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Trait for an underlying domain of abstract values which can form the *elements* of a
/// [PartialValue] and thus be used in dataflow analysis.
pub trait AbstractValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// Computes the join of two values (i.e. towards `Top``), if this is representable
    /// within the underlying domain.
    /// Otherwise return `None` (i.e. an instruction to use [PartialValue::Top]).
    ///
    /// The default checks equality between `self` and `other` and returns `self` if
    /// the two are identical, otherwise `None`.
    fn try_join(self, other: Self) -> Option<Self> {
        (self == other).then_some(self)
    }

    /// Computes the meet of two values (i.e. towards `Bottom`), if this is representable
    /// within the underlying domain.
    /// Otherwise return `None` (i.e. an instruction to use [PartialValue::Bottom]).
    ///
    /// The default checks equality between `self` and `other` and returns `self` if
    /// the two are identical, otherwise `None`.
    fn try_meet(self, other: Self) -> Option<Self> {
        (self == other).then_some(self)
    }
}

/// A representation of a value of [SumType](hugr_core::types::SumType), that may have
/// one or more possible tags, with a [PartialValue] representation of each element-value
/// of each possible tag.
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum<V>(pub HashMap<usize, Vec<PartialValue<V>>>);

impl<V> PartialSum<V> {
    /// New instance for a single known tag.
    /// (Multi-tag instances can be created via [Self::try_join_mut].)
    pub fn new_variant(tag: usize, values: impl IntoIterator<Item = PartialValue<V>>) -> Self {
        Self(HashMap::from([(tag, Vec::from_iter(values))]))
    }

    /// The number of possible variants we know about. (NOT the number of tags possible
    /// for the value's type, whatever [SumType](hugr_core::types::SumType) that might be.)
    pub fn num_variants(&self) -> usize {
        self.0.len()
    }

    fn assert_invariants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }
}

impl<V: AbstractValue> PartialSum<V> {
    /// Joins (towards `Top`) self with another [PartialSum]. If successful, returns
    /// whether `self` has changed.
    ///
    /// Fails (without mutation) with the conflicting tag if any common rows have different lengths.
    pub fn try_join_mut(&mut self, other: Self) -> Result<bool, usize> {
        for (k, v) in &other.0 {
            if self.0.get(k).is_some_and(|row| row.len() != v.len()) {
                return Err(*k);
            }
        }
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
        Ok(changed)
    }

    /// Mutates self according to lattice meet operation (towards `Bottom`). If successful,
    /// returns whether `self` has changed.
    ///
    /// # Errors
    /// Fails without mutation, either:
    /// * `Some(tag)` if the two [PartialSum]s both had rows with that `tag` but of different lengths
    /// * `None` if the two instances had no rows in common (i.e., the result is "Bottom")
    pub fn try_meet_mut(&mut self, other: Self) -> Result<bool, Option<usize>> {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for (k, v) in self.0.iter() {
            match other.0.get(k) {
                None => keys_to_remove.push(*k),
                Some(o_v) => {
                    if v.len() != o_v.len() {
                        return Err(Some(*k));
                    }
                }
            }
        }
        if keys_to_remove.len() == self.0.len() {
            return Err(None);
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

    /// Whether this sum might have the specified tag
    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }

    /// Turns this instance into a [SumOf] if it has exactly one possible tag,
    /// otherwise failing and returning itself back unmodified (also if there is another
    /// error, e.g. this instance is not described by `typ`).
    // ALAN is this too parametric? Should we fix V2 == Value? Is the 'Self' error useful (no?)
    pub fn try_into_value<V2: From<V> + TryFrom<SumOf<V2>>>(
        self,
        typ: &Type,
    ) -> Result<SumOf<V2>, Self> {
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
        match zip_eq(v, r.iter())
            .map(|(v, t)| v.clone().try_into_value(t))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(values) => Ok(SumOf {
                tag: *k,
                values,
                sum_type: st.clone(),
            }),
            Err(_) => Err(self),
        }
    }
}

impl<V: Clone> PartialSum<V> {
    /// If this Sum might have the specified `tag`, get the elements inside that tag.
    pub fn variant_values(&self, variant: usize) -> Option<Vec<PartialValue<V>>> {
        self.0.get(&variant).cloned()
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

        Some(match keys1.cmp(&keys2) {
            ord @ Ordering::Greater | ord @ Ordering::Less => ord,
            Ordering::Equal => {
                for (k, lhs) in &self.0 {
                    let Some(rhs) = other.0.get(k) else {
                        unreachable!()
                    };
                    let key_cmp = lhs.partial_cmp(rhs);
                    if key_cmp != Some(Ordering::Equal) {
                        return key_cmp;
                    }
                }
                Ordering::Equal
            }
        })
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

/// Wraps some underlying representation (knowledge) of values into a lattice
/// for use in dataflow analysis, including that an instance may be a [PartialSum]
/// of values of the underlying representation
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue<V> {
    /// No possibilities known (so far)
    Bottom,
    /// A single value (of the underlying representation)
    Value(V),
    /// Sum (with at least one, perhaps several, possible tags) of underlying values
    PartialSum(PartialSum<V>),
    /// Might be more than one distinct value of the underlying type `V`
    Top,
}

impl<V> From<V> for PartialValue<V> {
    fn from(v: V) -> Self {
        Self::Value(v)
    }
}

impl<V> From<PartialSum<V>> for PartialValue<V> {
    fn from(v: PartialSum<V>) -> Self {
        Self::PartialSum(v)
    }
}

impl<V> PartialValue<V> {
    fn assert_invariants(&self) {
        if let Self::PartialSum(ps) = self {
            ps.assert_invariants();
        }
    }

    /// New instance of a sum with a single known tag.
    pub fn new_variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        PartialSum::new_variant(tag, values).into()
    }

    /// New instance of unit type (i.e. the only possible value, with no contents)
    pub fn new_unit() -> Self {
        Self::new_variant(0, [])
    }
}

impl<V: AbstractValue> PartialValue<V> {
    /// If this value might be a Sum with the specified `tag`, get the elements inside that tag.
    ///
    /// # Panics
    ///
    /// if the value is believed, for that tag, to have a number of values other than `len`
    pub fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let vals = match self {
            PartialValue::Bottom | PartialValue::Value(_) => return None,
            PartialValue::PartialSum(ps) => ps.variant_values(tag)?,
            PartialValue::Top => vec![PartialValue::Top; len],
        };
        assert_eq!(vals.len(), len);
        Some(vals)
    }

    /// Tells us whether this value might be a Sum with the specified `tag`
    pub fn supports_tag(&self, tag: usize) -> bool {
        match self {
            PartialValue::Bottom | PartialValue::Value(_) => false,
            PartialValue::PartialSum(ps) => ps.supports_tag(tag),
            PartialValue::Top => true,
        }
    }

    /// Extracts a value (in any representation supporting both leaf values and sums)
    // ALAN is this too parametric? Should we fix V2 == Value? Is the error useful (should we have 'Self') or is it a smell?
    pub fn try_into_value<V2: From<V> + TryFrom<SumOf<V2>>>(
        self,
        typ: &Type,
    ) -> Result<V2, Option<<V2 as TryFrom<SumOf<V2>>>::Error>> {
        match self {
            Self::Value(v) => Ok(V2::from(v.clone())),
            Self::PartialSum(ps) => {
                let v = ps.try_into_value(typ).map_err(|_| None)?;
                V2::try_from(v).map_err(Some)
            }
            _ => Err(None),
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
            (Self::Value(h1), Self::Value(h2)) => match h1.clone().try_join(h2) {
                Some(h3) => {
                    let ch = h3 != *h1;
                    *self = Self::Value(h3);
                    ch
                }
                None => {
                    *self = Self::Top;
                    true
                }
            },
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
            (Self::Value(_), Self::PartialSum(_)) | (Self::PartialSum(_), Self::Value(_)) => {
                *self = Self::Top;
                true
            }
        }
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
            (Self::Value(h1), Self::Value(h2)) => match h1.clone().try_meet(h2) {
                Some(h3) => {
                    let ch = h3 != *h1;
                    *self = Self::Value(h3);
                    ch
                }
                None => {
                    *self = Self::Bottom;
                    true
                }
            },
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
            (Self::Value(_), Self::PartialSum(_)) | (Self::PartialSum(_), Self::Value(_)) => {
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
    use itertools::{zip_eq, Itertools as _};
    use prop::sample::subsequence;
    use proptest::prelude::*;

    use proptest_recurse::{StrategyExt, StrategySet};

    use super::{AbstractValue, PartialSum, PartialValue};

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestSumType {
        Branch(Vec<Vec<Arc<TestSumType>>>),
        /// None => unit, Some => TestValue <= this *usize*
        Leaf(Option<usize>),
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestValue(usize);

    impl AbstractValue for TestValue {}

    #[derive(Clone)]
    struct SumTypeParams {
        depth: usize,
        desired_size: usize,
        expected_branch_size: usize,
    }

    impl Default for SumTypeParams {
        fn default() -> Self {
            Self {
                depth: 5,
                desired_size: 20,
                expected_branch_size: 5,
            }
        }
    }

    impl TestSumType {
        fn check_value(&self, pv: &PartialValue<TestValue>) -> bool {
            match (self, pv) {
                (_, PartialValue::Bottom) | (_, PartialValue::Top) => true,
                (Self::Leaf(None), _) => pv == &PartialValue::new_unit(),
                (Self::Leaf(Some(max)), PartialValue::Value(TestValue(val))) => val <= max,
                (Self::Branch(sop), PartialValue::PartialSum(ps)) => {
                    for (k, v) in &ps.0 {
                        if *k >= sop.len() {
                            return false;
                        }
                        let prod = &sop[*k];
                        if prod.len() != v.len() {
                            return false;
                        }
                        if !zip_eq(prod, v).all(|(lhs, rhs)| lhs.check_value(rhs)) {
                            return false;
                        }
                    }
                    true
                }
                _ => false,
            }
        }
    }

    impl Arbitrary for TestSumType {
        type Parameters = SumTypeParams;
        type Strategy = SBoxedStrategy<Self>;
        fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
            fn arb(params: SumTypeParams, set: &mut StrategySet) -> SBoxedStrategy<TestSumType> {
                use proptest::collection::vec;
                let int_strat = (0..usize::MAX).prop_map(|i| TestSumType::Leaf(Some(i)));
                let leaf_strat = prop_oneof![Just(TestSumType::Leaf(None)), int_strat];
                leaf_strat.prop_mutually_recursive(
                    params.depth as u32,
                    params.desired_size as u32,
                    params.expected_branch_size as u32,
                    set,
                    move |set| {
                        let params2 = params.clone();
                        vec(
                            vec(
                                set.get::<TestSumType, _>(move |set| arb(params2, set))
                                    .prop_map(Arc::new),
                                1..=params.expected_branch_size,
                            ),
                            1..=params.expected_branch_size,
                        )
                        .prop_map(TestSumType::Branch)
                        .sboxed()
                    },
                )
            }

            arb(params, &mut StrategySet::default())
        }
    }

    fn single_sum_strat(
        tag: usize,
        elems: Vec<Arc<TestSumType>>,
    ) -> impl Strategy<Value = PartialSum<TestValue>> {
        elems
            .iter()
            .map(Arc::as_ref)
            .map(any_partial_value_of_type)
            .collect::<Vec<_>>()
            .prop_map(move |elems| PartialSum::new_variant(tag, elems))
    }

    fn partial_sum_strat(
        variants: &[Vec<Arc<TestSumType>>],
    ) -> impl Strategy<Value = PartialSum<TestValue>> {
        // We have to clone the `variants` here but only as far as the Vec<Vec<Arc<_>>>
        let tagged_variants = variants.iter().cloned().enumerate().collect::<Vec<_>>();
        // The type annotation here (and the .boxed() enabling it) are just for documentation
        let sum_variants_strat: BoxedStrategy<Vec<PartialSum<TestValue>>> =
            subsequence(tagged_variants, 1..=variants.len())
                .prop_flat_map(|selected_variants| {
                    selected_variants
                        .into_iter()
                        .map(|(tag, elems)| single_sum_strat(tag, elems))
                        .collect::<Vec<_>>()
                })
                .boxed();
        sum_variants_strat.prop_map(|psums: Vec<PartialSum<TestValue>>| {
            let mut psums = psums.into_iter();
            let first = psums.next().unwrap();
            psums.fold(first, |mut a, b| {
                a.try_join_mut(b).unwrap();
                a
            })
        })
    }

    fn any_partial_value_of_type(
        ust: &TestSumType,
    ) -> impl Strategy<Value = PartialValue<TestValue>> {
        match ust {
            TestSumType::Leaf(None) => Just(PartialValue::new_unit()).boxed(),
            TestSumType::Leaf(Some(i)) => (0..*i)
                .prop_map(TestValue)
                .prop_map(PartialValue::from)
                .boxed(),
            TestSumType::Branch(sop) => partial_sum_strat(sop).prop_map(PartialValue::from).boxed(),
        }
    }

    fn any_partial_value_with(
        params: <TestSumType as Arbitrary>::Parameters,
    ) -> impl Strategy<Value = PartialValue<TestValue>> {
        any_with::<TestSumType>(params).prop_flat_map(|t| any_partial_value_of_type(&t))
    }

    fn any_partial_value() -> impl Strategy<Value = PartialValue<TestValue>> {
        any_partial_value_with(Default::default())
    }

    fn any_partial_values<const N: usize>() -> impl Strategy<Value = [PartialValue<TestValue>; N]> {
        any::<TestSumType>().prop_flat_map(|ust| {
            TryInto::<[_; N]>::try_into(
                (0..N)
                    .map(|_| any_partial_value_of_type(&ust))
                    .collect_vec(),
            )
            .unwrap()
        })
    }

    fn any_typed_partial_value() -> impl Strategy<Value = (TestSumType, PartialValue<TestValue>)> {
        any::<TestSumType>()
            .prop_flat_map(|t| any_partial_value_of_type(&t).prop_map(move |v| (t.clone(), v)))
    }

    proptest! {
        #[test]
        fn partial_value_type((tst, pv) in any_typed_partial_value()) {
            prop_assert!(tst.check_value(&pv))
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
            prop_assert!(meet == v2.clone().meet(v1.clone()), "meet not symmetric");
            prop_assert!(meet == meet.clone().meet(v1.clone()), "repeated meet should be a no-op");
            prop_assert!(meet == meet.clone().meet(v2.clone()), "repeated meet should be a no-op");

            let join = v1.clone().join(v2.clone());
            prop_assert!(join >= v1, "join not >=: {:#?}", &join);
            prop_assert!(join >= v2, "join not >=: {:#?}", &join);
            prop_assert!(join == v2.clone().join(v1.clone()), "join not symmetric");
            prop_assert!(join == join.clone().join(v1.clone()), "repeated join should be a no-op");
            prop_assert!(join == join.clone().join(v2.clone()), "repeated join should be a no-op");
        }

        #[test]
        fn lattice_associative([v1, v2, v3] in any_partial_values()) {
            let a = v1.clone().meet(v2.clone()).meet(v3.clone());
            let b = v1.clone().meet(v2.clone().meet(v3.clone()));
            prop_assert!(a==b, "meet not associative");

            let a = v1.clone().join(v2.clone()).join(v3.clone());
            let b = v1.clone().join(v2.clone().join(v3.clone()));
            prop_assert!(a==b, "join not associative")
        }
    }
}
