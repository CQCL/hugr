use ascent::lattice::BoundedLattice;
use ascent::Lattice;
use hugr_core::ops::Value;
use hugr_core::types::{ConstTypeError, SumType, Type, TypeEnum, TypeRow};
use hugr_core::{HugrView, Wire};
use itertools::{zip_eq, Itertools};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::AbstractValue;

/// Trait for abstract values that can be wrapped by [PartialValue] for dataflow analysis.
/// (Allows the values to represent sums, but does not require this).
pub trait BaseValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// If the abstract value represents a [Sum] with a single known tag, deconstruct it
    /// into that tag plus the elements. The default just returns `None` which is
    /// appropriate if the abstract value never does (in which case [interpret_leaf_op]
    /// must produce a [PartialValue::new_variant] for any operation producing
    /// a sum).
    ///
    /// The signature is this way to optimize query/inspection (is-it-a-sum),
    /// at the cost of requiring more cloning during actual conversion
    /// (inside the lazy Iterator, or for the error case, as Self remains)
    ///
    /// [interpret_leaf_op]: super::DFContext::interpret_leaf_op
    /// [Sum]: TypeEnum::Sum
    /// [Tag]: hugr_core::ops::Tag
    fn as_sum(&self) -> Option<(usize, impl Iterator<Item = Self> + '_)> {
        let res: Option<(usize, <Vec<Self> as IntoIterator>::IntoIter)> = None;
        res
    }
}

/// Represents a sum with a single/known tag, abstracted over the representation of the elements.
/// (Identical to [Sum](hugr_core::ops::constant::Sum) except for the type abstraction.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sum<V> {
    /// The tag index of the variant.
    pub tag: usize,
    /// The value of the variant.
    ///
    /// Sum variants are always a row of values, hence the Vec.
    pub values: Vec<V>,
    /// The full type of the Sum, including the other variants.
    pub st: SumType,
}

/// A representation of a value of [SumType], that may have one or more possible tags,
/// with a [PartialValue] representation of each element-value of each possible tag.
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum<V>(pub HashMap<usize, Vec<PartialValue<V>>>);

impl<V> PartialSum<V> {
    /// New instance for a single known tag.
    /// (Multi-tag instances can be created via [Self::try_join_mut].)
    pub fn new_variant(tag: usize, values: impl IntoIterator<Item = PartialValue<V>>) -> Self {
        Self(HashMap::from([(tag, Vec::from_iter(values))]))
    }

    /// The number of possible variants we know about. (NOT the number
    /// of tags possible for the value's type, whatever [SumType] that might be.)
    pub fn num_variants(&self) -> usize {
        self.0.len()
    }
}

impl<V: BaseValue> PartialSum<V> {
    fn assert_invariants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

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
    /// Fails (without mutation) with the conflicting tag if any common rows have different lengths
    pub fn try_meet_mut(&mut self, other: Self) -> Result<bool, usize> {
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

    /// Whether this sum might have the specified tag
    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }

    /// Turns this instance into a [Sum] if it has exactly one possible tag,
    /// otherwise failing and returning itself back unmodified (also if there is another
    /// error, e.g. this instance is not described by `typ`).
    // ALAN is this too parametric? Should we fix V2 == Value? Is the 'Self' error useful (no?)
    pub fn try_into_value<V2: From<V> + TryFrom<Sum<V2>>>(
        self,
        typ: &Type,
    ) -> Result<Sum<V2>, Self> {
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
            Ok(values) => Ok(Sum {
                tag: *k,
                values,
                st: st.clone(),
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

/// Wraps some underlying representation of values (that `impl`s [BaseValue]) into
/// a lattice for use in dataflow analysis, including that an instance may be
/// a [PartialSum] of values of the underlying representation
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub struct PartialValue<V>(PVEnum<V>);

impl<V> PartialValue<V> {
    /// Allows to read the enum, which guarantees that we never return [PVEnum::Value]
    /// for a value whose [BaseValue::as_sum] is `Some` - any such value will be
    /// in the form of a [PVEnum::Sum] instead.
    pub fn as_enum(&self) -> &PVEnum<V> {
        &self.0
    }
}

/// The contents of a [PartialValue], i.e. used as a view.
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PVEnum<V> {
    /// No possibilities known (so far)
    Bottom,
    /// A single value (of the underlying representation)
    Value(V),
    /// Sum (with perhaps several possible tags) of underlying values
    Sum(PartialSum<V>),
    /// Might be more than one distinct value of the underlying type `V`
    Top,
}

impl<V: BaseValue> From<V> for PartialValue<V> {
    fn from(v: V) -> Self {
        v.as_sum()
            .map(|(tag, values)| Self::new_variant(tag, values.map(Self::from)))
            .unwrap_or(Self(PVEnum::Value(v)))
    }
}

impl<V> From<PartialSum<V>> for PartialValue<V> {
    fn from(v: PartialSum<V>) -> Self {
        Self(PVEnum::Sum(v))
    }
}

impl<V: BaseValue> PartialValue<V> {
    fn assert_invariants(&self) {
        match &self.0 {
            PVEnum::Sum(ps) => {
                ps.assert_invariants();
            }
            PVEnum::Value(v) => {
                assert!(v.as_sum().is_none())
            }
            _ => {}
        }
    }

    /// Extracts a value (in any representation supporting both leaf values and sums)
    // ALAN is this too parametric? Should we fix V2 == Value? Is the error useful (should we have 'Self') or is it a smell?
    pub fn try_into_value<V2: From<V> + TryFrom<Sum<V2>>>(
        self,
        typ: &Type,
    ) -> Result<V2, Option<<V2 as TryFrom<Sum<V2>>>::Error>> {
        match self.0 {
            PVEnum::Value(v) => Ok(V2::from(v.clone())),
            PVEnum::Sum(ps) => {
                let v = ps.try_into_value(typ).map_err(|_| None)?;
                V2::try_from(v).map_err(Some)
            }
            _ => Err(None),
        }
    }
}

impl<V: BaseValue> AbstractValue for PartialValue<V> {
    /// If this value might be a Sum with the specified `tag`, get the elements inside that tag.
    ///
    /// # Panics
    ///
    /// if the value is believed, for that tag, to have a number of values other than `len`
    fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
        let vals = match &self.0 {
            PVEnum::Bottom => return None,
            PVEnum::Value(v) => {
                assert!(v.as_sum().is_none());
                return None;
            }
            PVEnum::Sum(ps) => ps.variant_values(tag)?,
            PVEnum::Top => vec![PartialValue(PVEnum::Top); len],
        };
        assert_eq!(vals.len(), len);
        Some(vals)
    }

    /// Tells us whether this value might be a Sum with the specified `tag`
    fn supports_tag(&self, tag: usize) -> bool {
        match &self.0 {
            PVEnum::Bottom => false,
            PVEnum::Value(v) => {
                assert!(v.as_sum().is_none());
                false
            }
            PVEnum::Sum(ps) => ps.supports_tag(tag),
            PVEnum::Top => true,
        }
    }

    fn new_variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        PartialSum::new_variant(tag, values).into()
    }
}

impl TryFrom<Sum<Value>> for Value {
    type Error = ConstTypeError;

    fn try_from(value: Sum<Value>) -> Result<Self, Self::Error> {
        Self::sum(value.tag, value.values, value.st)
    }
}

impl<V: BaseValue> PartialValue<V>
where
    Value: From<V>,
{
    /// Turns this instance into a [Value], if it is either a single [value](PVEnum::Value) or
    /// a [sum](PVEnum::Sum) with a single known tag, extracting the desired type from a HugrView and Wire.
    ///
    /// # Errors
    /// `None` if the analysis did not result in a single value on that wire
    /// `Some(e)` if conversion to a [Value] produced a [ConstTypeError]
    ///
    /// # Panics
    ///
    /// If a [Type] for the specified wire could not be extracted from the Hugr
    pub fn try_into_wire_value(
        self,
        hugr: &impl HugrView,
        w: Wire,
    ) -> Result<Value, Option<ConstTypeError>> {
        let (_, typ) = hugr
            .out_value_types(w.node())
            .find(|(p, _)| *p == w.source())
            .unwrap();
        self.try_into_value(&typ)
    }
}

impl<V: BaseValue> Lattice for PartialValue<V> {
    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    fn join_mut(&mut self, other: Self) -> bool {
        self.assert_invariants();
        // println!("join {self:?}\n{:?}", &other);
        match (&self.0, other.0) {
            (PVEnum::Top, _) => false,
            (_, other @ PVEnum::Top) => {
                self.0 = other;
                true
            }
            (_, PVEnum::Bottom) => false,
            (PVEnum::Bottom, other) => {
                self.0 = other;
                true
            }
            (PVEnum::Value(h1), PVEnum::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    self.0 = PVEnum::Top;
                    true
                }
            }
            (PVEnum::Sum(_), PVEnum::Sum(ps2)) => {
                let Self(PVEnum::Sum(ps1)) = self else {
                    unreachable!()
                };
                match ps1.try_join_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        self.0 = PVEnum::Top;
                        true
                    }
                }
            }
            (PVEnum::Value(ref v), PVEnum::Sum(_)) | (PVEnum::Sum(_), PVEnum::Value(ref v)) => {
                assert!(v.as_sum().is_none());
                self.0 = PVEnum::Top;
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
        match (&self.0, other.0) {
            (PVEnum::Bottom, _) => false,
            (_, other @ PVEnum::Bottom) => {
                self.0 = other;
                true
            }
            (_, PVEnum::Top) => false,
            (PVEnum::Top, other) => {
                self.0 = other;
                true
            }
            (PVEnum::Value(h1), PVEnum::Value(h2)) => {
                if h1 == &h2 {
                    false
                } else {
                    self.0 = PVEnum::Bottom;
                    true
                }
            }
            (PVEnum::Sum(_), PVEnum::Sum(ps2)) => {
                let ps1 = match &mut self.0 {
                    PVEnum::Sum(ps1) => ps1,
                    _ => unreachable!(),
                };
                match ps1.try_meet_mut(ps2) {
                    Ok(ch) => ch,
                    Err(_) => {
                        self.0 = PVEnum::Bottom;
                        true
                    }
                }
            }
            (PVEnum::Value(ref v), PVEnum::Sum(_)) | (PVEnum::Sum(_), PVEnum::Value(ref v)) => {
                assert!(v.as_sum().is_none());
                self.0 = PVEnum::Bottom;
                true
            }
        }
    }
}

impl<V: BaseValue> BoundedLattice for PartialValue<V> {
    fn top() -> Self {
        Self(PVEnum::Top)
    }

    fn bottom() -> Self {
        Self(PVEnum::Bottom)
    }
}

impl<V: PartialEq> PartialOrd for PartialValue<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (&self.0, &other.0) {
            (PVEnum::Bottom, PVEnum::Bottom) => Some(Ordering::Equal),
            (PVEnum::Top, PVEnum::Top) => Some(Ordering::Equal),
            (PVEnum::Bottom, _) => Some(Ordering::Less),
            (_, PVEnum::Bottom) => Some(Ordering::Greater),
            (PVEnum::Top, _) => Some(Ordering::Greater),
            (_, PVEnum::Top) => Some(Ordering::Less),
            (PVEnum::Value(v1), PVEnum::Value(v2)) => (v1 == v2).then_some(Ordering::Equal),
            (PVEnum::Sum(ps1), PVEnum::Sum(ps2)) => ps1.partial_cmp(ps2),
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

    use super::{BaseValue, PVEnum, PartialSum, PartialValue};
    use crate::dataflow::AbstractValue;

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestSumType {
        Branch(Vec<Vec<Arc<TestSumType>>>),
        /// None => unit, Some => TestValue <= this *usize*
        Leaf(Option<usize>),
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestValue(usize);

    impl BaseValue for TestValue {}

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
            match (self, pv.as_enum()) {
                (_, PVEnum::Bottom) | (_, PVEnum::Top) => true,
                (Self::Leaf(None), _) => pv == &PartialValue::new_unit(),
                (Self::Leaf(Some(max)), PVEnum::Value(TestValue(val))) => val <= max,
                (Self::Branch(sop), PVEnum::Sum(ps)) => {
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
        variants: &Vec<Vec<Arc<TestSumType>>>,
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

            let join = v1.clone().join(v2.clone());
            prop_assert!(join >= v1, "join not >=: {:#?}", &join);
            prop_assert!(join >= v2, "join not >=: {:#?}", &join);
        }
    }
}
