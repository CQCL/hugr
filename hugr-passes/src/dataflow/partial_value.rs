use ascent::lattice::BoundedLattice;
use ascent::Lattice;
use hugr_core::ops::Value;
use hugr_core::types::{ConstTypeError, SumType, Type, TypeEnum, TypeRow};
use hugr_core::{HugrView, Wire};
use itertools::{zip_eq, Itertools};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Trait for values which can be deconstructed into Sums (with a single known tag).
/// Required for values used in dataflow analysis.
pub trait AbstractValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// Deconstruct a value into a single known tag plus a row of values, if it is a [Sum].
    /// Note that one can just always return `None` but this will mean the analysis
    /// is unable to understand untupling, and may give inconsistent results wrt. [Tag]
    /// operations, etc.
    ///
    /// The signature is this way to optimize query/inspection (is-it-a-sum),
    /// at the cost of requiring more cloning during actual conversion
    /// (inside the lazy Iterator, or for the error case, as Self remains)
    ///
    /// [Sum]: TypeEnum::Sum
    /// [Tag]: hugr_core::ops::Tag
    fn as_sum(&self) -> Option<(usize, impl Iterator<Item = Self> + '_)>;
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

impl<V: AbstractValue> PartialSum<V> {
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

/// Wraps some underlying representation (knowledge) of values into a lattice
/// for use in dataflow analysis, including that an instance may be a [PartialSum]
/// of values of the underlying representation
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub struct PartialValue<V>(PVEnum<V>);

impl<V> PartialValue<V> {
    /// Allows to read the enum, which guarantees that we never return [PVEnum::Value]
    /// for a value whose [AbstractValue::as_sum] is `Some` - any such value will be
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

impl<V: AbstractValue> From<V> for PartialValue<V> {
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

impl<V: AbstractValue> PartialValue<V> {
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

    /// New instance of a sum with a single known tag.
    pub fn new_variant(tag: usize, values: impl IntoIterator<Item = Self>) -> Self {
        PartialSum::new_variant(tag, values).into()
    }

    /// New instance of unit type (i.e. the only possible value, with no contents)
    pub fn new_unit() -> Self {
        Self::new_variant(0, [])
    }

    /// If this value might be a Sum with the specified `tag`, get the elements inside that tag.
    ///
    /// # Panics
    ///
    /// if the value is believed, for that tag, to have a number of values other than `len`
    pub fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V>>> {
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
    pub fn supports_tag(&self, tag: usize) -> bool {
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

impl TryFrom<Sum<Value>> for Value {
    type Error = ConstTypeError;

    fn try_from(value: Sum<Value>) -> Result<Self, Self::Error> {
        Self::sum(value.tag, value.values, value.st)
    }
}

impl<V: AbstractValue> PartialValue<V>
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

impl<V: AbstractValue> Lattice for PartialValue<V> {
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

impl<V: AbstractValue> BoundedLattice for PartialValue<V> {
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
    use itertools::{zip_eq, Either, Itertools as _};
    use proptest::prelude::*;

    use hugr_core::{
        std_extensions::arithmetic::int_types::{self, ConstInt, INT_TYPES, LOG_WIDTH_BOUND},
        types::{Type, TypeArg, TypeEnum},
    };

    use super::{PVEnum, PartialSum, PartialValue};
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
                Self::Unit => Just(PartialValue::new_unit()).boxed(),
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
            match (self, pv.as_enum()) {
                (_, PVEnum::Bottom) | (_, PVEnum::Top) => true,
                (_, PVEnum::Value(v)) => self.get_type() == v.get_type(),
                (TestSumType::Branch(_, sop), PVEnum::Sum(ps)) => {
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
                (Self::Leaf(l), PVEnum::Sum(ps)) => l.type_check(ps),
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
                pvs.prop_map(move |pvs| PartialValue::new_variant(index, pvs))
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
