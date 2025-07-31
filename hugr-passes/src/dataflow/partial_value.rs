use ascent::Lattice;
use ascent::lattice::BoundedLattice;
use hugr_core::Node;
use hugr_core::types::{SumType, Type, TypeArg, TypeEnum, TypeRow};
use itertools::{Itertools, zip_eq};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use thiserror::Error;

use super::row_contains_bottom;

/// Trait for an underlying domain of abstract values which can form the *elements* of a
/// [`PartialValue`] and thus be used in dataflow analysis.
pub trait AbstractValue: Clone + std::fmt::Debug + PartialEq + Eq + Hash {
    /// Computes the join of two values (i.e. towards `Top``), if this is representable
    /// within the underlying domain. Return the new value, and whether this is different from
    /// the old `self`.
    ///
    /// If the join is not representable, return `None` - i.e., we should use [`PartialValue::Top`].
    ///
    /// The default checks equality between `self` and `other` and returns `(self,false)` if
    /// the two are identical, otherwise `None`.
    fn try_join(self, other: Self) -> Option<(Self, bool)> {
        (self == other).then_some((self, false))
    }

    /// Computes the meet of two values (i.e. towards `Bottom`), if this is representable
    /// within the underlying domain. Return the new value, and whether this is different from
    /// the old `self`.
    /// If the meet is not representable, return `None` - i.e., we should use [`PartialValue::Bottom`].
    ///
    /// The default checks equality between `self` and `other` and returns `(self, false)` if
    /// the two are identical, otherwise `None`.
    fn try_meet(self, other: Self) -> Option<(Self, bool)> {
        (self == other).then_some((self, false))
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

/// The output of an [`LoadFunction`](hugr_core::ops::LoadFunction) - a "pointer"
/// to a function at a specific node, instantiated with the provided type-args.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LoadedFunction<N> {
    /// The [FuncDefn](hugr_core::ops::FuncDefn) or `FuncDecl`` that was loaded
    pub func_node: N,
    /// The type arguments provided when loading
    pub args: Vec<TypeArg>,
}

/// A representation of a value of [`SumType`], that may have one or more possible tags,
/// with a [`PartialValue`] representation of each element-value of each possible tag.
#[derive(PartialEq, Clone, Eq)]
pub struct PartialSum<V, N = Node>(pub HashMap<usize, Vec<PartialValue<V, N>>>);

impl<V, N> PartialSum<V, N> {
    /// New instance for a single known tag.
    /// (Multi-tag instances can be created via [`Self::try_join_mut`].)
    pub fn new_variant(tag: usize, values: impl IntoIterator<Item = PartialValue<V, N>>) -> Self {
        Self(HashMap::from([(tag, Vec::from_iter(values))]))
    }

    /// The number of possible variants we know about. (NOT the number
    /// of tags possible for the value's type, whatever [`SumType`] that might be.)
    #[must_use]
    pub fn num_variants(&self) -> usize {
        self.0.len()
    }

    fn assert_invariants(&self) {
        assert_ne!(self.num_variants(), 0);
        for pv in self.0.values().flat_map(|x| x.iter()) {
            pv.assert_invariants();
        }
    }

    /// Whether this sum might have the specified tag
    #[must_use]
    pub fn supports_tag(&self, tag: usize) -> bool {
        self.0.contains_key(&tag)
    }

    /// Can this ever occur at runtime? See [`PartialValue::contains_bottom`]
    #[must_use]
    pub fn contains_bottom(&self) -> bool {
        self.0
            .iter()
            .all(|(_tag, elements)| row_contains_bottom(elements))
    }
}

impl<V: AbstractValue, N: PartialEq + PartialOrd> PartialSum<V, N> {
    /// Joins (towards `Top`) self with another [`PartialSum`]. If successful, returns
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
    /// * `Some(tag)` if the two [`PartialSum`]s both had rows with that `tag` but of different lengths
    /// * `None` if the two instances had no rows in common (i.e., the result is "Bottom")
    pub fn try_meet_mut(&mut self, other: Self) -> Result<bool, Option<usize>> {
        let mut changed = false;
        let mut keys_to_remove = vec![];
        for (k, v) in &self.0 {
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
}

/// Trait implemented by value types into which [`PartialValue`]s can be converted,
///
/// so long as the PV has no [Top](PartialValue::Top), [Bottom](PartialValue::Bottom)
/// or [`PartialSum`]s with more than one possible tag. See [`PartialSum::try_into_sum`]
/// and [`PartialValue::try_into_concrete`].
///
/// `V` is the type of [`AbstractValue`] from which `Self` can (fallibly) be constructed,
/// `N` is the type of [`HugrNode`](hugr_core::core::HugrNode) for function pointers
pub trait AsConcrete<V, N>: Sized {
    /// Kind of error raised when creating `Self` from a value `V`, see [`Self::from_value`]
    type ValErr: std::error::Error;
    /// Kind of error that may be raised when creating `Self` from a [Sum] of `Self`s,
    /// see [`Self::from_sum`]
    type SumErr: std::error::Error;

    /// Convert an abstract value into concrete
    fn from_value(val: V) -> Result<Self, Self::ValErr>;

    /// Convert a sum (of concrete values, already recursively converted) into concrete
    fn from_sum(sum: Sum<Self>) -> Result<Self, Self::SumErr>;

    /// Convert a function pointer into a concrete value
    fn from_func(func: LoadedFunction<N>) -> Result<Self, LoadedFunction<N>>;
}

impl<V: AbstractValue, N: std::fmt::Debug> PartialSum<V, N> {
    /// Turns this instance into a [Sum] of some "concrete" value type `C`,
    /// *if* this `PartialSum` has exactly one possible tag.
    ///
    /// # Errors
    ///
    /// If this `PartialSum` had multiple possible tags; or if `typ` was not a [`TypeEnum::Sum`]
    /// supporting the single possible tag with the correct number of elements and no row variables;
    /// or if converting a child element failed via [`PartialValue::try_into_concrete`].
    #[allow(clippy::type_complexity)] // Since C is a parameter, can't declare type aliases
    pub fn try_into_sum<C: AsConcrete<V, N>>(
        self,
        typ: &Type,
    ) -> Result<Sum<C>, ExtractValueError<V, N, C::ValErr, C::SumErr>> {
        if self.0.len() != 1 {
            return Err(ExtractValueError::MultipleVariants(self));
        }
        let (tag, v) = self.0.into_iter().exactly_one().unwrap();
        if let TypeEnum::Sum(st) = typ.as_type_enum() {
            if let Some(r) = st.get_variant(tag) {
                if let Ok(r) = TypeRow::try_from(r.clone()) {
                    if v.len() == r.len() {
                        return Ok(Sum {
                            tag,
                            values: zip_eq(v, r.iter())
                                .map(|(v, t)| v.try_into_concrete(t))
                                .collect::<Result<Vec<_>, _>>()?,
                            st: st.clone(),
                        });
                    }
                }
            }
        }
        Err(ExtractValueError::BadSumType {
            typ: typ.clone(),
            tag,
            num_elements: v.len(),
        })
    }
}

/// An error converting a [`PartialValue`] or [`PartialSum`] into a concrete value type
/// via [`PartialValue::try_into_concrete`] or [`PartialSum::try_into_sum`]
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum ExtractValueError<V, N, VE, SE> {
    #[error("PartialSum value had multiple possible tags: {0}")]
    MultipleVariants(PartialSum<V, N>),
    #[error("Value contained `Bottom`")]
    ValueIsBottom,
    #[error("Value contained `Top`")]
    ValueIsTop,
    #[error("Could not convert element from abstract value into concrete: {0}")]
    CouldNotConvert(V, #[source] VE),
    #[error("Could not build Sum from concrete element values")]
    CouldNotBuildSum(#[source] SE),
    #[error("Could not convert into concrete function pointer {0}")]
    CouldNotLoadFunction(LoadedFunction<N>),
    #[error("Expected a SumType with tag {tag} having {num_elements} elements, found {typ}")]
    BadSumType {
        typ: Type,
        tag: usize,
        num_elements: usize,
    },
}

impl<V: Clone, N: Clone> PartialSum<V, N> {
    /// If this Sum might have the specified `tag`, get the elements inside that tag.
    #[must_use]
    pub fn variant_values(&self, variant: usize) -> Option<Vec<PartialValue<V, N>>> {
        self.0.get(&variant).cloned()
    }
}

impl<V: PartialEq, N: PartialEq + PartialOrd> PartialOrd for PartialSum<V, N> {
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
            ord @ (Ordering::Greater | Ordering::Less) => ord,
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

impl<V: std::fmt::Debug, N: std::fmt::Debug> std::fmt::Debug for PartialSum<V, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: Hash, N: Hash> Hash for PartialSum<V, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (k, v) in &self.0 {
            k.hash(state);
            v.hash(state);
        }
    }
}

/// Wraps some underlying representation (knowledge) of values into a lattice
/// for use in dataflow analysis, including that an instance may be a [`PartialSum`]
/// of values of the underlying representation
#[derive(PartialEq, Clone, Eq, Hash, Debug)]
pub enum PartialValue<V, N = Node> {
    /// No possibilities known (so far)
    Bottom,
    /// The output of an [`LoadFunction`](hugr_core::ops::LoadFunction)
    LoadedFunction(LoadedFunction<N>),
    /// A single value (of the underlying representation)
    Value(V),
    /// Sum (with at least one, perhaps several, possible tags) of underlying values
    PartialSum(PartialSum<V, N>),
    /// Might be more than one distinct value of the underlying type `V`
    Top,
}

impl<V, N> From<V> for PartialValue<V, N> {
    fn from(v: V) -> Self {
        Self::Value(v)
    }
}

impl<V, N> From<PartialSum<V, N>> for PartialValue<V, N> {
    fn from(v: PartialSum<V, N>) -> Self {
        Self::PartialSum(v)
    }
}

impl<V, N> PartialValue<V, N> {
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
    #[must_use]
    pub fn new_unit() -> Self {
        Self::new_variant(0, [])
    }

    /// New instance of self for a [`LoadFunction`](hugr_core::ops::LoadFunction)
    pub fn new_load(func_node: N, args: impl Into<Vec<TypeArg>>) -> Self {
        Self::LoadedFunction(LoadedFunction {
            func_node,
            args: args.into(),
        })
    }

    /// Tells us whether this value might be a Sum with the specified `tag`
    pub fn supports_tag(&self, tag: usize) -> bool {
        match self {
            PartialValue::Bottom | PartialValue::Value(_) | PartialValue::LoadedFunction(_) => {
                false
            }
            PartialValue::PartialSum(ps) => ps.supports_tag(tag),
            PartialValue::Top => true,
        }
    }

    /// A value contains bottom means that it cannot occur during execution:
    /// it may be an artefact during bootstrapping of the analysis, or else
    /// the value depends upon a `panic` or a loop that
    /// [never terminates](super::TailLoopTermination::NeverBreaks).
    pub fn contains_bottom(&self) -> bool {
        match self {
            PartialValue::Bottom => true,
            PartialValue::Top | PartialValue::Value(_) | PartialValue::LoadedFunction(_) => false,
            PartialValue::PartialSum(ps) => ps.contains_bottom(),
        }
    }
}

impl<V: AbstractValue, N: Clone> PartialValue<V, N> {
    /// If this value might be a Sum with the specified `tag`, get the elements inside that tag.
    ///
    /// # Panics
    ///
    /// if the value is believed, for that tag, to have a number of values other than `len`
    pub fn variant_values(&self, tag: usize, len: usize) -> Option<Vec<PartialValue<V, N>>> {
        let vals = match self {
            PartialValue::Bottom | PartialValue::Value(_) | PartialValue::LoadedFunction(_) => {
                return None;
            }
            PartialValue::PartialSum(ps) => ps.variant_values(tag)?,
            PartialValue::Top => vec![PartialValue::Top; len],
        };
        assert_eq!(vals.len(), len);
        Some(vals)
    }
}

impl<V: AbstractValue, N: std::fmt::Debug> PartialValue<V, N> {
    /// Turns this instance into some "concrete" value type `C`, *if* it is a single value,
    /// or a [Sum](PartialValue::PartialSum) (of a single tag) convertible by
    /// [`PartialSum::try_into_sum`].
    ///
    /// # Errors
    ///
    /// If this `PartialValue` was `Top` or `Bottom`, or was a [`PartialSum`](PartialValue::PartialSum)
    /// that could not be converted into a [Sum] by [`PartialSum::try_into_sum`] (e.g. if `typ` is
    /// incorrect), or if that [Sum] could not be converted into a `V2`.
    pub fn try_into_concrete<C: AsConcrete<V, N>>(
        self,
        typ: &Type,
    ) -> Result<C, ExtractValueError<V, N, C::ValErr, C::SumErr>> {
        match self {
            Self::Value(v) => {
                C::from_value(v.clone()).map_err(|e| ExtractValueError::CouldNotConvert(v, e))
            }
            Self::LoadedFunction(lf) => {
                C::from_func(lf).map_err(ExtractValueError::CouldNotLoadFunction)
            }
            Self::PartialSum(ps) => {
                C::from_sum(ps.try_into_sum(typ)?).map_err(ExtractValueError::CouldNotBuildSum)
            }
            Self::Top => Err(ExtractValueError::ValueIsTop),
            Self::Bottom => Err(ExtractValueError::ValueIsBottom),
        }
    }
}

impl<V: AbstractValue, N: PartialEq + PartialOrd> Lattice for PartialValue<V, N> {
    fn join_mut(&mut self, other: Self) -> bool {
        self.assert_invariants();
        let mut old_self = Self::Top;
        std::mem::swap(self, &mut old_self);
        let (res, ch) = match (old_self, other) {
            (old @ Self::Top, _) | (old, Self::Bottom) => (old, false),
            (_, other @ Self::Top) | (Self::Bottom, other) => (other, true),
            (Self::Value(h1), Self::Value(h2)) => match h1.clone().try_join(h2) {
                Some((h3, b)) => (Self::Value(h3), b),
                None => (Self::Top, true),
            },
            (Self::LoadedFunction(lf1), Self::LoadedFunction(lf2))
                if lf1.func_node == lf2.func_node =>
            {
                // TODO we should also join the TypeArgs but at the moment these are ignored
                (Self::LoadedFunction(lf1), false)
            }
            (Self::PartialSum(mut ps1), Self::PartialSum(ps2)) => match ps1.try_join_mut(ps2) {
                Ok(ch) => (Self::PartialSum(ps1), ch),
                Err(_) => (Self::Top, true),
            },
            _ => (Self::Top, true),
        };
        *self = res;
        ch
    }

    fn meet_mut(&mut self, other: Self) -> bool {
        self.assert_invariants();
        let mut old_self = Self::Bottom;
        std::mem::swap(self, &mut old_self);
        let (res, ch) = match (old_self, other) {
            (old @ Self::Bottom, _) | (old, Self::Top) => (old, false),
            (_, other @ Self::Bottom) | (Self::Top, other) => (other, true),
            (Self::Value(h1), Self::Value(h2)) => match h1.try_meet(h2) {
                Some((h3, ch)) => (Self::Value(h3), ch),
                None => (Self::Bottom, true),
            },
            (Self::LoadedFunction(lf1), Self::LoadedFunction(lf2))
                if lf1.func_node == lf2.func_node =>
            {
                // TODO we should also meet the TypeArgs but at the moment these are ignored
                (Self::LoadedFunction(lf1), false)
            }
            (Self::PartialSum(mut ps1), Self::PartialSum(ps2)) => match ps1.try_meet_mut(ps2) {
                Ok(ch) => (Self::PartialSum(ps1), ch),
                Err(_) => (Self::Bottom, true),
            },
            _ => (Self::Bottom, true),
        };
        *self = res;
        ch
    }
}

impl<V: AbstractValue, N: PartialEq + PartialOrd> BoundedLattice for PartialValue<V, N> {
    fn top() -> Self {
        Self::Top
    }

    fn bottom() -> Self {
        Self::Bottom
    }
}

impl<V: PartialEq, N: PartialEq + PartialOrd> PartialOrd for PartialValue<V, N> {
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
            (Self::LoadedFunction(lf1), Self::LoadedFunction(lf2)) => {
                (lf1 == lf2).then_some(Ordering::Equal)
            }
            (Self::PartialSum(ps1), Self::PartialSum(ps2)) => ps1.partial_cmp(ps2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use ascent::{Lattice, lattice::BoundedLattice};
    use hugr_core::NodeIndex;
    use itertools::{Itertools as _, zip_eq};
    use prop::sample::subsequence;
    use proptest::prelude::*;

    use proptest_recurse::{StrategyExt, StrategySet};

    use super::{AbstractValue, LoadedFunction, PartialSum, PartialValue};

    #[derive(Debug, PartialEq, Eq, Clone)]
    enum TestSumType {
        Branch(Vec<Vec<Arc<TestSumType>>>),
        LeafVal(usize), // contains a TestValue <= this usize
        LeafPtr(usize), // contains a LoadedFunction with node <= this *usize*
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
                (_, PartialValue::Bottom | PartialValue::Top) => true,
                (Self::LeafVal(max), PartialValue::Value(TestValue(val))) => val <= max,
                (
                    Self::LeafPtr(max),
                    PartialValue::LoadedFunction(LoadedFunction { func_node, args }),
                ) => args.is_empty() && func_node.index() <= *max,
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
                let leaf_strat = prop_oneof![
                    (0..usize::MAX).prop_map(TestSumType::LeafVal),
                    // This is the maximum value accepted by portgraph::NodeIndex::new
                    (0..((2usize ^ 31) - 2)).prop_map(TestSumType::LeafPtr)
                ];
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
    ) -> impl Strategy<Value = PartialSum<TestValue>> + use<> {
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
    ) -> impl Strategy<Value = PartialValue<TestValue>> + use<> {
        match ust {
            TestSumType::LeafVal(i) => (0..=*i)
                .prop_map(TestValue)
                .prop_map(PartialValue::from)
                .boxed(),
            TestSumType::LeafPtr(i) => (0..=*i)
                .prop_map(|i| {
                    PartialValue::LoadedFunction(LoadedFunction {
                        func_node: portgraph::NodeIndex::new(i).into(),
                        args: vec![],
                    })
                })
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
            prop_assert!(tst.check_value(&pv));
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
            prop_assert!(a==b, "join not associative");
        }
    }
}
