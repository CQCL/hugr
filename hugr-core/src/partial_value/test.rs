use std::sync::Arc;

use itertools::{zip_eq, Either, Itertools as _};
use lazy_static::lazy_static;
use proptest::prelude::*;

use crate::{
    ops::Value,
    std_extensions::arithmetic::int_types::{
        self, get_log_width, ConstInt, INT_TYPES, LOG_WIDTH_BOUND,
    },
    types::{CustomType, Type, TypeEnum},
};

use super::{PartialSum, PartialValue, ValueHandle, ValueKey};
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
    fn assert_invariants(&self) {
        match self {
            Self::Int(t) => {
                if let TypeEnum::Extension(ct) = t.as_type_enum() {
                    assert_eq!("int", ct.name());
                    assert_eq!(&int_types::EXTENSION_ID, ct.extension());
                } else {
                    panic!("Expected int type, got {:#?}", t);
                }
            }
            _ => (),
        }
    }

    fn get_type(&self) -> Type {
        match self {
            Self::Int(t) => t.clone(),
            Self::Unit => Type::UNIT,
        }
    }

    fn type_check(&self, ps: &PartialSum) -> bool {
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

    fn partial_value_strategy(self) -> impl Strategy<Value = PartialValue> {
        match self {
            Self::Int(t) => {
                let TypeEnum::Extension(ct) = t.as_type_enum() else {
                    unreachable!()
                };
                let lw = get_log_width(&ct.args()[0]).unwrap();
                (0u64..(1 << (2u64.pow(lw as u32) - 1)))
                    .prop_map(move |x| {
                        let ki = ConstInt::new_u(lw, x).unwrap();
                        ValueHandle::new(ValueKey::new(ki.clone()), Arc::new(ki.into())).into()
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
        let int_strat = (0..LOG_WIDTH_BOUND).prop_map(|i| Self::Int(INT_TYPES[i as usize].clone()));
        prop_oneof![Just(TestSumLeafType::Unit), int_strat].boxed()
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum TestSumType {
    Branch(usize, Vec<Vec<Arc<TestSumType>>>),
    Leaf(TestSumLeafType),
}

impl TestSumType {
    const UNIT: TestSumLeafType = TestSumLeafType::Unit;

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
            TestSumType::Leaf(l) => {
                l.assert_invariants();
            }
            _ => (),
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
                    .map(|row| row.iter().map(|x| x.get_type()).collect_vec().into()),
            ),
            TestSumType::Leaf(l) => l.get_type(),
        }
    }

    fn type_check(&self, pv: &PartialValue) -> bool {
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
        ust.assert_invariants();
    }
}

fn any_partial_value_of_type(ust: TestSumType) -> impl Strategy<Value = PartialValue> {
    ust.select().prop_flat_map(|x| match x {
        Either::Left(l) => l.partial_value_strategy().boxed(),
        Either::Right((index, usts)) => {
            let pvs = usts
                .into_iter()
                .map(|x| {
                    any_partial_value_of_type(
                        Arc::<TestSumType>::try_unwrap(x).unwrap_or_else(|x| x.as_ref().clone()),
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
    any::<TestSumType>()
        .prop_flat_map(|t| any_partial_value_of_type(t.clone()).prop_map(move |v| (t.clone(), v)))
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
        prop_assert!(v <= PartialValue::Top);
        prop_assert!(v >= PartialValue::Bottom);
    }

    #[test]
    fn meet_join_self_noop(v1 in any_partial_value()) {
        let mut subject = v1.clone();
        assert!(!subject.join_mut(v1.clone()));
        assert_eq!(subject, v1);
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
