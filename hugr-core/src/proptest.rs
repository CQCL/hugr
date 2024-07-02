use ::proptest::collection::vec;
use ::proptest::prelude::*;
use lazy_static::lazy_static;
use smol_str::SmolStr;

use crate::Hugr;

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
/// The types [Type], [TypeEnum], [SumType], [FunctionType], [TypeArg],
/// [TypeParam], as well as several others, form a mutually recursive hierarchy.
///
/// The proptest [proptest::strategy::Strategy::prop_recursive] is inadequate to
/// generate values for these types.  Instead, the Arbitrary instances take a
/// `RecursionDepth` as their (or part of their)
/// [proptest::arbitrary::Arbitrary::Parameters]. We then use that parameter to
/// generate children of that value. Usually we forward it unchanged, but in
/// crucial locations(grep for `descend`) we instead forward the `descend` of
/// it.
///
/// Consider the tree of values generated. Each node is labelled with a
/// [RecursionDepth].
///
/// Consider a path between two different nodes of the same kind(e.g. two
/// [Type]s, or two [FunctionType]s).  The path must be non-increasing in
/// [RecursionDepth] because each child's [RecursionDepth] is derived from it's
/// parents.
///
/// We must maintain the invariant that the [RecursionDepth] of the start of the
/// path is strictly greater than the [RecursionDepth] of the end of the path.
///
/// With this invariant in place we are guaranteed to generate a finite tree
/// because there are only finitely many different types a node can take.
///
/// We could instead use the `proptest-recurse` crate to implement [Arbitrary]
/// impls for these mutually recursive types. We did try, but it wasn't simple
/// enough to be obviously better, so this will do for now.
pub struct RecursionDepth(usize);

impl RecursionDepth {
    const DEFAULT_RECURSION_DEPTH: usize = 4;
    pub fn descend(&self) -> Self {
        if self.leaf() {
            *self
        } else {
            Self(self.0 - 1)
        }
    }

    pub fn leaf(&self) -> bool {
        self.0 == 0
    }

    pub fn new() -> Self {
        Self(Self::DEFAULT_RECURSION_DEPTH)
    }
}

impl Default for RecursionDepth {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: TryInto<usize>> From<I> for RecursionDepth
where
    <I as TryInto<usize>>::Error: std::fmt::Debug,
{
    fn from(s: I) -> Self {
        Self(s.try_into().unwrap())
    }
}

lazy_static! {
    /// A strategy for a [String] suitable for an [IdentList].
    /// Shrinks to contain only ASCII letters.
    static ref ANY_IDENT_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            // we shrink to more readable (i.e. :alpha:) names
            string_regex(r"[[:alpha:]]+").unwrap(),
            string_regex(crate::hugr::ident::PATH_COMPONENT_REGEX_STR).unwrap(),
        ].sboxed()
    };

    /// A strategy for an arbitrary nonempty [String].
    /// Shrinks to contain only ASCII letters.
    static ref ANY_NONEMPTY_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            // we shrink to more readable (i.e. :alpha:) names
            string_regex(r"[[:alpha:]]+").unwrap(),
            string_regex(r".+").unwrap(),
        ].sboxed()
    };

    /// A strategy for an arbitrary [String].
    /// Shrinks to contain only ASCII letters.
    static ref ANY_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            // we shrink to more readable (i.e. :alpha:) names
            string_regex(r"[[:alpha:]]*").unwrap(),
            string_regex(r".*").unwrap(),
        ].sboxed()
    };

    /// A strategy for an arbitrary non-recursive [serde_yaml::Value].
    /// In particular, no `Mapping`, `Sequence`, or `Tagged`.
    ///
    /// This is used as the base strategy for the general
    /// [recursive](Strategy::prop_recursive) strategy.
    static ref ANY_SERDE_YAML_VALUE_LEAF: SBoxedStrategy<serde_yaml::Value> = {
        use serde_yaml::value::Value;
        prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map_into(),
            any::<u64>().prop_map_into(),
            any::<i64>().prop_map_into(),
            // floats don't round trip !?!
            // any::<f64>().prop_map_into(),
            Just(Value::Number(3.into())),
            any_string().prop_map_into(),
        ].sboxed()
    };

    /// A strategy that returns one of a fixed number of example [Hugr]s.
    static ref ANY_HUGR: SBoxedStrategy<Hugr>= {
        // TODO we need more examples
        // This is currently used for Value::Function
        // With more uses we may need variants that return more constrained
        // HUGRs.
        prop_oneof![
            Just(crate::builder::test::simple_dfg_hugr()),
        ].sboxed()
    };
}

pub fn any_nonempty_string() -> SBoxedStrategy<String> {
    ANY_NONEMPTY_STRING.to_owned()
}

pub fn any_nonempty_smolstr() -> SBoxedStrategy<SmolStr> {
    ANY_NONEMPTY_STRING.to_owned().prop_map_into().sboxed()
}

pub fn any_ident_string() -> SBoxedStrategy<String> {
    ANY_IDENT_STRING.to_owned()
}

pub fn any_string() -> SBoxedStrategy<String> {
    ANY_STRING.to_owned()
}

pub fn any_smolstr() -> SBoxedStrategy<SmolStr> {
    ANY_STRING.clone().prop_map_into().sboxed()
}

pub fn any_serde_yaml_value() -> impl Strategy<Value = serde_yaml::Value> {
    // use serde_yaml::value::{Tag, TaggedValue, Value};
    ANY_SERDE_YAML_VALUE_LEAF
        .clone()
        .prop_recursive(
            3,  // No more than 3 branch levels deep
            32, // Target around 32 total elements
            3,  // Each collection is up to 3 elements long
            |element| {
                prop_oneof![
                    // TODO TaggedValue doesn't roundtrip through JSON
                    // (any_nonempty_string().prop_map(Tag::new), element.clone()).prop_map(|(tag, value)| Value::Tagged(Box::new(TaggedValue { tag, value }))),
                    proptest::collection::vec(element.clone(), 0..3).prop_map_into(),
                    vec((any_string().prop_map_into(), element.clone()), 0..3)
                        .prop_map(|x| x.into_iter().collect::<serde_yaml::Mapping>().into())
                ]
            },
        )
        .boxed()
}

pub fn any_hugr() -> SBoxedStrategy<Hugr> {
    ANY_HUGR.to_owned()
}
