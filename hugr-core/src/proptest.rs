//! Generator functions for property testing the Hugr data structures.

use ::proptest::collection::vec;
use ::proptest::prelude::*;
use smol_str::SmolStr;
use std::sync::LazyLock;

use crate::Hugr;

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
/// The types [Type], [`TypeEnum`], [`SumType`], [`FunctionType`], [`TypeArg`],
/// [`TypeParam`], as well as several others, form a mutually recursive hierarchy.
///
/// The proptest [`proptest::strategy::Strategy::prop_recursive`] is inadequate to
/// generate values for these types.  Instead, the Arbitrary instances take a
/// `RecursionDepth` as their (or part of their)
/// [`proptest::arbitrary::Arbitrary::Parameters`]. We then use that parameter to
/// generate children of that value. Usually we forward it unchanged, but in
/// crucial locations(grep for `descend`) we instead forward the `descend` of
/// it.
///
/// Consider the tree of values generated. Each node is labelled with a
/// [`RecursionDepth`].
///
/// Consider a path between two different nodes of the same kind(e.g. two
/// [Type]s, or two [`FunctionType`]s).  The path must be non-increasing in
/// [`RecursionDepth`] because each child's [`RecursionDepth`] is derived from it's
/// parents.
///
/// We must maintain the invariant that the [`RecursionDepth`] of the start of the
/// path is strictly greater than the [`RecursionDepth`] of the end of the path.
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
    /// Decrement the recursion depth counter.
    #[must_use]
    pub fn descend(&self) -> Self {
        if self.leaf() { *self } else { Self(self.0 - 1) }
    }

    /// Returns `true` if the recursion depth counter is zero.
    #[must_use]
    pub fn leaf(&self) -> bool {
        self.0 == 0
    }

    /// Create a new [`RecursionDepth`] with the default recursion depth.
    #[must_use]
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

/// A strategy for a [String] suitable for an [IdentList].
/// Shrinks to contain only ASCII letters.
static ANY_IDENT_STRING: LazyLock<SBoxedStrategy<String>> = LazyLock::new(|| {
    use proptest::string::string_regex;
    prop_oneof![
        // we shrink to more readable (i.e. :alpha:) names
        string_regex(r"[[:alpha:]]+").unwrap(),
        string_regex(crate::hugr::ident::PATH_COMPONENT_REGEX_STR).unwrap(),
    ]
    .sboxed()
});

/// A strategy for an arbitrary nonempty [String].
/// Shrinks to contain only ASCII letters.
static ANY_NONEMPTY_STRING: LazyLock<SBoxedStrategy<String>> = LazyLock::new(|| {
    use proptest::string::string_regex;
    prop_oneof![
        // we shrink to more readable (i.e. :alpha:) names
        string_regex(r"[[:alpha:]]+").unwrap(),
        string_regex(r".+").unwrap(),
    ]
    .sboxed()
});

/// A strategy for an arbitrary [String].
/// Shrinks to contain only ASCII letters.
static ANY_STRING: LazyLock<SBoxedStrategy<String>> = LazyLock::new(|| {
    use proptest::string::string_regex;
    prop_oneof![
        // we shrink to more readable (i.e. :alpha:) names
        string_regex(r"[[:alpha:]]*").unwrap(),
        string_regex(r".*").unwrap(),
    ]
    .sboxed()
});

/// A strategy for an arbitrary non-recursive [serde_json::Value].
/// In particular, no `Array` or `Object`
///
/// This is used as the base strategy for the general
/// [recursive](Strategy::prop_recursive) strategy.
static ANY_SERDE_JSON_VALUE_LEAF: LazyLock<SBoxedStrategy<serde_json::Value>> =
    LazyLock::new(|| {
        use serde_json::value::Value;
        prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map_into(),
            any::<u64>().prop_map_into(),
            any::<i64>().prop_map_into(),
            // floats don't round trip !?!
            // any::<f64>().prop_map_into(),
            Just(Value::Number(3.into())),
            any_string().prop_map_into(),
        ]
        .sboxed()
    });

/// A strategy that returns one of a fixed number of example [Hugr]s.
static ANY_HUGR: LazyLock<SBoxedStrategy<Hugr>> = LazyLock::new(|| {
    // TODO we need more examples
    // This is currently used for Value::Function
    // With more uses we may need variants that return more constrained
    // HUGRs.
    prop_oneof![Just(crate::builder::test::simple_dfg_hugr()),].sboxed()
});

/// A strategy for generating an arbitrary nonempty [String].
pub fn any_nonempty_string() -> SBoxedStrategy<String> {
    ANY_NONEMPTY_STRING.to_owned()
}

/// A strategy for generating an arbitrary nonempty [`SmolStr`].
pub fn any_nonempty_smolstr() -> SBoxedStrategy<SmolStr> {
    ANY_NONEMPTY_STRING.to_owned().prop_map_into().sboxed()
}

/// A strategy for generating an arbitrary nonempty identifier [String].
pub fn any_ident_string() -> SBoxedStrategy<String> {
    ANY_IDENT_STRING.to_owned()
}

/// A strategy for generating an arbitrary [String].
pub fn any_string() -> SBoxedStrategy<String> {
    ANY_STRING.to_owned()
}

/// A strategy for generating an arbitrary [`SmolStr`].
pub fn any_smolstr() -> SBoxedStrategy<SmolStr> {
    ANY_STRING.clone().prop_map_into().sboxed()
}

/// A strategy for generating an arbitrary [`serde_json::Value`].
pub fn any_serde_json_value() -> impl Strategy<Value = serde_json::Value> {
    ANY_SERDE_JSON_VALUE_LEAF
        .clone()
        .prop_recursive(
            3,  // No more than 3 branch levels deep
            32, // Target around 32 total elements
            3,  // Each collection is up to 3 elements long
            |element| {
                prop_oneof![
                    proptest::collection::vec(element.clone(), 0..3).prop_map_into(),
                    vec((any_string().prop_map_into(), element.clone()), 0..3).prop_map(|x| x
                        .into_iter()
                        .collect::<serde_json::Map<String, serde_json::Value>>()
                        .into())
                ]
            },
        )
        .boxed()
}

/// A strategy for generating an arbitrary HUGR.
pub fn any_hugr() -> SBoxedStrategy<Hugr> {
    ANY_HUGR.to_owned()
}
