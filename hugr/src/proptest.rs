use ::proptest::collection::vec;
use ::proptest::prelude::*;
use ::proptest::strategy::Strategy;
use lazy_static::lazy_static;
use smol_str::SmolStr;

#[derive(Clone, Copy, Debug)]
pub struct TypeDepth(usize);

impl TypeDepth {
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
}

impl Default for TypeDepth {
    fn default() -> Self {
        Self(3)
    }
}

impl From<usize> for TypeDepth {
    fn from(s: usize) -> Self {
        Self(s)
    }
}

lazy_static! {
    static ref ANY_IDENT_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            string_regex(r"[[:alpha:]]{1,3}").unwrap(),
            string_regex(crate::hugr::ident::PATH_COMPONENT_NICE_REGEX_STR).unwrap(),
            string_regex(crate::hugr::ident::PATH_COMPONENT_REGEX_STR).unwrap(),
        ].sboxed()
    };

    static ref ANY_NONEMPTY_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            string_regex(r"[[:alpha:]]{1,3}").unwrap(),
            string_regex(r"[[:alpha:]]+").unwrap(),
            string_regex(r".+").unwrap(),
        ].sboxed()
    };

    static ref ANY_STRING: SBoxedStrategy<String> = {
        use proptest::string::string_regex;
        prop_oneof![
            string_regex(r"[[:alpha:]]{0,3}").unwrap(),
            string_regex(r"[[:alpha:]]*").unwrap(),
            string_regex(r".*").unwrap(),
        ].sboxed()
    };

    static ref ANY_SERDE_YAML_VALUE_LEAF: SBoxedStrategy<serde_yaml::Value> = {
        use serde_yaml::value::Value;
        prop_oneof![
            Just(Value::Null),
            any::<bool>().prop_map_into(),
            any::<u64>().prop_map_into(),
            any::<i64>().prop_map_into(),
            // any::<f64>().prop_map_into(),
            Just(Value::Number(3.into())),
            any_string().prop_map_into(),
        ].sboxed()
    };
        // .prop_recursive(
        //     3,  // No more than 3 branch levels deep
        //     32, // Target around 32 total elements
        //     3,  // Each collection is up to 3 elements long
        //     |element| prop_oneof![
        //         (any_string().prop_map(Tag::new), element.clone()).prop_map(|(tag, value)| Value::Tagged(Box::new(TaggedValue { tag, value }))),
        //         proptest::collection::vec(element.clone(), 0..3).prop_map_into(),
        //     ]
        // ).sboxed()
}

pub fn any_nonempty_string() -> SBoxedStrategy<String> {
    ANY_NONEMPTY_STRING.clone()
}

pub fn any_nonempty_smolstr() -> SBoxedStrategy<SmolStr> {
    ANY_NONEMPTY_STRING.clone().prop_map_into().sboxed()
}

pub fn any_ident_string() -> SBoxedStrategy<String> {
    ANY_IDENT_STRING.clone()
}

pub fn any_string() -> SBoxedStrategy<String> {
    ANY_STRING.clone()
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
                    // TaggedValue doesn't roundtrip through JSON
                    // (any_nonempty_string().prop_map(Tag::new), element.clone()).prop_map(|(tag, value)| Value::Tagged(Box::new(TaggedValue { tag, value }))),
                    proptest::collection::vec(element.clone(), 0..3).prop_map_into(),
                    vec((any_string().prop_map_into(), element.clone()), 0..3)
                        .prop_map(|x| x.into_iter().collect::<serde_yaml::Mapping>().into())
                ]
            },
        )
        .boxed()
}
