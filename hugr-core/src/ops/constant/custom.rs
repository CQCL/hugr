//! Representation of custom constant values.
//!
//! These can be used as [`Const`] operations in HUGRs.
//!
//! [`Const`]: crate::ops::Const

use std::any::Any;
use std::hash::{Hash, Hasher};

use downcast_rs::{Downcast, impl_downcast};
use thiserror::Error;

use crate::IncomingPort;
use crate::extension::resolution::{
    ExtensionResolutionError, WeakExtensionRegistry, resolve_type_extensions,
};
use crate::macros::impl_box_clone;
use crate::types::{CustomCheckFailure, Type};

use super::{Value, ValueName};

/// Extensible constant values.
///
/// We use [typetag] to provide an `impl Serialize for dyn CustomConst`, and
/// similarly [serde::Deserialize]. When implementing this trait, include the
/// [`#[typetag::serde]`](typetag) attribute to enable serialization.
///
/// Note that when serializing through the [`dyn CustomConst`] a dictionary will
/// be serialized with two attributes, `"c"`  the tag and `"v"` the
/// `CustomConst`:
///
#[cfg_attr(not(miri), doc = "```")] // this doctest depends on typetag, so fails with miri
#[cfg_attr(miri, doc = "```ignore")]
/// use serde::{Serialize,Deserialize};
/// use hugr::{
///   types::Type,ops::constant::{OpaqueValue, ValueName, CustomConst},
///   extension::ExtensionSet, std_extensions::arithmetic::int_types};
/// use serde_json::json;
///
/// #[derive(std::fmt::Debug, Clone, Hash, Serialize,Deserialize)]
/// struct CC(i64);
///
/// #[typetag::serde]
/// impl CustomConst for CC {
///   fn name(&self) -> ValueName { "CC".into() }
///   fn get_type(&self) -> Type { int_types::INT_TYPES[5].clone() }
/// }
///
/// assert_eq!(serde_json::to_value(CC(2)).unwrap(), json!(2));
/// assert_eq!(serde_json::to_value(&CC(2) as &dyn CustomConst).unwrap(), json!({
///   "c": "CC",
///   "v": 2
/// }));
/// ```
#[typetag::serde(tag = "c", content = "v")]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + TryHash + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> ValueName;

    /// Check the value.
    fn validate(&self) -> Result<(), CustomCheckFailure> {
        Ok(())
    }

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    ///
    /// If the type implements `PartialEq`, use [`downcast_equal_consts`] to compare the values.
    ///
    /// Note that this does not require any equivalent of [Eq]: it is permissible to return
    /// `false` if in doubt, and in particular, there is no requirement for reflexivity
    /// (i.e. `x.equal_consts(x)` can be `false`). However, we do expect both
    /// symmetry (`x.equal_consts(y) == y.equal_consts(x)`) and transitivity
    /// (if `x.equal_consts(y) && y.equal_consts(z)` then `x.equal_consts(z)`).
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overridden
        false
    }

    /// Update the extensions associated with the internal values.
    ///
    /// This is used to ensure that any extension reference [`CustomConst::get_type`] remains
    /// valid when serializing and deserializing the constant.
    ///
    /// See the helper methods in [`crate::extension::resolution`].
    fn update_extensions(
        &mut self,
        _extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        Ok(())
    }

    /// Report the type.
    fn get_type(&self) -> Type;
}

/// Fallible hash function.
///
/// Prerequisite for `CustomConst`. Allows to declare a custom hash function,
/// but the easiest options are either to `impl TryHash for ... {}` to indicate
/// "not hashable", or else to implement/derive [Hash].
pub trait TryHash {
    /// Hashes the value, if possible; else return `false` without mutating the `Hasher`.
    /// This relates with [`CustomConst::equal_consts`] just like [Hash] with [Eq]:
    /// * if `x.equal_consts(y)` ==> `x.try_hash(s)` behaves equivalently to `y.try_hash(s)`
    /// * if `x.hash(s)` behaves differently from `y.hash(s)` ==> `x.equal_consts(y) == false`
    ///
    /// As with [Hash], these requirements can trivially be satisfied by either
    /// * `equal_consts` always returning `false`, or
    /// * `try_hash` always behaving the same (e.g. returning `false`, as it does by default)
    ///
    /// Note: uses `dyn` rather than being parametrized by `<H: Hasher>` to be object-safe.
    fn try_hash(&self, _state: &mut dyn Hasher) -> bool {
        false
    }
}

impl<T: Hash> TryHash for T {
    fn try_hash(&self, mut st: &mut dyn Hasher) -> bool {
        Hash::hash(self, &mut st);
        true
    }
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

/// Const equality for types that have `PartialEq`
pub fn downcast_equal_consts<T: CustomConst + PartialEq>(
    constant: &T,
    other: &dyn CustomConst,
) -> bool {
    if let Some(other) = other.as_any().downcast_ref::<T>() {
        constant == other
    } else {
        false
    }
}

/// Serialize any `CustomConst` using the `impl Serialize for &dyn CustomConst`.
fn serialize_custom_const(cc: &dyn CustomConst) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::to_value(cc)
}

/// Deserialize a `Box<&dyn CustomConst>` and attempt to downcast it to `CC`;
/// propagating failure.
fn deserialize_custom_const<CC: CustomConst>(
    value: serde_json::Value,
) -> Result<CC, serde_json::Error> {
    match deserialize_dyn_custom_const(value)?.downcast::<CC>() {
        Ok(cc) => Ok(*cc),
        Err(dyn_cc) => Err(<serde_json::Error as serde::de::Error>::custom(format!(
            "Failed to deserialize [{}]: {:?}",
            std::any::type_name::<CC>(),
            dyn_cc
        ))),
    }
}

/// Deserialize a `Box<&dyn CustomConst>`.
fn deserialize_dyn_custom_const(
    value: serde_json::Value,
) -> Result<Box<dyn CustomConst>, serde_json::Error> {
    serde_json::from_value(value)
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A constant value stored as a serialized blob that can report its own type.
pub struct CustomSerialized {
    typ: Type,
    value: serde_json::Value,
}

#[derive(Debug, Error)]
#[error("Error serializing value into CustomSerialized: err: {err}, value: {payload:?}")]
pub struct SerializeError {
    #[source]
    err: serde_json::Error,
    payload: Box<dyn CustomConst>,
}

#[derive(Debug, Error)]
#[error("Error deserializing value from CustomSerialized: err: {err}, value: {payload:?}")]
pub struct DeserializeError {
    #[source]
    err: serde_json::Error,
    payload: serde_json::Value,
}

impl CustomSerialized {
    /// Creates a new [`CustomSerialized`].
    pub fn new(typ: impl Into<Type>, value: serde_json::Value) -> Self {
        Self {
            typ: typ.into(),
            value,
        }
    }

    /// Returns the inner value.
    #[must_use]
    pub fn value(&self) -> &serde_json::Value {
        &self.value
    }

    /// If `cc` is a [Self], returns a clone of `cc` coerced to [Self].
    /// Otherwise, returns a [Self] with `cc` serialized in it's value.
    pub fn try_from_custom_const_ref(cc: &impl CustomConst) -> Result<Self, SerializeError> {
        Self::try_from_dyn_custom_const(cc)
    }

    /// If `cc` is a [Self], returns a clone of `cc` coerced to [Self].
    /// Otherwise, returns a [Self] with `cc` serialized in it's value.
    pub fn try_from_dyn_custom_const(cc: &dyn CustomConst) -> Result<Self, SerializeError> {
        Ok(match cc.as_any().downcast_ref::<Self>() {
            Some(cs) => cs.clone(),
            None => Self::new(
                cc.get_type(),
                serialize_custom_const(cc).map_err(|err| SerializeError {
                    err,
                    payload: cc.clone_box(),
                })?,
            ),
        })
    }

    /// If `cc` is a [Self], return `cc` coerced to [Self]. Otherwise,
    /// returns a [Self] with `cc` serialized in it's value.
    /// Never clones `cc` outside of error paths.
    pub fn try_from_custom_const(cc: impl CustomConst) -> Result<Self, SerializeError> {
        Self::try_from_custom_const_box(Box::new(cc))
    }

    /// If `cc` is a [Self], return `cc` coerced to [Self]. Otherwise,
    /// returns a [Self] with `cc` serialized in it's value.
    /// Never clones `cc` outside of error paths.
    pub fn try_from_custom_const_box(cc: Box<dyn CustomConst>) -> Result<Self, SerializeError> {
        match cc.downcast::<Self>() {
            Ok(x) => Ok(*x),
            Err(cc) => {
                let typ = cc.get_type();
                let value = serialize_custom_const(cc.as_ref())
                    .map_err(|err| SerializeError { err, payload: cc })?;
                Ok(Self::new(typ, value))
            }
        }
    }

    /// Attempts to deserialize the value in self into a `Box<dyn CustomConst>`.
    /// This can fail, in particular when the `impl CustomConst` for the trait
    /// is not linked into the running executable.
    /// If deserialization fails, returns self in a box.
    ///
    /// Note that if the inner value is a [Self] we do not recursively
    /// deserialize it.
    #[must_use]
    pub fn into_custom_const_box(self) -> Box<dyn CustomConst> {
        // ideally we would not have to clone, but serde_json does not allow us
        // to recover the value from the error
        deserialize_dyn_custom_const(self.value.clone()).unwrap_or_else(|_| Box::new(self))
    }

    /// Attempts to deserialize the value in self into a `CC`. Propagates failure.
    ///
    /// Note that if the inner value is a [Self] we do not recursively
    /// deserialize it. In particular if that inner value were a [Self] whose
    /// inner value were a `CC`, then we would still fail.
    pub fn try_into_custom_const<CC: CustomConst>(self) -> Result<CC, DeserializeError> {
        // ideally we would not have to clone, but serde_json does not allow us
        // to recover the value from the error
        deserialize_custom_const(self.value.clone()).map_err(|err| DeserializeError {
            err,
            payload: self.value,
        })
    }
}

impl TryHash for CustomSerialized {
    fn try_hash(&self, mut st: &mut dyn Hasher) -> bool {
        // Consistent with equality, same serialization <=> same hash.
        self.value.to_string().hash(&mut st);
        true
    }
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> ValueName {
        format!("json:{:?}", self.value).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }

    fn update_extensions(
        &mut self,
        extensions: &WeakExtensionRegistry,
    ) -> Result<(), ExtensionResolutionError> {
        resolve_type_extensions(&mut self.typ, extensions)
    }
    fn get_type(&self) -> Type {
        self.typ.clone()
    }
}

/// This module is used by the serde annotations on `super::OpaqueValue`
pub(super) mod serde_extension_value {
    use serde::{Deserializer, Serializer};

    use super::{CustomConst, CustomSerialized};

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Box<dyn CustomConst>, D::Error> {
        use serde::Deserialize;
        // We deserialize a CustomSerialized, i.e. not a dyn CustomConst.
        let cs = CustomSerialized::deserialize(deserializer)?;
        // We return the inner serialized CustomConst if we can, otherwise the
        // CustomSerialized itself.
        Ok(cs.into_custom_const_box())
    }

    pub fn serialize<S: Serializer>(
        konst: impl AsRef<dyn CustomConst>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::Serialize;
        // we create a CustomSerialized, then serialize it. Note we do not
        // serialize it as a dyn CustomConst.
        let cs = CustomSerialized::try_from_dyn_custom_const(konst.as_ref())
            .map_err(<S::Error as serde::ser::Error>::custom)?;
        cs.serialize(serializer)
    }
}

/// Given a singleton list of constant operations, return the value.
#[must_use]
pub fn get_single_input_value<T: CustomConst>(consts: &[(IncomingPort, Value)]) -> Option<&T> {
    let [(_, c)] = consts else {
        return None;
    };
    c.get_custom_value()
}

/// Given a list of two constant operations, return the values.
#[must_use]
pub fn get_pair_of_input_values<T: CustomConst>(
    consts: &[(IncomingPort, Value)],
) -> Option<(&T, &T)> {
    let [(_, c0), (_, c1)] = consts else {
        return None;
    };
    Some((c0.get_custom_value()?, c1.get_custom_value()?))
}

// these tests depend on the `typetag` crate.
#[cfg(all(test, not(miri)))]
mod test {

    use rstest::rstest;

    use crate::{
        extension::prelude::{ConstUsize, usize_t},
        ops::{Value, constant::custom::serialize_custom_const},
        std_extensions::collections::list::ListValue,
    };

    use super::{super::OpaqueValue, CustomConst, CustomConstBoxClone, CustomSerialized};

    struct SerializeCustomConstExample<CC: CustomConst + serde::Serialize + 'static> {
        cc: CC,
        tag: &'static str,
        json: serde_json::Value,
    }

    impl<CC: CustomConst + serde::Serialize + 'static> SerializeCustomConstExample<CC> {
        fn new(cc: CC, tag: &'static str) -> Self {
            let json = serde_json::to_value(&cc).unwrap();
            Self { cc, tag, json }
        }
    }

    fn scce_usize() -> SerializeCustomConstExample<ConstUsize> {
        SerializeCustomConstExample::new(ConstUsize::new(12), "ConstUsize")
    }

    fn scce_list() -> SerializeCustomConstExample<ListValue> {
        let cc = ListValue::new(
            usize_t(),
            [ConstUsize::new(1), ConstUsize::new(2)]
                .into_iter()
                .map(Value::extension),
        );
        SerializeCustomConstExample::new(cc, "ListValue")
    }

    #[rstest]
    #[cfg_attr(miri, ignore = "miri is incompatible with the typetag crate")]
    #[case(scce_usize())]
    #[case(scce_list())]
    fn test_custom_serialized_try_from<
        CC: CustomConst + serde::Serialize + Clone + PartialEq + 'static + Sized,
    >(
        #[case] example: SerializeCustomConstExample<CC>,
    ) {
        assert_eq!(example.json, serde_json::to_value(&example.cc).unwrap()); // sanity check
        let expected_json: serde_json::Value = [
            ("c".into(), example.tag.into()),
            ("v".into(), example.json.clone()),
        ]
        .into_iter()
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();

        // check serialize_custom_const
        assert_eq!(expected_json, serialize_custom_const(&example.cc).unwrap());

        let expected_custom_serialized =
            CustomSerialized::new(example.cc.get_type(), expected_json);

        // check all the try_from/try_into/into variations
        assert_eq!(
            &expected_custom_serialized,
            &CustomSerialized::try_from_custom_const(example.cc.clone()).unwrap()
        );
        assert_eq!(
            &expected_custom_serialized,
            &CustomSerialized::try_from_custom_const_ref(&example.cc).unwrap()
        );
        assert_eq!(
            &expected_custom_serialized,
            &CustomSerialized::try_from_custom_const_box(example.cc.clone_box()).unwrap()
        );
        assert_eq!(
            &expected_custom_serialized,
            &CustomSerialized::try_from_dyn_custom_const(example.cc.clone_box().as_ref()).unwrap()
        );
        assert_eq!(
            &example.cc.clone_box(),
            &expected_custom_serialized.clone().into_custom_const_box()
        );
        assert_eq!(
            &example.cc,
            &expected_custom_serialized
                .clone()
                .try_into_custom_const()
                .unwrap()
        );

        // check OpaqueValue serializes/deserializes as a CustomSerialized
        let ev: OpaqueValue = example.cc.clone().into();
        let ev_val = serde_json::to_value(&ev).unwrap();
        assert_eq!(
            &ev_val,
            &serde_json::to_value(&expected_custom_serialized).unwrap()
        );
        assert_eq!(ev, serde_json::from_value(ev_val).unwrap());
    }

    fn example_custom_serialized() -> (ConstUsize, CustomSerialized) {
        let inner = scce_usize().cc;
        (
            inner.clone(),
            CustomSerialized::try_from_custom_const(inner).unwrap(),
        )
    }

    fn example_nested_custom_serialized() -> (CustomSerialized, CustomSerialized) {
        let inner = example_custom_serialized().1;
        (
            inner.clone(),
            CustomSerialized::new(inner.get_type(), serialize_custom_const(&inner).unwrap()),
        )
    }

    #[rstest]
    #[cfg_attr(miri, ignore = "miri is incompatible with the typetag crate")]
    #[case(example_custom_serialized())]
    #[case(example_nested_custom_serialized())]
    fn test_try_from_custom_serialized_recursive<CC: CustomConst + PartialEq>(
        #[case] example: (CC, CustomSerialized),
    ) {
        let (inner, cs) = example;
        // check all the try_from/try_into/into variations

        assert_eq!(
            &cs,
            &CustomSerialized::try_from_custom_const(cs.clone()).unwrap()
        );
        assert_eq!(
            &cs,
            &CustomSerialized::try_from_custom_const_ref(&cs).unwrap()
        );
        assert_eq!(
            &cs,
            &CustomSerialized::try_from_custom_const_box(cs.clone_box()).unwrap()
        );
        assert_eq!(
            &cs,
            &CustomSerialized::try_from_dyn_custom_const(cs.clone_box().as_ref()).unwrap()
        );
        assert_eq!(&inner.clone_box(), &cs.clone().into_custom_const_box());
        assert_eq!(&inner, &cs.clone().try_into_custom_const().unwrap());

        let ev: OpaqueValue = cs.clone().into();
        // A serialization round-trip results in an OpaqueValue with the value of inner
        assert_eq!(
            OpaqueValue::new(inner),
            serde_json::from_value(serde_json::to_value(&ev).unwrap()).unwrap()
        );
    }
}

#[cfg(test)]
mod proptest {
    use ::proptest::prelude::*;

    use crate::{
        ops::constant::CustomSerialized,
        proptest::{any_serde_json_value, any_string},
        types::Type,
    };

    impl Arbitrary for CustomSerialized {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            let typ = any::<Type>();
            // here we manually construct a serialized `dyn CustomConst`.
            // The "c" and "v" come from the `typetag::serde` annotation on
            // `trait CustomConst`.
            // TODO This is not ideal, if we were to accidentally
            // generate a valid tag(e.g. "ConstInt") then things will
            // go wrong: the serde::Deserialize impl for that type will
            // interpret "v" and fail.
            let value = (any_serde_json_value(), any_string()).prop_map(|(content, tag)| {
                [("c".into(), tag.into()), ("v".into(), content)]
                    .into_iter()
                    .collect::<serde_json::Map<String, _>>()
                    .into()
            });
            (typ, value)
                .prop_map(|(typ, value)| CustomSerialized { typ, value })
                .boxed()
        }
    }
}
