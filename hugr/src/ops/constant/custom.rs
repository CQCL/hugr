//! Representation of custom constant values.
//!
//! These can be used as [`Const`] operations in HUGRs.
//!
//! [`Const`]: crate::ops::Const

use std::any::Any;

use downcast_rs::{impl_downcast, Downcast};
use thiserror::Error;

use crate::extension::ExtensionSet;
use crate::macros::impl_box_clone;

use crate::types::{CustomCheckFailure, Type};

use super::ValueName;

/// Constant value for opaque [`CustomType`]s.
///
/// When implementing this trait, include the `#[typetag::serde]` attribute to
/// enable serialization.
///
/// [`CustomType`]: crate::types::CustomType
#[typetag::serde(tag = "c", content = "v")]
pub trait CustomConst:
    Send + Sync + std::fmt::Debug + CustomConstBoxClone + Any + Downcast
{
    /// An identifier for the constant.
    fn name(&self) -> ValueName;

    /// The extension(s) defining the custom constant
    /// (a set to allow, say, a [List] of [USize])
    ///
    /// [List]: crate::std_extensions::collections::LIST_TYPENAME
    /// [USize]: crate::extension::prelude::USIZE_T
    fn extension_reqs(&self) -> ExtensionSet;

    /// Check the value.
    fn validate(&self) -> Result<(), CustomCheckFailure> {
        Ok(())
    }

    /// Compare two constants for equality, using downcasting and comparing the definitions.
    ///
    /// If the type implements `PartialEq`, use [`downcast_equal_consts`] to compare the values.
    fn equal_consts(&self, _other: &dyn CustomConst) -> bool {
        // false unless overloaded
        false
    }

    /// Report the type.
    fn get_type(&self) -> Type;
}

impl PartialEq for dyn CustomConst {
    fn eq(&self, other: &Self) -> bool {
        (*self).equal_consts(other)
    }
}

/// Const equality for types that have PartialEq
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

/// Serialize any CustomConst using the `impl Serialize for &dyn CustomConst`.
/// In particular this works on `&dyn CustomConst` and `Box<dyn CustomConst>::Target`.
/// See tests below
fn serialize_custom_const(cc: &dyn CustomConst) -> Result<serde_yaml::Value, serde_yaml::Error> {
    serde_yaml::to_value(cc)
}

fn deserialize_custom_const<CC: CustomConst>(
    value: serde_yaml::Value,
) -> Result<CC, serde_yaml::Error> {
    match deserialize_dyn_custom_const(value)?.downcast::<CC>() {
        Ok(cc) => Ok(*cc),
        Err(dyn_cc) => Err(<serde_yaml::Error as serde::de::Error>::custom(format!(
            "Failed to deserialize [{}]: {:?}",
            std::any::type_name::<CC>(),
            dyn_cc
        ))),
    }
}

fn deserialize_dyn_custom_const(
    value: serde_yaml::Value,
) -> Result<Box<dyn CustomConst>, serde_yaml::Error> {
    serde_yaml::from_value(value)
}

impl_downcast!(CustomConst);
impl_box_clone!(CustomConst, CustomConstBoxClone);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A constant value stored as a serialized blob that can report its own type.
pub struct CustomSerialized {
    typ: Type,
    value: serde_yaml::Value,
    extensions: ExtensionSet,
}

#[derive(Debug, Error)]
pub enum CustomSerializedError {
    #[error("Error serializing value into CustomSerialised: err: {err}, value: {payload:?}")]
    SerializePayloadError {
        #[source]
        err: serde_yaml::Error,
        payload: Box<dyn CustomConst>,
    },
    #[error("Error serializing value into CustomSerialised: err: {err}, value: {payload:?}")]
    DeserializePayloadError {
        #[source]
        err: serde_yaml::Error,
        payload: serde_yaml::Value,
    },
}

impl CustomSerializedError {
    fn new_ser(err: serde_yaml::Error, payload: Box<dyn CustomConst>) -> Self {
        Self::SerializePayloadError { err, payload }
    }

    fn new_de(err: serde_yaml::Error, payload: serde_yaml::Value) -> Self {
        Self::DeserializePayloadError { err, payload }
    }
}

impl CustomSerialized {
    /// Creates a new [`CustomSerialized`].
    pub fn new(
        typ: impl Into<Type>,
        value: serde_yaml::Value,
        exts: impl Into<ExtensionSet>,
    ) -> Self {
        Self {
            typ: typ.into(),
            value,
            extensions: exts.into(),
        }
    }

    /// Returns the inner value.
    pub fn value(&self) -> &serde_yaml::Value {
        &self.value
    }

    /// TODO
    pub fn try_from_custom_const_ref(
        cc: &(impl CustomConst + ?Sized),
    ) -> Result<Self, CustomSerializedError> {
        Self::try_from_custom_const_box(cc.clone_box())
    }

    /// TODO
    pub fn try_from_custom_const(cc: impl CustomConst) -> Result<Self, CustomSerializedError> {
        Self::try_from_custom_const_box(Box::new(cc))
    }

    /// TODO
    pub fn try_from_custom_const_box(
        cc: Box<dyn CustomConst>,
    ) -> Result<Self, CustomSerializedError> {
        match cc.downcast::<Self>() {
            Ok(x) => Ok(*x),
            Err(cc) => {
                let (typ, extension_reqs) = (cc.get_type(), cc.extension_reqs());
                let value = serialize_custom_const(cc.as_ref())
                    .map_err(|err| CustomSerializedError::new_ser(err, cc))?;
                Ok(Self::new(typ, value, extension_reqs))
            }
        }
    }

    /// TODO
    pub fn into_custom_const_box(self) -> Box<dyn CustomConst> {
        let (typ, extensions) = (self.get_type().clone(), self.extension_reqs());
        // ideally we would not have to clone, but serde_json does not allow us
        // to recover the value from the error
        let cc_box =
            deserialize_dyn_custom_const(self.value.clone()).unwrap_or_else(|_| Box::new(self));
        assert_eq!(cc_box.get_type(), typ);
        assert_eq!(cc_box.extension_reqs(), extensions);
        cc_box
    }

    /// TODO
    pub fn try_into_custom_const<CC: CustomConst>(self) -> Result<CC, CustomSerializedError> {
        let CustomSerialized {
            typ,
            value,
            extensions,
        } = self;
        // ideally we would not have to clone, but serde_json does not allow us
        // to recover the value from the error
        let cc: CC = deserialize_custom_const(value.clone())
            .map_err(|err| CustomSerializedError::new_de(err, value))?;
        assert_eq!(cc.get_type(), typ);
        assert_eq!(cc.extension_reqs(), extensions);
        Ok(cc)
    }
}

#[typetag::serde]
impl CustomConst for CustomSerialized {
    fn name(&self) -> ValueName {
        format!("yaml:{:?}", self.value).into()
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        Some(self) == other.downcast_ref()
    }

    fn extension_reqs(&self) -> ExtensionSet {
        self.extensions.clone()
    }
    fn get_type(&self) -> Type {
        self.typ.clone()
    }
}

impl TryFrom<&dyn CustomConst> for CustomSerialized {
    type Error = CustomSerializedError;
    fn try_from(value: &dyn CustomConst) -> Result<Self, Self::Error> {
        Self::try_from_custom_const_ref(value)
    }
}

impl TryFrom<Box<dyn CustomConst>> for CustomSerialized {
    type Error = CustomSerializedError;
    fn try_from(value: Box<dyn CustomConst>) -> Result<Self, Self::Error> {
        Self::try_from_custom_const_box(value)
    }
}

impl From<CustomSerialized> for Box<dyn CustomConst> {
    fn from(cs: CustomSerialized) -> Self {
        cs.into_custom_const_box()
    }
}

#[cfg(test)]
mod test {

    use rstest::rstest;

    use crate::{
        extension::{
            prelude::{ConstUsize, USIZE_T},
            ExtensionSet,
        },
        ops::{
            constant::custom::{deserialize_dyn_custom_const, serialize_custom_const},
            Value,
        },
        std_extensions::{arithmetic::int_types::ConstInt, collections::ListValue},
    };

    use super::{CustomConst, CustomConstBoxClone, CustomSerialized};

    struct SerializeCustomConstExample<CC: CustomConst + serde::Serialize + 'static> {
        cc: CC,
        tag: &'static str,
        yaml: serde_yaml::Value,
    }

    impl<CC: CustomConst + serde::Serialize + 'static> SerializeCustomConstExample<CC> {
        fn new(cc: CC, tag: &'static str) -> Self {
            let yaml = serde_yaml::to_value(&cc).unwrap();
            Self { cc, tag, yaml }
        }
    }

    fn ser_cc_ex1() -> SerializeCustomConstExample<ConstUsize> {
        SerializeCustomConstExample::new(ConstUsize::new(12), "ConstUsize")
    }

    fn ser_cc_ex2() -> SerializeCustomConstExample<ListValue> {
        SerializeCustomConstExample::new(
            ListValue::new(
                USIZE_T,
                [ConstUsize::new(1), ConstUsize::new(2)]
                    .into_iter()
                    .map(Value::extension),
            ),
            "ListValue",
        )
    }

    fn ser_cc_ex3() -> SerializeCustomConstExample<CustomSerialized> {
        SerializeCustomConstExample::new(
            CustomSerialized::new(USIZE_T, serde_yaml::Value::Null, ExtensionSet::default()),
            "CustomSerialized",
        )
    }

    #[rstest]
    #[case(ser_cc_ex1())]
    #[case(ser_cc_ex2())]
    #[case(ser_cc_ex3())]
    fn test_serialize_custom_const<CC: CustomConst + serde::Serialize + 'static + Sized>(
        #[case] example: SerializeCustomConstExample<CC>,
    ) {
        let expected_yaml: serde_yaml::Value =
            [("c".into(), example.tag.into()), ("v".into(), example.yaml)]
                .into_iter()
                .collect::<serde_yaml::Mapping>()
                .into();

        let yaml_by_ref = serialize_custom_const(&example.cc as &CC).unwrap();
        assert_eq!(expected_yaml, yaml_by_ref);

        let yaml_by_dyn_ref = serialize_custom_const(&example.cc as &dyn CustomConst).unwrap();
        assert_eq!(expected_yaml, yaml_by_dyn_ref);
    }

    #[test]
    fn custom_serialized_from_into_custom_const() {
        let const_int = ConstInt::new_s(4, 1).unwrap();

        let cs: CustomSerialized = CustomSerialized::try_from_custom_const_ref(&const_int).unwrap();

        assert_eq!(const_int.get_type(), cs.get_type());
        assert_eq!(const_int.extension_reqs(), cs.extension_reqs());
        assert_eq!(&serialize_custom_const(&const_int).unwrap(), cs.value());

        let deser_const_int: ConstInt = cs.try_into_custom_const().unwrap();

        assert_eq!(const_int, deser_const_int);
    }

    #[test]
    fn custom_serialized_from_into_custom_serialised() {
        let const_int = ConstInt::new_s(4, 1).unwrap();
        let cs0: CustomSerialized =
            CustomSerialized::try_from_custom_const_ref(&const_int).unwrap();

        let cs1 = CustomSerialized::try_from_custom_const_ref(&cs0).unwrap();
        assert_eq!(&cs0, &cs1);

        let deser_const_int: ConstInt = cs0.try_into_custom_const().unwrap();
        assert_eq!(&const_int, &deser_const_int);
    }

    #[test]
    fn custom_serialized_try_from_dyn_custom_const() {
        let const_int = ConstInt::new_s(4, 1).unwrap();
        let cs: CustomSerialized = const_int.clone_box().try_into().unwrap();
        assert_eq!(const_int.get_type(), cs.get_type());
        assert_eq!(const_int.extension_reqs(), cs.extension_reqs());
        assert_eq!(&serialize_custom_const(&const_int).unwrap(), cs.value());

        let deser_const_int: ConstInt = {
            let dyn_box: Box<dyn CustomConst> = cs.into();
            *dyn_box.downcast().unwrap()
        };
        assert_eq!(const_int, deser_const_int)
    }

    #[test]
    fn nested_custom_serialized() {
        let const_int = ConstInt::new_s(4, 1).unwrap();
        let cs_inner: CustomSerialized =
            CustomSerialized::try_from_custom_const_ref(&const_int).unwrap();

        let cs_inner_ser = serialize_custom_const(&cs_inner).unwrap();

        // cs_outer is a CustomSerialized of a CustomSerialized of a ConstInt
        let cs_outer = CustomSerialized::new(
            cs_inner.get_type(),
            cs_inner_ser.clone(),
            cs_inner.extension_reqs(),
        );
        // TODO should this be &const_int == cs_outer.value() ???
        assert_eq!(&cs_inner_ser, cs_outer.value());

        assert_eq!(const_int.get_type(), cs_outer.get_type());
        assert_eq!(const_int.extension_reqs(), cs_outer.extension_reqs());

        // TODO should this be &const_int = cs_outer.value() ???
        let inner_deser: Box<dyn CustomConst> =
            deserialize_dyn_custom_const(cs_outer.value().clone()).unwrap();
        assert_eq!(
            &cs_inner,
            inner_deser.downcast_ref::<CustomSerialized>().unwrap()
        );
    }
}
