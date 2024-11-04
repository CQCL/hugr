use serde::{Deserialize, Serialize};

use crate::types::{PolyFuncTypeBase, RowVariable, TypeRow};

use super::{CustomValidator, OpDefSignature, SignatureFunc};
#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
struct SerSignatureFunc {
    /// If the type scheme is available explicitly, store it.
    signature: Option<PolyFuncTypeBase<RowVariable>>,
    /// Whether an associated binary function is expected.
    /// If `signature` is `None`, a true value here indicates a custom compute function.
    /// If `signature` is not `None`, a true value here indicates a custom validation function.
    binary: bool,
    #[serde(default, skip_serializing_if = "TypeRow::is_empty")]
    static_inputs: TypeRow,
}

pub(super) fn serialize<S>(value: &super::SignatureFunc, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        SignatureFunc::PolyFuncType(op_sig) => SerSignatureFunc {
            signature: Some(op_sig.poly_func_type().clone()),
            static_inputs: op_sig.static_inputs().clone(),
            binary: false,
        },
        SignatureFunc::CustomValidator(CustomValidator { poly_func, .. })
        | SignatureFunc::MissingValidateFunc(poly_func) => SerSignatureFunc {
            signature: Some(poly_func.poly_func_type().clone()),
            static_inputs: poly_func.static_inputs().clone(),
            binary: true,
        },
        SignatureFunc::CustomFunc(_) | SignatureFunc::MissingComputeFunc => SerSignatureFunc {
            signature: None,
            static_inputs: TypeRow::new(),
            binary: true,
        },
    }
    .serialize(serializer)
}

pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<super::SignatureFunc, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let SerSignatureFunc {
        signature,
        binary,
        static_inputs,
    } = SerSignatureFunc::deserialize(deserializer)?;

    match (signature, binary) {
        (Some(sig), false) => Ok(OpDefSignature::from(sig)
            .with_static_inputs(static_inputs)
            .into()),
        (Some(sig), true) => Ok(SignatureFunc::MissingValidateFunc(
            OpDefSignature::from(sig).with_static_inputs(static_inputs),
        )),
        (None, true) => Ok(SignatureFunc::MissingComputeFunc),
        (None, false) => Err(serde::de::Error::custom(
            "No signature provided and custom computation not expected.",
        )),
    }
}

#[cfg(test)]
mod test {
    use cool_asserts::assert_matches;
    use serde::de::Error;

    use super::*;
    use crate::{
        extension::{
            prelude::{BOOL_T, USIZE_T},
            CustomSignatureFunc, CustomValidator, ExtensionRegistry, OpDef, SignatureError,
            ValidateTypeArgs,
        },
        type_row,
        types::{Signature, TypeArg},
    };

    #[derive(serde::Deserialize, serde::Serialize, Debug)]
    /// Wrapper we can derive serde for, to allow round-trip serialization
    struct Wrapper {
        #[serde(
            serialize_with = "serialize",
            deserialize_with = "deserialize",
            flatten
        )]
        inner: SignatureFunc,
    }
    // Define test-only conversions via serialization roundtrip
    impl TryFrom<SerSignatureFunc> for SignatureFunc {
        type Error = serde_json::Error;
        fn try_from(value: SerSignatureFunc) -> Result<Self, Self::Error> {
            let ser = serde_json::to_value(value).unwrap();
            let w: Wrapper = serde_json::from_value(ser)?;
            Ok(w.inner)
        }
    }

    impl From<SignatureFunc> for SerSignatureFunc {
        fn from(value: SignatureFunc) -> Self {
            let ser = serde_json::to_value(Wrapper { inner: value }).unwrap();
            serde_json::from_value(ser).unwrap()
        }
    }
    struct CustomSig;

    impl CustomSignatureFunc for CustomSig {
        fn compute_signature<'o, 'a: 'o>(
            &'a self,
            _arg_values: &[TypeArg],
            _def: &'o crate::extension::op_def::OpDef,
            _extension_registry: &crate::extension::ExtensionRegistry,
        ) -> Result<crate::types::OpDefSignature, crate::extension::SignatureError> {
            Ok(Default::default())
        }

        fn static_params(&self) -> &[crate::types::type_param::TypeParam] {
            &[]
        }
    }

    struct NoValidate;
    impl ValidateTypeArgs for NoValidate {
        fn validate<'o, 'a: 'o>(
            &self,
            _arg_values: &[TypeArg],
            _def: &'o OpDef,
            _extension_registry: &ExtensionRegistry,
        ) -> Result<(), SignatureError> {
            Ok(())
        }
    }

    #[test]
    fn test_serial_sig_func() {
        // test round-trip
        let sig = OpDefSignature::new([], Signature::new_endo(USIZE_T.clone()))
            .with_static_inputs(BOOL_T);
        let simple: SignatureFunc = sig.clone().into();
        let ser: SerSignatureFunc = simple.into();
        let expected_ser = SerSignatureFunc {
            signature: Some(sig.poly_func_type().clone()),
            binary: false,
            static_inputs: type_row![BOOL_T],
        };

        assert_eq!(ser, expected_ser);
        let deser = SignatureFunc::try_from(ser).unwrap();
        assert_matches!( deser,
        SignatureFunc::PolyFuncType(op_def_sig) => {
            assert_eq!(op_def_sig, sig.clone());
        });

        let with_custom: SignatureFunc = CustomValidator::new(sig.clone(), NoValidate).into();
        let ser: SerSignatureFunc = with_custom.into();
        let expected_ser = SerSignatureFunc {
            signature: Some(sig.poly_func_type().clone()),
            static_inputs: type_row![BOOL_T],
            binary: true,
        };
        assert_eq!(ser, expected_ser);
        let mut deser = SignatureFunc::try_from(ser.clone()).unwrap();
        assert_matches!(&deser,
            SignatureFunc::MissingValidateFunc(poly_func) => {
                assert_eq!(poly_func, &sig.clone());
            }
        );

        // re-serializing should give the same result
        assert_eq!(
            SerSignatureFunc::from(SignatureFunc::try_from(ser).unwrap()),
            expected_ser
        );

        deser.ignore_missing_validation();
        assert_matches!(&deser, &SignatureFunc::PolyFuncType(_));

        let custom: SignatureFunc = CustomSig.into();
        let ser: SerSignatureFunc = custom.into();
        let expected_ser = SerSignatureFunc {
            static_inputs: type_row![],
            signature: None,
            binary: true,
        };
        assert_eq!(ser, expected_ser);

        let deser = SignatureFunc::try_from(ser).unwrap();
        assert_matches!(&deser, &SignatureFunc::MissingComputeFunc);

        assert_eq!(SerSignatureFunc::from(deser), expected_ser);

        let bad_ser = SerSignatureFunc {
            signature: None,
            static_inputs: type_row![],
            binary: false,
        };

        let err = SignatureFunc::try_from(bad_ser).unwrap_err();

        assert_eq!(
            err.to_string(),
            serde_json::Error::custom("No signature provided and custom computation not expected.")
                .to_string()
        );
    }
}
