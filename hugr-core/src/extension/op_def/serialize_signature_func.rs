use serde::{Deserialize, Serialize};

use super::{CustomValidator, PolyFuncTypeRV, SignatureFunc};
#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
struct SerSignatureFunc {
    /// If the type scheme is available explicitly, store it.
    signature: Option<PolyFuncTypeRV>,
    /// Whether an associated binary function is expected.
    /// If `signature` is `None`, a true value here indicates a custom compute function.
    /// If `signature` is not `None`, a true value here indicates a custom validation function.
    binary: bool,
}

pub(super) fn serialize<S>(value: &super::SignatureFunc, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        SignatureFunc::PolyFuncType(poly) => SerSignatureFunc {
            signature: Some(poly.clone()),
            binary: false,
        },
        SignatureFunc::CustomValidator(CustomValidator { poly_func, .. })
        | SignatureFunc::MissingValidateFunc(poly_func) => SerSignatureFunc {
            signature: Some(poly_func.clone()),
            binary: true,
        },
        SignatureFunc::CustomFunc(_) | SignatureFunc::MissingComputeFunc => SerSignatureFunc {
            signature: None,
            binary: true,
        },
    }
    .serialize(serializer)
}

pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<super::SignatureFunc, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let SerSignatureFunc { signature, binary } = SerSignatureFunc::deserialize(deserializer)?;

    match (signature, binary) {
        (Some(sig), false) => Ok(sig.into()),
        (Some(sig), true) => Ok(SignatureFunc::MissingValidateFunc(sig)),
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
            CustomSignatureFunc, CustomValidator, OpDef, SignatureError, ValidateTypeArgs,
            prelude::usize_t,
        },
        types::{FuncValueType, Signature, TypeArg},
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
        ) -> Result<crate::types::PolyFuncTypeRV, crate::extension::SignatureError> {
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
        ) -> Result<(), SignatureError> {
            Ok(())
        }
    }

    #[test]
    fn test_serial_sig_func() {
        // test round-trip
        let sig: FuncValueType = Signature::new_endo(usize_t().clone()).into();
        let simple: SignatureFunc = sig.clone().into();
        let ser: SerSignatureFunc = simple.into();
        let expected_ser = SerSignatureFunc {
            signature: Some(sig.clone().into()),
            binary: false,
        };

        assert_eq!(ser, expected_ser);
        let deser = SignatureFunc::try_from(ser).unwrap();
        assert_matches!( deser,
        SignatureFunc::PolyFuncType(poly_func) => {
            assert_eq!(poly_func, sig.clone().into());
        });

        let with_custom: SignatureFunc = CustomValidator::new(sig.clone(), NoValidate).into();
        let ser: SerSignatureFunc = with_custom.into();
        let expected_ser = SerSignatureFunc {
            signature: Some(sig.clone().into()),
            binary: true,
        };
        assert_eq!(ser, expected_ser);
        let mut deser = SignatureFunc::try_from(ser.clone()).unwrap();
        assert_matches!(&deser,
            SignatureFunc::MissingValidateFunc(poly_func) => {
                assert_eq!(poly_func, &PolyFuncTypeRV::from(sig.clone()));
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
            signature: None,
            binary: true,
        };
        assert_eq!(ser, expected_ser);

        let deser = SignatureFunc::try_from(ser).unwrap();
        assert_matches!(&deser, &SignatureFunc::MissingComputeFunc);

        assert_eq!(SerSignatureFunc::from(deser), expected_ser);

        let bad_ser = SerSignatureFunc {
            signature: None,
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
