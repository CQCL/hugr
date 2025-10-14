//! Derivation to serialize and deserialize Hugrs and Packages as envelopes in a
//! serde compatible way.
//!
//! This module provides a default wrapper, [`AsStringEnvelope`], that decodes
//! hugrs and packages using the [`STD_REG`] extension registry.
//!
//! When a different extension registry is needed, use the
//! [`impl_serde_as_string_envelope!`] macro to create a custom wrapper.
//!
//! These are meant to be used with `serde_with`'s `#[serde_as]` decorator, see
//! <https://docs.rs/serde_with/latest/serde_with>.

use crate::std_extensions::STD_REG;

/// De/Serialize a package or hugr by encoding it into a textual Envelope and
/// storing it as a string.
///
/// This is similar to [`AsBinaryEnvelope`], but uses a textual envelope instead
/// of a binary one.
///
/// Note that only PRELUDE extensions are used to decode the package's content.
/// When serializing a HUGR, any additional extensions required to load it are
/// embedded in the envelope. Packages should manually add any required
/// extensions before serializing.
///
/// # Examples
///
/// ```rust
/// # use serde::{Deserialize, Serialize};
/// # use serde_json::json;
/// # use serde_with::{serde_as};
/// # use hugr_core::Hugr;
/// # use hugr_core::package::Package;
/// # use hugr_core::envelope::serde_with::AsStringEnvelope;
/// #
/// #[serde_as]
/// #[derive(Deserialize, Serialize)]
/// struct A {
///     #[serde_as(as = "AsStringEnvelope")]
///     package: Package,
///     #[serde_as(as = "Vec<AsStringEnvelope>")]
///     hugrs: Vec<Hugr>,
/// }
/// ```
///
/// # Backwards compatibility
///
/// When reading an encoded HUGR, the `AsStringEnvelope` deserializer will first
/// try to decode the value as an string-encoded envelope. If that fails, it
/// will fallback to decoding the legacy HUGR serde definition. This temporary
/// compatibility is required to support `hugr <= 0.19` and will be removed in
/// a future version.
pub struct AsStringEnvelope;

/// De/Serialize a package or hugr by encoding it into a binary envelope and
/// storing it as a base64-encoded string.
///
/// This is similar to [`AsStringEnvelope`], but uses a binary envelope instead
/// of a string.
/// When deserializing, if the string starts with the envelope magic 'HUGRiHJv'
/// it will be loaded as a string envelope without base64 decoding.
///
/// Note that only PRELUDE extensions are used to decode the package's content.
/// When serializing a HUGR, any additional extensions required to load it are
/// embedded in the envelope. Packages should manually add any required
/// extensions before serializing.
///
/// # Examples
///
/// ```rust
/// # use serde::{Deserialize, Serialize};
/// # use serde_json::json;
/// # use serde_with::{serde_as};
/// # use hugr_core::Hugr;
/// # use hugr_core::package::Package;
/// # use hugr_core::envelope::serde_with::AsBinaryEnvelope;
/// #
/// #[serde_as]
/// #[derive(Deserialize, Serialize)]
/// struct A {
///     #[serde_as(as = "AsBinaryEnvelope")]
///     package: Package,
///     #[serde_as(as = "Vec<AsBinaryEnvelope>")]
///     hugrs: Vec<Hugr>,
/// }
/// ```
///
/// # Backwards compatibility
///
/// When reading an encoded HUGR, the `AsBinaryEnvelope` deserializer will first
/// try to decode the value as an binary-encoded envelope. If that fails, it
/// will fallback to decoding a string envelope instead, and then finally to
/// decoding the legacy HUGR serde definition. This temporary compatibility
/// layer is required to support `hugr <= 0.19` and will be removed in a future
/// version.
pub struct AsBinaryEnvelope;

/// Implements [`serde_with::DeserializeAs`] and [`serde_with::SerializeAs`] for
/// the helper to deserialize `Hugr` and `Package` types, using the given
/// extension registry.
///
/// This macro is used to implement the default [`AsStringEnvelope`] wrapper.
///
/// # Parameters
///
/// - `$adaptor`: The name of the adaptor type to implement.
/// - `$extension_reg`: A reference to the extension registry to use for deserialization.
///
/// # Examples
///
/// ```rust
/// # use serde::{Deserialize, Serialize};
/// # use serde_json::json;
/// # use serde_with::{serde_as};
/// # use hugr_core::Hugr;
/// # use hugr_core::package::Package;
/// # use hugr_core::envelope::serde_with::AsStringEnvelope;
/// # use hugr_core::envelope::serde_with::impl_serde_as_string_envelope;
/// # use hugr_core::extension::ExtensionRegistry;
/// #
/// struct CustomAsEnvelope;
///
/// impl_serde_as_string_envelope!(CustomAsEnvelope, &hugr_core::extension::EMPTY_REG);
///
/// #[serde_as]
/// #[derive(Deserialize, Serialize)]
/// struct A {
///     #[serde_as(as = "CustomAsEnvelope")]
///     package: Package,
/// }
/// ```
///
#[macro_export]
macro_rules! impl_serde_as_string_envelope {
    ($adaptor:ident, $extension_reg:expr) => {
        impl<'de> serde_with::DeserializeAs<'de, $crate::package::Package> for $adaptor {
            fn deserialize_as<D>(deserializer: D) -> Result<$crate::package::Package, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Helper;
                impl serde::de::Visitor<'_> for Helper {
                    type Value = $crate::package::Package;

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
                        formatter.write_str("a string-encoded envelope")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                        $crate::package::Package::load_str(value, Some(extensions))
                            .map_err(serde::de::Error::custom)
                    }
                }

                deserializer.deserialize_str(Helper)
            }
        }

        impl<'de> serde_with::DeserializeAs<'de, $crate::Hugr> for $adaptor {
            fn deserialize_as<D>(deserializer: D) -> Result<$crate::Hugr, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Helper;
                impl<'vis> serde::de::Visitor<'vis> for Helper {
                    type Value = $crate::Hugr;

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
                        formatter.write_str("a string-encoded envelope")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                        $crate::Hugr::load_str(value, Some(extensions))
                            .map_err(serde::de::Error::custom)
                    }

                    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::MapAccess<'vis>,
                    {
                        // Backwards compatibility: If the encoded value is not a
                        // string, we may have a legacy HUGR serde structure instead. In that
                        // case, we can add an envelope header and try again.
                        //
                        // TODO: Remove this fallback in 0.21.0
                        let deserializer = serde::de::value::MapAccessDeserializer::new(map);
                        #[expect(deprecated)]
                        let mut hugr =
                            $crate::hugr::serialize::serde_deserialize_hugr(deserializer)
                                .map_err(serde::de::Error::custom)?;

                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                        hugr.resolve_extension_defs(extensions)
                            .map_err(serde::de::Error::custom)?;
                        Ok(hugr)
                    }
                }

                // TODO: Go back to `deserialize_str` once the fallback is removed.
                deserializer.deserialize_any(Helper)
            }
        }

        impl serde_with::SerializeAs<$crate::package::Package> for $adaptor {
            fn serialize_as<S>(
                source: &$crate::package::Package,
                serializer: S,
            ) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let str = source
                    .store_str($crate::envelope::EnvelopeConfig::text())
                    .map_err(serde::ser::Error::custom)?;
                serializer.collect_str(&str)
            }
        }

        impl serde_with::SerializeAs<$crate::Hugr> for $adaptor {
            fn serialize_as<S>(source: &$crate::Hugr, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                // Include any additional extension required to load the HUGR in the envelope.
                let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                let mut extra_extensions = $crate::extension::ExtensionRegistry::default();
                for ext in $crate::hugr::views::HugrView::extensions(source).iter() {
                    if !extensions.contains(ext.name()) {
                        extra_extensions.register_updated(ext.clone());
                    }
                }

                let str = source
                    .store_str_with_exts(
                        $crate::envelope::EnvelopeConfig::text(),
                        &extra_extensions,
                    )
                    .map_err(serde::ser::Error::custom)?;
                serializer.collect_str(&str)
            }
        }
    };
}
pub use impl_serde_as_string_envelope;

impl_serde_as_string_envelope!(AsStringEnvelope, &STD_REG);

/// Implements [`serde_with::DeserializeAs`] and [`serde_with::SerializeAs`] for
/// the helper to deserialize `Hugr` and `Package` types, using the given
/// extension registry.
///
/// This macro is used to implement the default [`AsBinaryEnvelope`] wrapper.
///
/// # Parameters
///
/// - `$adaptor`: The name of the adaptor type to implement.
/// - `$extension_reg`: A reference to the extension registry to use for deserialization.
///
/// # Examples
///
/// ```rust
/// # use serde::{Deserialize, Serialize};
/// # use serde_json::json;
/// # use serde_with::{serde_as};
/// # use hugr_core::Hugr;
/// # use hugr_core::package::Package;
/// # use hugr_core::envelope::serde_with::AsBinaryEnvelope;
/// # use hugr_core::envelope::serde_with::impl_serde_as_binary_envelope;
/// # use hugr_core::extension::ExtensionRegistry;
/// #
/// struct CustomAsEnvelope;
///
/// impl_serde_as_binary_envelope!(CustomAsEnvelope, &hugr_core::extension::EMPTY_REG);
///
/// #[serde_as]
/// #[derive(Deserialize, Serialize)]
/// struct A {
///     #[serde_as(as = "CustomAsEnvelope")]
///     package: Package,
/// }
/// ```
///
#[macro_export]
macro_rules! impl_serde_as_binary_envelope {
    ($adaptor:ident, $extension_reg:expr) => {
        impl<'de> serde_with::DeserializeAs<'de, $crate::package::Package> for $adaptor {
            fn deserialize_as<D>(deserializer: D) -> Result<$crate::package::Package, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Helper;
                impl serde::de::Visitor<'_> for Helper {
                    type Value = $crate::package::Package;

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
                        formatter.write_str("a base64-encoded envelope")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        use $crate::envelope::serde_with::base64::{DecoderReader, STANDARD};

                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;

                        if value
                            .as_bytes()
                            .starts_with($crate::envelope::MAGIC_NUMBERS)
                        {
                            // If the string starts with the envelope magic 'HUGRiHJv',
                            // skip the base64 decoding.
                            let reader = std::io::Cursor::new(value.as_bytes());
                            $crate::package::Package::load(reader, Some(extensions))
                                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))
                        } else {
                            let reader = DecoderReader::new(value.as_bytes(), &STANDARD);
                            let buf_reader = std::io::BufReader::new(reader);
                            $crate::package::Package::load(buf_reader, Some(extensions))
                                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))
                        }
                    }
                }

                deserializer.deserialize_str(Helper)
            }
        }

        impl<'de> serde_with::DeserializeAs<'de, $crate::Hugr> for $adaptor {
            fn deserialize_as<D>(deserializer: D) -> Result<$crate::Hugr, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct Helper;
                impl<'vis> serde::de::Visitor<'vis> for Helper {
                    type Value = $crate::Hugr;

                    fn expecting(
                        &self,
                        formatter: &mut std::fmt::Formatter<'_>,
                    ) -> std::fmt::Result {
                        formatter.write_str("a base64-encoded envelope")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        use $crate::envelope::serde_with::base64::{DecoderReader, STANDARD};

                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;

                        if value
                            .as_bytes()
                            .starts_with($crate::envelope::MAGIC_NUMBERS)
                        {
                            // If the string starts with the envelope magic 'HUGRiHJv',
                            // skip the base64 decoding.
                            let reader = std::io::Cursor::new(value.as_bytes());
                            $crate::Hugr::load(reader, Some(extensions))
                                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))
                        } else {
                            let reader = DecoderReader::new(value.as_bytes(), &STANDARD);
                            let buf_reader = std::io::BufReader::new(reader);
                            $crate::Hugr::load(buf_reader, Some(extensions))
                                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))
                        }
                    }

                    fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::MapAccess<'vis>,
                    {
                        // Backwards compatibility: If the encoded value is not a
                        // string, we may have a legacy HUGR serde structure instead. In that
                        // case, we can add an envelope header and try again.
                        //
                        // TODO: Remove this fallback in a breaking change
                        let deserializer = serde::de::value::MapAccessDeserializer::new(map);
                        #[expect(deprecated)]
                        let mut hugr =
                            $crate::hugr::serialize::serde_deserialize_hugr(deserializer)
                                .map_err(serde::de::Error::custom)?;

                        let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                        hugr.resolve_extension_defs(extensions)
                            .map_err(serde::de::Error::custom)?;
                        Ok(hugr)
                    }
                }

                // TODO: Go back to `deserialize_str` once the fallback is removed.
                deserializer.deserialize_any(Helper)
            }
        }

        impl serde_with::SerializeAs<$crate::package::Package> for $adaptor {
            fn serialize_as<S>(
                source: &$crate::package::Package,
                serializer: S,
            ) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                use $crate::envelope::serde_with::base64::{EncoderStringWriter, STANDARD};

                let mut writer = EncoderStringWriter::new(&STANDARD);
                source
                    .store(&mut writer, $crate::envelope::EnvelopeConfig::binary())
                    .map_err(serde::ser::Error::custom)?;
                let str = writer.into_inner();
                serializer.collect_str(&str)
            }
        }

        impl serde_with::SerializeAs<$crate::Hugr> for $adaptor {
            fn serialize_as<S>(source: &$crate::Hugr, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                // Include any additional extension required to load the HUGR in the envelope.
                let extensions: &$crate::extension::ExtensionRegistry = $extension_reg;
                let mut extra_extensions = $crate::extension::ExtensionRegistry::default();
                for ext in $crate::hugr::views::HugrView::extensions(source).iter() {
                    if !extensions.contains(ext.name()) {
                        extra_extensions.register_updated(ext.clone());
                    }
                }
                use $crate::envelope::serde_with::base64::{EncoderStringWriter, STANDARD};

                let mut writer = EncoderStringWriter::new(&STANDARD);
                source
                    .store_with_exts(
                        &mut writer,
                        $crate::envelope::EnvelopeConfig::binary(),
                        &extra_extensions,
                    )
                    .map_err(serde::ser::Error::custom)?;
                let str = writer.into_inner();
                serializer.collect_str(&str)
            }
        }
    };
}
pub use impl_serde_as_binary_envelope;

impl_serde_as_binary_envelope!(AsBinaryEnvelope, &STD_REG);

// Hidden re-export required to expand the binary envelope macros on external
// crates.
#[doc(hidden)]
pub mod base64 {
    pub use base64::Engine;
    pub use base64::engine::general_purpose::STANDARD;
    pub use base64::read::DecoderReader;
    pub use base64::write::EncoderStringWriter;
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use serde::{Deserialize, Serialize};
    use serde_with::serde_as;

    use crate::Hugr;
    use crate::package::Package;

    use super::*;

    #[serde_as]
    #[derive(Deserialize, Serialize)]
    struct TextPkg {
        #[serde_as(as = "AsStringEnvelope")]
        data: Package,
    }

    #[serde_as]
    #[derive(Default, Deserialize, Serialize)]
    struct TextHugr {
        #[serde_as(as = "AsStringEnvelope")]
        data: Hugr,
    }

    #[serde_as]
    #[derive(Deserialize, Serialize)]
    struct BinaryPkg {
        #[serde_as(as = "AsBinaryEnvelope")]
        data: Package,
    }

    #[serde_as]
    #[derive(Default, Deserialize, Serialize)]
    struct BinaryHugr {
        #[serde_as(as = "AsBinaryEnvelope")]
        data: Hugr,
    }

    #[derive(Default, Deserialize, Serialize)]
    struct LegacyHugr {
        #[serde(deserialize_with = "Hugr::serde_deserialize")]
        #[serde(serialize_with = "Hugr::serde_serialize")]
        data: Hugr,
    }

    impl Default for TextPkg {
        fn default() -> Self {
            // Default package with a single hugr (so it can be decoded as a hugr too).
            Self {
                data: Package::from_hugr(Hugr::default()),
            }
        }
    }

    impl Default for BinaryPkg {
        fn default() -> Self {
            // Default package with a single hugr (so it can be decoded as a hugr too).
            Self {
                data: Package::from_hugr(Hugr::default()),
            }
        }
    }

    fn decode<T: for<'a> serde::Deserialize<'a>>(encoded: String) -> Result<(), serde_json::Error> {
        let _: T = serde_json::de::from_str(&encoded)?;
        Ok(())
    }

    #[rstest]
    // Text formats are swappable
    #[case::text_pkg_text_pkg(TextPkg::default(), decode::<TextPkg>, false)]
    #[case::text_pkg_text_hugr(TextPkg::default(), decode::<TextHugr>, false)]
    #[case::text_hugr_text_pkg(TextHugr::default(), decode::<TextPkg>, false)]
    #[case::text_hugr_text_hugr(TextHugr::default(), decode::<TextHugr>, false)]
    // Binary formats can read each other
    #[case::bin_pkg_bin_pkg(BinaryPkg::default(), decode::<BinaryPkg>, false)]
    #[case::bin_pkg_bin_hugr(BinaryPkg::default(), decode::<BinaryHugr>, false)]
    #[case::bin_hugr_bin_pkg(BinaryHugr::default(), decode::<BinaryPkg>, false)]
    #[case::bin_hugr_bin_hugr(BinaryHugr::default(), decode::<BinaryHugr>, false)]
    // Binary formats can read text ones
    #[case::text_pkg_bin_pkg(TextPkg::default(), decode::<BinaryPkg>, false)]
    #[case::text_pkg_bin_hugr(TextPkg::default(), decode::<BinaryHugr>, false)]
    #[case::text_hugr_bin_pkg(TextHugr::default(), decode::<BinaryPkg>, false)]
    #[case::text_hugr_bin_hugr(TextHugr::default(), decode::<BinaryHugr>, false)]
    // But text formats can't read binary
    #[case::bin_pkg_text_pkg(BinaryPkg::default(), decode::<TextPkg>, true)]
    #[case::bin_pkg_text_hugr(BinaryPkg::default(), decode::<TextHugr>, true)]
    #[case::bin_hugr_text_pkg(BinaryHugr::default(), decode::<TextPkg>, true)]
    #[case::bin_hugr_text_hugr(BinaryHugr::default(), decode::<TextHugr>, true)]
    // We can read old hugrs into hugrs, but not packages
    #[case::legacy_hugr_text_pkg(LegacyHugr::default(), decode::<TextPkg>, true)]
    #[case::legacy_hugr_text_hugr(LegacyHugr::default(), decode::<TextHugr>, false)]
    #[case::legacy_hugr_bin_pkg(LegacyHugr::default(), decode::<BinaryPkg>, true)]
    #[case::legacy_hugr_bin_hugr(LegacyHugr::default(), decode::<BinaryHugr>, false)]
    // Decoding any new format as legacy hugr always fails
    #[case::text_pkg_legacy_hugr(TextPkg::default(), decode::<LegacyHugr>, true)]
    #[case::text_hugr_legacy_hugr(TextHugr::default(), decode::<LegacyHugr>, true)]
    #[case::bin_pkg_legacy_hugr(BinaryPkg::default(), decode::<LegacyHugr>, true)]
    #[case::bin_hugr_legacy_hugr(BinaryHugr::default(), decode::<LegacyHugr>, true)]
    #[cfg_attr(all(miri, feature = "zstd"), ignore)] // FFI calls (required to compress with zstd) are not supported in miri
    fn check_format_compatibility(
        #[case] encoder: impl serde::Serialize,
        #[case] decoder: fn(String) -> Result<(), serde_json::Error>,
        #[case] errors: bool,
    ) {
        let encoded = serde_json::to_string(&encoder).unwrap();
        let decoded = decoder(encoded);
        match (errors, decoded) {
            (false, Err(e)) => {
                panic!("Decoding error: {e}");
            }
            (true, Ok(_)) => {
                panic!("Roundtrip should have failed");
            }
            _ => {}
        }
    }
}
