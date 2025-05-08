//! Derivation to serialize and deserialize Hugrs and Packages as envelopes in a
//! serde compatible way.
//!
//! These are meant to be used with `serde_with`'s `#[serde_as]` decorator, see
//! <https://docs.rs/serde_with/latest/serde_with>.

use serde::Deserializer;
use serde::{de, Serializer};
use serde_with::SerializeAs;

use crate::package::Package;
use crate::Hugr;

use super::EnvelopeConfig;

/// De/Serialize a package or hugr by encoding it into a textual Envelope and
/// storing it as a string.
///
/// Note that only PRELUDE extensions are used to decode the package's content.
/// Additional extensions should be included in the serialized envelope.
///
// TODO: Support parametrizing the extensions somehow? Not sure if possible.
// The current impl will use the PRELUDE extensions, plus any one encoded in the
// package definition.
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
/// compatibility layer is meant to be removed in 0.21.0.
pub struct AsStringEnvelope;

impl<'de> serde_with::DeserializeAs<'de, Package> for AsStringEnvelope {
    fn deserialize_as<D>(deserializer: D) -> Result<Package, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Helper;
        impl de::Visitor<'_> for Helper {
            type Value = Package;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string-encoded envelope")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Package::load_str(value, None).map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(Helper)
    }
}

impl<'de> serde_with::DeserializeAs<'de, Hugr> for AsStringEnvelope {
    fn deserialize_as<D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Helper;
        impl<'vis> de::Visitor<'vis> for Helper {
            type Value = Hugr;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string-encoded envelope")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Hugr::load_str(value, None).map_err(de::Error::custom)
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'vis>,
            {
                // Backwards compatibility: If the encoded value is not a
                // string, we may have a legacy HUGR serde structure instead. In that
                // case, we can add an envelope header and try again.
                //
                // TODO: Remove this fallback in 0.21.0
                let deserializer = serde::de::value::MapAccessDeserializer::new(map);
                Hugr::serde_deserialize(deserializer).map_err(de::Error::custom)
            }
        }

        // TODO: Go back to `deserialize_str` once the fallback is removed.
        deserializer.deserialize_any(Helper)
    }
}

impl SerializeAs<Package> for AsStringEnvelope {
    fn serialize_as<S>(source: &Package, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let str = source
            .store_str(EnvelopeConfig::text())
            .map_err(serde::ser::Error::custom)?;
        serializer.collect_str(&str)
    }
}

impl SerializeAs<Hugr> for AsStringEnvelope {
    fn serialize_as<S>(source: &Hugr, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let str = source
            .store_str(EnvelopeConfig::text())
            .map_err(serde::ser::Error::custom)?;
        serializer.collect_str(&str)
    }
}
