use crate::{
    Hugr,
    extension::{EMPTY_REG, ExtensionRegistry},
};
use serde::{Deserializer, Serializer};

/// A seed for deserializing a value with a given set of extensions.
///
/// Values deserializable with [`ExtensionsSeed`] will provide an implementation
/// of [`serde::de::DeserializeSeed`] for [`ExtensionsSeed<V>`].
pub struct ExtensionsSeed<'a, V> {
    /// The extensions to use for deserialization.
    pub extensions: &'a ExtensionRegistry,
    _marker: std::marker::PhantomData<V>,
}

impl<'a, V> ExtensionsSeed<'a, V> {
    /// Create a new `ExtensionsSeed` with the given value and extensions.
    pub fn new(extensions: &'a ExtensionRegistry) -> Self {
        Self {
            extensions,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new `ExtensionsSeed` with no extensions.
    pub fn empty() -> Self {
        Self {
            extensions: &EMPTY_REG,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for ExtensionsSeed<'_, Hugr> {
    type Value = Hugr;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Helper<'a>(&'a ExtensionRegistry);
        impl serde::de::Visitor<'_> for Helper<'_> {
            type Value = Hugr;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a string-encoded envelope")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Hugr::load_str(value, Some(self.0)).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_str(Helper(self.extensions))
    }
}

impl serde::Serialize for Hugr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let str = self
            .store_str(crate::envelope::EnvelopeConfig::text())
            .map_err(serde::ser::Error::custom)?;
        serializer.collect_str(&str)
    }
}
