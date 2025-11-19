//! Encoding / decoding of Package json, used in the `PackageJson` envelope format.
use derive_more::{Display, Error};
use std::io;

use crate::extension::ExtensionRegistry;
use crate::extension::resolution::ExtensionResolutionError;

use crate::{Extension, Hugr};

/// Write the Package in json format into an io writer.
pub(super) fn to_json_writer<'h>(
    hugrs: impl IntoIterator<Item = &'h Hugr>,
    extensions: &ExtensionRegistry,
    writer: impl io::Write,
) -> Result<(), PackageEncodingError> {
    let pkg_ser = PackageSer {
        modules: hugrs.into_iter().map(HugrSer).collect(),
        extensions: extensions.iter().map(std::convert::AsRef::as_ref).collect(),
    };

    // Validate the hugr serializations against the schema.
    //
    // NOTE: The schema definition is currently broken, so this check always succeeds.
    // See <https://github.com/CQCL/hugr/issues/2401>
    #[cfg(all(test, not(miri)))]
    if std::env::var("HUGR_TEST_SCHEMA").is_ok_and(|x| !x.is_empty()) {
        use crate::hugr::serialize::test::check_hugr_serialization_schema;

        for hugr in &pkg_ser.modules {
            check_hugr_serialization_schema(hugr.0);
        }
    }

    serde_json::to_writer(writer, &pkg_ser).map_err(PackageEncodingError::JsonEncoding)?;
    Ok(())
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error)]
#[non_exhaustive]
#[display("Error reading or writing a package in JSON format.")]
pub enum PackageEncodingError {
    /// Error raised while parsing the package json.
    JsonEncoding(serde_json::Error),
    /// Error raised while reading from a file.
    IOError(io::Error),
    /// Could not resolve the extension needed to encode the hugr.
    ExtensionResolution(ExtensionResolutionError),
    /// Error resolving packaged extensions.
    PackagedExtension(ExtensionResolutionError),
}

/// A private package structure implementing the serde traits.
///
/// We use this to avoid exposing a public implementation of Serialize/Deserialize,
/// as the json definition is not stable, and should always be wrapped in an Envelope.
#[derive(Debug, serde::Serialize)]
struct PackageSer<'h> {
    pub modules: Vec<HugrSer<'h>>,
    pub extensions: Vec<&'h Extension>,
}
#[derive(Debug, serde::Serialize)]
#[serde(transparent)]
struct HugrSer<'h>(#[serde(serialize_with = "Hugr::serde_serialize")] pub &'h Hugr);

/// A private package structure implementing the serde traits.
///
/// We use this to avoid exposing a public implementation of Serialize/Deserialize,
/// as the json definition is not stable, and should always be wrapped in an Envelope.
#[derive(Debug, serde::Deserialize)]
pub(super) struct PackageDeser {
    pub modules: Vec<HugrDeser>,
    pub extensions: Vec<Extension>,
}
#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub(super) struct HugrDeser(#[serde(deserialize_with = "Hugr::serde_deserialize")] pub Hugr);
