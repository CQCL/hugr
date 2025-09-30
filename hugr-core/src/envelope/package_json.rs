//! Encoding / decoding of Package json, used in the `PackageJson` envelope format.
use derive_more::{Display, Error, From};
use itertools::Itertools;
use std::io;

use super::WithGenerator;
use crate::extension::ExtensionRegistry;
use crate::extension::resolution::ExtensionResolutionError;
use crate::package::Package;
use crate::{Extension, Hugr};

/// Read a Package in json format from an io reader.
/// Returns package and the combined extension registry
/// of the provided registry and the package extensions.
pub(super) fn from_json_reader(
    reader: impl io::Read,
    extension_registry: &ExtensionRegistry,
) -> Result<Package, PackageEncodingError> {
    let val: serde_json::Value = serde_json::from_reader(reader)?;

    let PackageDeser {
        modules,
        extensions: pkg_extensions,
    } = serde_json::from_value::<PackageDeser>(val.clone())?;
    let mut modules = modules.into_iter().map(|h| h.0).collect_vec();
    let pkg_extensions = ExtensionRegistry::new_with_extension_resolution(
        pkg_extensions,
        &extension_registry.into(),
    )
    .map_err(|err| WithGenerator::new(err, &modules))?;

    // Resolve the operations in the modules using the defined registries.
    let mut combined_registry = extension_registry.clone();
    combined_registry.extend(&pkg_extensions);

    modules
        .iter_mut()
        .try_for_each(|module| module.resolve_extension_defs(&combined_registry))
        .map_err(|err| WithGenerator::new(err, &modules))?;

    Ok(Package {
        modules,
        extensions: pkg_extensions,
    })
}

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

    serde_json::to_writer(writer, &pkg_ser)?;
    Ok(())
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
#[display("Error reading or writing a package in JSON format.")]
pub enum PackageEncodingError {
    /// Error raised while parsing the package json.
    JsonEncoding(#[from] serde_json::Error),
    /// Error raised while reading from a file.
    IOError(#[from] io::Error),
    /// Could not resolve the extension needed to encode the hugr.
    ExtensionResolution(#[from] WithGenerator<ExtensionResolutionError>),
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
struct PackageDeser {
    pub modules: Vec<HugrDeser>,
    pub extensions: Vec<Extension>,
}
#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
struct HugrDeser(#[serde(deserialize_with = "Hugr::serde_deserialize")] pub Hugr);
