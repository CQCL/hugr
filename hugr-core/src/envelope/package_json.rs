//! Encoding / decoding of Package json, used in the `PackageJson` envelope format.
use derive_more::{Display, Error, From};
use itertools::Itertools;
use std::io;

use crate::extension::resolution::ExtensionResolutionError;
use crate::extension::ExtensionRegistry;
use crate::hugr::ExtensionError;
use crate::package::Package;
use crate::{Extension, Hugr};

/// Read a Package in json format from an io reader.
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
    )?;

    // Resolve the operations in the modules using the defined registries.
    let mut combined_registry = extension_registry.clone();
    combined_registry.extend(&pkg_extensions);

    for module in &mut modules {
        module.resolve_extension_defs(&combined_registry)?;
    }

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
        extensions: extensions.iter().map(|e| e.as_ref()).collect(),
    };
    serde_json::to_writer(writer, &pkg_ser)?;
    Ok(())
}

/// Error raised while loading a package.
#[derive(Debug, Display, Error, From)]
#[non_exhaustive]
pub enum PackageEncodingError {
    /// Error raised while parsing the package json.
    JsonEncoding(serde_json::Error),
    /// Error raised while reading from a file.
    IOError(io::Error),
    /// Could not resolve the extension needed to encode the hugr.
    ExtensionResolution(ExtensionResolutionError),
    /// Could not resolve the runtime extensions for the hugr.
    RuntimeExtensionResolution(ExtensionError),
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
