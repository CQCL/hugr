//! Bundles of hugr modules along with the extension required to load them.

use std::io;

use crate::envelope::{EnvelopeConfig, EnvelopeError, read_envelope, write_envelope};
use crate::extension::ExtensionRegistry;
use crate::hugr::{HugrView, ValidationError};
use crate::std_extensions::STD_REG;
use crate::{Hugr, Node};
use thiserror::Error;

#[derive(Debug, Default, Clone, PartialEq)]
/// Package of module HUGRs.
pub struct Package {
    /// Module HUGRs included in the package.
    pub modules: Vec<Hugr>,
    /// Extensions used in the modules.
    ///
    /// This is a superset of the extensions used in the modules.
    pub extensions: ExtensionRegistry,
}

impl Package {
    /// Create a new package from a list of hugrs.
    ///
    /// The hugr extensions are not automatically added to the package. Make
    /// sure to manually include any non-standard extensions to
    /// [`Package::extensions`].
    pub fn new(modules: impl IntoIterator<Item = Hugr>) -> Self {
        let modules: Vec<Hugr> = modules.into_iter().collect();
        Self {
            modules,
            extensions: ExtensionRegistry::default(),
        }
    }

    /// Create a new package containing a single HUGR.
    ///
    /// The hugr extensions are not automatically added to the package. Make
    /// sure to manually include any non-standard extensions to
    /// [`Package::extensions`].
    pub fn from_hugr(hugr: Hugr) -> Self {
        Package {
            extensions: ExtensionRegistry::default(),
            modules: vec![hugr],
        }
    }

    /// Validate the modules of the package.
    ///
    /// Ensures that the top-level extension list is a superset of the extensions used in the modules.
    pub fn validate(&self) -> Result<(), PackageValidationError> {
        for hugr in &self.modules {
            hugr.validate()?;
        }
        Ok(())
    }

    /// Read a Package from a HUGR envelope.
    ///
    /// To load a Package, all the extensions used in its definition must be
    /// available. The Envelope may include some of the extensions, but any
    /// additional extensions must be provided in the `extensions` parameter. If
    /// `extensions` is `None`, the default [`crate::std_extensions::STD_REG`]
    /// is used.
    pub fn load(
        reader: impl io::BufRead,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        let extensions = extensions.unwrap_or(&STD_REG);
        let (_, pkg) = read_envelope(reader, extensions)?;
        Ok(pkg)
    }

    /// Read a Package from a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::load` with a bytearray instead.
    ///
    /// To load a Package, all the extensions used in its definition must be
    /// available. The Envelope may include some of the extensions, but any
    /// additional extensions must be provided in the `extensions` parameter. If
    /// `extensions` is `None`, the default [`crate::std_extensions::STD_REG`]
    /// is used.
    pub fn load_str(
        envelope: impl AsRef<str>,
        extensions: Option<&ExtensionRegistry>,
    ) -> Result<Self, EnvelopeError> {
        Self::load(envelope.as_ref().as_bytes(), extensions)
    }

    /// Store the Package in a HUGR envelope.
    ///
    /// The Envelope will embed the definitions of the extensions in
    /// [`Package::extensions`]. Any other extension used in the definition must
    /// be passed to [`Package::load`] to load back the package.
    pub fn store(
        &self,
        writer: impl io::Write,
        config: EnvelopeConfig,
    ) -> Result<(), EnvelopeError> {
        write_envelope(writer, self, config)
    }

    /// Store the Package in a HUGR envelope encoded in a string.
    ///
    /// Note that not all envelopes are valid strings. In the general case,
    /// it is recommended to use `Package::store` with a bytearray instead.
    /// See [`EnvelopeFormat::ascii_printable`][crate::envelope::EnvelopeFormat::ascii_printable].
    ///
    /// The Envelope will embed the definitions of the extensions in
    /// [`Package::extensions`]. Any other extension used in the definition must
    /// be passed to [`Package::load_str`] to load back the package.
    pub fn store_str(&self, config: EnvelopeConfig) -> Result<String, EnvelopeError> {
        if !config.format.ascii_printable() {
            return Err(EnvelopeError::NonASCIIFormat {
                format: config.format,
            });
        }

        let mut buf = Vec::new();
        self.store(&mut buf, config)?;
        Ok(String::from_utf8(buf).expect("Envelope is valid utf8"))
    }
}

impl AsRef<[Hugr]> for Package {
    fn as_ref(&self) -> &[Hugr] {
        &self.modules
    }
}

/// Error raised while validating a package.
#[derive(Debug, Error)]
#[non_exhaustive]
#[error("Package validation error.")]
pub enum PackageValidationError {
    /// Error raised while validating the package hugrs.
    Validation(#[from] ValidationError<Node>),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::builder::test::{
        simple_cfg_hugr, simple_dfg_hugr, simple_funcdef_hugr, simple_module_hugr,
    };
    use rstest::rstest;

    #[rstest]
    #[case::module("module", simple_module_hugr())]
    #[case::funcdef("funcdef", simple_funcdef_hugr())]
    #[case::dfg("dfg", simple_dfg_hugr())]
    #[case::cfg("cfg", simple_cfg_hugr())]
    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    fn hugr_to_package(#[case] test_name: &str, #[case] hugr: Hugr) {
        let package = &Package::from_hugr(hugr.clone());
        assert_eq!(package.modules.len(), 1);

        assert_eq!(
            package.modules[0].entrypoint_optype(),
            hugr.entrypoint_optype()
        );

        insta::assert_snapshot!(test_name, hugr.mermaid_string());
    }

    #[rstest]
    fn package_properties() {
        let module = simple_module_hugr();
        let dfg = simple_dfg_hugr();

        let pkg = Package::new([module, dfg]);
        pkg.validate().unwrap();

        assert_eq!(pkg.modules.len(), 2);
    }
}
