use crate::envelope::EnvelopeHeader;

#[derive(Debug, Clone, PartialEq)]
pub struct PackageDescription {
    pub header: EnvelopeHeader,
    pub modules: Vec<Option<ModuleDescription>>,
    pub packaged_extensions: Vec<Option<ExtensionDescription>>,
}

#[derive(derive_more::Display, Debug, Clone, PartialEq)]
#[display("Extension {name} v{version}")]
pub struct ExtensionDescription {
    /// Name of the extension.
    pub name: String,
    /// Version of the extension.
    pub version: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDescription {
    /// Generator specified in the module metadata.
    pub generator: Option<String>,
    /// Generator specified used extensions in the module metadata.
    pub used_extensions_metadata: Option<Vec<ExtensionDescription>>,
    /// Extensions used in the module computed while resolving, expected to be a subset of `used_extensions_metadata`.
    pub used_extensions_resolved: Option<Vec<ExtensionDescription>>,
    /// Public symbols defined in the module.
    pub public_symbols: Option<Vec<String>>,
}

impl std::fmt::Display for PackageDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl std::fmt::Display for ModuleDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
