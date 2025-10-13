use crate::envelope::EnvelopeHeader;

#[derive(Clone, Debug, PartialEq)]
struct PartialVec<T> {
    vec: Vec<Option<T>>,
}
impl<T> Default for PartialVec<T> {
    fn default() -> Self {
        Self { vec: Vec::new() }
    }
}

impl<T: Clone> PartialVec<T> {
    fn set_len(&mut self, n: usize) {
        self.vec.resize(n, None);
    }
    fn set_index(&mut self, index: usize, value: T) {
        if index >= self.vec.len() {
            self.vec.resize(index + 1, None);
        }
        self.vec[index] = Some(value);
    }
    fn len(&self) -> usize {
        self.vec.len()
    }
    fn into_vec(self) -> Vec<Option<T>> {
        self.vec
    }

    fn merge(&mut self, other: Self) {
        self.set_len(self.len().max(other.len()));
        for (i, item) in other.vec.into_iter().enumerate() {
            if let Some(item) = item {
                self.set_index(i, item);
            }
        }
    }
}

pub trait MergeDescriptions {
    fn merge(self, other: Self) -> Self;
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PackageDesc {
    pub header: EnvelopeHeader,
    modules: PartialVec<ModuleDescription>,
    packaged_extensions: PartialVec<ExtensionDescr>,
}

impl PackageDesc {
    pub fn new(header: EnvelopeHeader) -> Self {
        Self {
            header,
            ..Default::default()
        }
    }
    pub fn with_n_modules(mut self, n: usize) -> Self {
        self.modules.set_len(n);
        self
    }
    pub fn n_modules(&self) -> usize {
        self.modules.len()
    }
    pub fn with_module(mut self, index: usize, module: ModuleDescription) -> Self {
        self.modules.set_index(index, module);
        self
    }
    pub fn with_n_packaged_extensions(mut self, n: usize) -> Self {
        self.packaged_extensions.set_len(n);
        self
    }
    pub fn n_packaged_extensions(&self) -> usize {
        self.packaged_extensions.len()
    }
}

impl MergeDescriptions for PackageDesc {
    fn merge(mut self, other: Self) -> Self {
        self.modules.merge(other.modules);
        self.packaged_extensions.merge(other.packaged_extensions);
        self
    }
}

#[derive(derive_more::Display, Debug, Clone, PartialEq)]
#[display("Extension {name} v{version}")]
pub struct ExtensionDescr {
    /// Name of the extension.
    pub name: String,
    /// Version of the extension.
    pub version: String,
}

impl ExtensionDescr {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModuleDescription {
    /// Generator specified in the module metadata.
    pub generator: Option<String>,
    /// Generator specified used extensions in the module metadata.
    pub used_extensions_metadata: Option<Vec<ExtensionDescr>>,
    /// Extensions used in the module computed while resolving, expected to be a subset of `used_extensions_metadata`.
    pub used_extensions_resolved: Option<Vec<ExtensionDescr>>,
    /// Public symbols defined in the module.
    pub public_symbols: Option<Vec<String>>,
}

impl ModuleDescription {
    pub fn with_generator(mut self, generator: impl Into<String>) -> Self {
        self.generator = Some(generator.into());
        self
    }
    pub fn with_used_extensions_metadata(
        mut self,
        used_extensions_metadata: impl IntoIterator<Item = ExtensionDescr>,
    ) -> Self {
        self.used_extensions_metadata = Some(used_extensions_metadata.into_iter().collect());
        self
    }
    pub fn with_used_extensions_resolved(
        mut self,
        used_extensions_resolved: impl IntoIterator<Item = ExtensionDescr>,
    ) -> Self {
        self.used_extensions_resolved = Some(used_extensions_resolved.into_iter().collect());
        self
    }
    pub fn with_public_symbols(mut self, public_symbols: Vec<String>) -> Self {
        self.public_symbols = Some(public_symbols);
        self
    }
}

impl std::fmt::Display for PackageDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl std::fmt::Display for ModuleDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
