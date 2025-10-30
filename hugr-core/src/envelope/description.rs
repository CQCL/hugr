//! Description of the contents of a HUGR envelope used for debugging and error reporting.
use itertools::Itertools;
use semver::Version;

use crate::{
    HugrView, Node,
    envelope::{EnvelopeHeader, USED_EXTENSIONS_KEY},
    ops::{DataflowOpTrait, OpType},
};

mod wrapper {
    use super::ModuleDesc;

    use super::PackageDesc;

    /// Wrapper type associating a value with its description.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Described<T, D> {
        val: T,
        description: D,
    }

    impl<T, D> AsRef<T> for Described<T, D> {
        fn as_ref(&self) -> &T {
            &self.val
        }
    }

    impl<T, D> AsMut<T> for Described<T, D> {
        fn as_mut(&mut self) -> &mut T {
            &mut self.val
        }
    }

    impl<T, D> Described<T, D> {
        /// Create a new described value.
        pub fn new(val: T, description: D) -> Self {
            Self { val, description }
        }

        /// Unwrap the described value, discarding the description.
        pub fn into_inner(self) -> T {
            self.val
        }

        /// Get a reference to the description.
        pub fn description(&self) -> &D {
            &self.description
        }

        /// Map the described value to another value, keeping the description.
        pub fn map<F, U>(self, f: F) -> Described<U, D>
        where
            F: FnOnce(T) -> U,
        {
            Described {
                val: f(self.val),
                description: self.description,
            }
        }

        /// Consume the described value and return the inner value and description.
        pub fn into_parts(self) -> (T, D) {
            (self.val, self.description)
        }
    }

    /// Result type wrapped with a package description.
    pub type PackageDescResult<T, E> = Described<Result<T, E>, PackageDesc>;

    /// Result type wrapped with a module description.
    pub type ModuleDescResult<T, E> = Described<Result<T, E>, ModuleDesc>;

    /// Package wrapped with its description.
    pub type DescribedPackage = Described<crate::package::Package, PackageDesc>;

    /// Transpose a described result into a result of a described value or described error.
    pub fn transpose<T, E, D>(
        desc_result: Described<Result<T, E>, D>,
    ) -> Result<Described<T, D>, Described<E, D>> {
        match desc_result.val {
            Ok(val) => Ok(Described {
                val,
                description: desc_result.description,
            }),
            Err(err) => Err(Described {
                val: err,
                description: desc_result.description,
            }),
        }
    }
}

pub use wrapper::DescribedPackage;
pub(crate) use wrapper::{Described, ModuleDescResult, PackageDescResult, transpose};

type PartialVec<T> = Vec<Option<T>>;
fn set_partial_len<T: Clone>(vec: &mut PartialVec<T>, n: usize) {
    vec.resize(n, None);
}
fn set_partial_index<T: Clone>(vec: &mut PartialVec<T>, index: usize, value: T) {
    if index >= vec.len() {
        set_partial_len(vec, index + 1);
    }
    vec[index] = Some(value);
}

/// High-level description of a HUGR package.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct PackageDesc {
    /// Envelope header information.
    #[serde(serialize_with = "header_serialize")]
    pub header: EnvelopeHeader,
    /// Description of the modules in the package.
    pub modules: PartialVec<ModuleDesc>,
    /// Description of the extensions in the package.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub packaged_extensions: PartialVec<ExtensionDesc>,
}

fn header_serialize<S>(header: &EnvelopeHeader, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&header.to_string())
}

impl PackageDesc {
    /// Creates a new `PackageDesc` with the given header.
    pub(super) fn new(header: EnvelopeHeader) -> Self {
        Self {
            header,
            ..Default::default()
        }
    }

    /// Sets the number of modules in the package.
    pub(crate) fn set_n_modules(&mut self, n: usize) {
        set_partial_len(&mut self.modules, n);
    }

    /// Returns the package header.
    pub fn header(&self) -> EnvelopeHeader {
        self.header
    }

    /// Returns the number of modules in the package.
    pub fn n_modules(&self) -> usize {
        self.modules.len()
    }

    /// Sets a module description at the specified index.
    pub(crate) fn set_module(&mut self, index: usize, module: impl Into<ModuleDesc>) {
        set_partial_index(&mut self.modules, index, module.into());
    }

    /// Sets a packaged extension description at the specified index.
    pub(crate) fn set_packaged_extension(&mut self, index: usize, ext: impl Into<ExtensionDesc>) {
        set_partial_index(&mut self.packaged_extensions, index, ext.into());
    }

    /// Returns the number of packaged extensions in the package.
    pub fn n_packaged_extensions(&self) -> usize {
        self.packaged_extensions.len()
    }

    /// Returns the generator(s) of the package modules, if any.
    /// Concatenates multiple generators with commas.
    pub fn generator(&self) -> Option<String> {
        let generators: Vec<String> = self
            .modules
            .iter()
            .flatten()
            .flat_map(|m| &m.generator)
            .unique()
            .cloned()
            .collect();
        if generators.is_empty() {
            return None;
        }

        Some(generators.join(", "))
    }

    /// Returns an iterator over the module descriptions.
    pub fn modules(&self) -> impl Iterator<Item = &Option<ModuleDesc>> {
        self.modules.iter()
    }

    /// Returns an iterator over the packaged extension descriptions.
    pub fn packaged_extensions(&self) -> impl Iterator<Item = &Option<ExtensionDesc>> {
        self.packaged_extensions.iter()
    }

    /// Wraps a value with this package description.
    pub fn wrap<T>(self, val: T) -> Described<T, Self>
    where
        Self: Sized,
    {
        wrapper::Described::new(val, self)
    }
}

/// High level description of an extension.
#[derive(derive_more::Display, Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[display("Extension {name} v{version}")]
pub struct ExtensionDesc {
    /// Name of the extension.
    pub name: String,
    /// Version of the extension.
    pub version: Version,
}

impl ExtensionDesc {
    /// Create a new extension description.
    pub fn new(name: impl ToString, version: impl Into<Version>) -> Self {
        Self {
            name: name.to_string(),
            version: version.into(),
        }
    }
}

impl<E: AsRef<crate::Extension>> From<&E> for ExtensionDesc {
    fn from(ext: &E) -> Self {
        let ext = ext.as_ref();
        Self {
            name: ext.name.to_string(),
            version: ext.version.clone(),
        }
    }
}

fn extend_option_vec<T: Clone>(vec: &mut Option<Vec<T>>, items: impl IntoIterator<Item = T>) {
    if let Some(existing) = vec {
        existing.extend(items);
    } else {
        vec.replace(items.into_iter().collect());
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
/// Description of the entrypoint of a module.
pub struct Entrypoint {
    /// Node id of the entrypoint.
    pub node: Node,
    #[serde(serialize_with = "op_serialize")]
    /// Operation type of the entrypoint node.
    pub optype: OpType,
}

impl Entrypoint {
    /// Create a new entrypoint description.
    pub fn new(node: Node, optype: OpType) -> Self {
        Self { node, optype }
    }
}

/// Get a string representation of an OpType for description purposes.
pub fn op_string(op: &OpType) -> String {
    match op {
        OpType::FuncDefn(defn) => format!(
            "FuncDefn({})",
            func_symbol(defn.func_name(), defn.signature())
        ),
        OpType::FuncDecl(decl) => format!(
            "FuncDecl({})",
            func_symbol(decl.func_name(), decl.signature())
        ),
        OpType::DFG(dfg) => format!("DFG({})", dfg.signature()),
        _ => format!("{op}"),
    }
}
fn op_serialize<S>(op_type: &OpType, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(op_string(op_type).as_str())
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
/// High-level description of a module in a HUGR package.
pub struct ModuleDesc {
    /// Number of nodes in the module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_nodes: Option<usize>,
    /// The entrypoint node and the corresponding operation type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entrypoint: Option<Entrypoint>,
    /// Extensions used in the module computed while resolving, expected to be a subset of `used_extensions_generator`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_extensions_resolved: Option<Vec<ExtensionDesc>>,
    /// Generator specified in the module metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    /// Generator specified used extensions in the module metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub used_extensions_generator: Option<Vec<ExtensionDesc>>,
    /// Public symbols defined in the module.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_symbols: Option<Vec<String>>,
}

impl ModuleDesc {
    /// Sets the number of nodes in the module.
    pub fn set_num_nodes(&mut self, num_nodes: usize) {
        self.num_nodes = Some(num_nodes);
    }

    /// Sets the entrypoint of the module.
    pub fn set_entrypoint(&mut self, node: Node, optype: OpType) {
        self.entrypoint = Some(Entrypoint::new(node, optype));
    }

    /// Sets the generator for the module.
    pub fn set_generator(&mut self, generator: impl Into<String>) {
        self.generator = Some(generator.into());
    }

    /// Sets the extensions used by the generator in the module metadata.
    pub fn set_used_extensions_generator(
        &mut self,
        used_extensions_metadata: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        self.used_extensions_generator = Some(used_extensions_metadata.into_iter().collect());
    }

    /// Extends the extensions used by the generator in the module metadata.
    pub fn extend_used_extensions_metadata(
        &mut self,
        exts: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        extend_option_vec(&mut self.used_extensions_generator, exts);
    }

    /// Sets the resolved extensions used in the module.
    pub fn set_used_extensions_resolved(
        &mut self,
        used_extensions_resolved: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        self.used_extensions_resolved = Some(used_extensions_resolved.into_iter().collect());
    }

    /// Extends the resolved extensions used in the module.
    pub fn extend_used_extensions_resolved(
        &mut self,
        exts: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        extend_option_vec(&mut self.used_extensions_resolved, exts);
    }

    /// Sets the public symbols defined in the module.
    pub fn set_public_symbols(&mut self, symbols: impl IntoIterator<Item = String>) {
        self.public_symbols = Some(symbols.into_iter().collect());
    }

    /// Extends the public symbols defined in the module.
    pub fn extend_public_symbols(&mut self, symbols: impl IntoIterator<Item = String>) {
        extend_option_vec(&mut self.public_symbols, symbols);
    }

    /// Loads the generator from the HUGR metadata.
    pub(crate) fn load_generator(&mut self, hugr: &impl HugrView) {
        if let Some(val) = hugr.get_metadata(hugr.module_root(), crate::envelope::GENERATOR_KEY) {
            self.set_generator(super::format_generator(val));
        }
    }

    /// Loads the extensions used by the generator from the HUGR metadata.
    pub(crate) fn load_used_extensions_generator(
        &mut self,
        hugr: &impl HugrView,
    ) -> Result<(), serde_json::Error> {
        let Some(exts) = hugr.get_metadata(hugr.module_root(), USED_EXTENSIONS_KEY) else {
            return Ok(()); // No used extensions metadata, nothing to check
        };
        let used_exts: Vec<ExtensionDesc> = serde_json::from_value(exts.clone())?;

        self.set_used_extensions_generator(used_exts);
        Ok(())
    }

    /// Loads the resolved extensions used in the module from the HUGR.
    pub(crate) fn load_used_extensions_resolved(&mut self, hugr: &impl HugrView) {
        self.set_used_extensions_resolved(
            hugr.extensions()
                .iter()
                .map(|ext| ExtensionDesc::new(&ext.name, ext.version.clone())),
        )
    }

    /// Loads the public symbols defined in the module from the HUGR.
    pub(crate) fn load_public_symbols(&mut self, hugr: &impl HugrView) {
        let symbols = hugr
            .children(hugr.module_root())
            .filter_map(|n| match hugr.get_optype(n) {
                OpType::FuncDecl(decl) if *decl.visibility() == crate::Visibility::Public => {
                    Some(func_symbol(decl.func_name(), decl.signature()))
                }
                OpType::FuncDefn(defn) if *defn.visibility() == crate::Visibility::Public => {
                    Some(func_symbol(defn.func_name(), defn.signature()))
                }
                _ => None,
            });

        self.set_public_symbols(symbols);
    }

    /// Loads the entrypoint of the module from the HUGR.
    pub(crate) fn load_entrypoint(&mut self, hugr: &impl HugrView<Node = Node>) {
        let node = hugr.entrypoint();
        self.set_entrypoint(node, hugr.get_optype(node).clone());
    }

    /// Loads the number of nodes in the module from the HUGR.
    pub(crate) fn load_num_nodes(&mut self, hugr: &impl HugrView) {
        self.set_num_nodes(hugr.num_nodes());
    }

    /// Loads full description of the module from the HUGR.
    pub(crate) fn load_from_hugr(&mut self, hugr: &impl HugrView<Node = Node>) {
        self.load_num_nodes(hugr);
        self.load_entrypoint(hugr);
        self.load_generator(hugr);
        self.load_used_extensions_resolved(hugr);
        self.load_public_symbols(hugr);
        // invalid used extensions metadata is ignored here, treated as not present
        self.load_used_extensions_generator(hugr).ok();
    }

    /// Wraps a value with this module description.
    pub fn wrap<T>(self, val: T) -> Described<T, Self>
    where
        Self: Sized,
    {
        wrapper::Described::new(val, self)
    }
}

fn func_symbol(name: &str, signature: &crate::types::PolyFuncType) -> String {
    format!("{name}: {}", signature)
}
impl<H: HugrView<Node = Node>> From<&H> for ModuleDesc {
    fn from(hugr: &H) -> Self {
        let mut desc = ModuleDesc::default();
        desc.load_from_hugr(hugr);
        desc
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::{fixture, rstest};
    use semver::Version;

    #[fixture]
    fn empty_package_desc() -> PackageDesc {
        PackageDesc::default()
    }

    #[fixture]
    fn empty_module_desc() -> ModuleDesc {
        ModuleDesc::default()
    }

    #[fixture]
    fn test_extension() -> ExtensionDesc {
        ExtensionDesc::new("test_ext", Version::new(1, 0, 0))
    }

    #[rstest]
    fn test_package_desc_new() {
        let header = EnvelopeHeader::default();
        let package = PackageDesc::new(header);
        assert_eq!(package.header(), header);
        assert_eq!(package.n_modules(), 0);
        assert_eq!(package.n_packaged_extensions(), 0);
    }

    #[rstest]
    fn test_package_desc_set_n_modules(mut empty_package_desc: PackageDesc) {
        empty_package_desc.set_n_modules(5);
        assert_eq!(empty_package_desc.n_modules(), 5);
    }

    #[rstest]
    fn test_package_desc_set_module(
        mut empty_package_desc: PackageDesc,
        empty_module_desc: ModuleDesc,
    ) {
        empty_package_desc.set_module(0, empty_module_desc.clone());
        assert_eq!(
            empty_package_desc.modules().next().unwrap().as_ref(),
            Some(&empty_module_desc)
        );
    }

    #[rstest]
    fn test_package_desc_set_packaged_extension(
        mut empty_package_desc: PackageDesc,
        test_extension: ExtensionDesc,
    ) {
        empty_package_desc.set_packaged_extension(0, test_extension.clone());
        assert_eq!(
            empty_package_desc
                .packaged_extensions()
                .next()
                .unwrap()
                .as_ref(),
            Some(&test_extension)
        );
    }

    #[rstest]
    fn test_package_desc_generator(mut empty_package_desc: PackageDesc) {
        let mut module = ModuleDesc::default();
        module.set_generator("test_generator");
        empty_package_desc.set_module(0, module);
        assert_eq!(
            empty_package_desc.generator(),
            Some("test_generator".to_string())
        );
    }

    #[rstest]
    fn test_module_desc_set_num_nodes(mut empty_module_desc: ModuleDesc) {
        empty_module_desc.set_num_nodes(10);
        assert_eq!(empty_module_desc.num_nodes, Some(10));
    }

    #[rstest]
    fn test_module_desc_set_entrypoint(mut empty_module_desc: ModuleDesc) {
        let node = Node::from(portgraph::NodeIndex::new(0));
        let optype: OpType = crate::ops::DFG {
            signature: Default::default(),
        }
        .into();
        empty_module_desc.set_entrypoint(node, optype.clone());
        assert_eq!(empty_module_desc.entrypoint.as_ref().unwrap().node, node);
        assert_eq!(
            empty_module_desc.entrypoint.as_ref().unwrap().optype,
            optype
        );
    }

    #[rstest]
    #[case("test_generator", Some("test_generator".to_string()))]
    #[case("", None)]
    fn test_module_desc_generator(#[case] input: &str, #[case] expected: Option<String>) {
        let mut module = ModuleDesc::default();
        if !input.is_empty() {
            module.set_generator(input);
        }
        assert_eq!(module.generator, expected);
    }

    #[test]
    fn test_extension_desc_new() {
        let name = "test_extension";
        let version = Version::new(1, 0, 0);
        let extension = ExtensionDesc::new(name, version.clone());
        assert_eq!(extension.name, name);
        assert_eq!(extension.version, version);
    }

    #[rstest]
    fn test_package_desc_n_packaged_extensions(
        mut empty_package_desc: PackageDesc,
        test_extension: ExtensionDesc,
    ) {
        assert_eq!(empty_package_desc.n_packaged_extensions(), 0);

        empty_package_desc.set_packaged_extension(0, test_extension);
        assert_eq!(empty_package_desc.n_packaged_extensions(), 1);
    }

    #[rstest]
    fn test_package_desc_modules_iterator(
        mut empty_package_desc: PackageDesc,
        empty_module_desc: ModuleDesc,
    ) {
        empty_package_desc.set_module(0, empty_module_desc.clone());

        let modules: Vec<_> = empty_package_desc.modules().collect();
        assert_eq!(modules.len(), 1);
        assert_eq!(modules[0].as_ref(), Some(&empty_module_desc));
    }

    #[rstest]
    fn test_package_desc_packaged_extensions_iterator(
        mut empty_package_desc: PackageDesc,
        test_extension: ExtensionDesc,
    ) {
        empty_package_desc.set_packaged_extension(0, test_extension.clone());

        let extensions: Vec<_> = empty_package_desc.packaged_extensions().collect();
        assert_eq!(extensions.len(), 1);
        assert_eq!(extensions[0].as_ref(), Some(&test_extension));
    }

    #[rstest]
    fn test_module_desc_set_used_extensions_generator(
        mut empty_module_desc: ModuleDesc,
        test_extension: ExtensionDesc,
    ) {
        empty_module_desc.set_used_extensions_generator(vec![test_extension.clone()]);

        assert_eq!(
            empty_module_desc
                .used_extensions_generator
                .as_ref()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            empty_module_desc
                .used_extensions_generator
                .as_ref()
                .unwrap()[0],
            test_extension
        );
    }

    #[rstest]
    fn test_module_desc_extend_used_extensions_metadata(mut empty_module_desc: ModuleDesc) {
        let extension1 = ExtensionDesc::new("test_ext1", Version::new(1, 0, 0));
        let extension2 = ExtensionDesc::new("test_ext2", Version::new(2, 0, 0));

        empty_module_desc.set_used_extensions_generator(vec![extension1.clone()]);
        empty_module_desc.extend_used_extensions_metadata(vec![extension2.clone()]);

        let extensions = empty_module_desc
            .used_extensions_generator
            .as_ref()
            .unwrap();
        assert_eq!(extensions.len(), 2);
        assert!(extensions.contains(&extension1));
        assert!(extensions.contains(&extension2));
    }

    #[rstest]
    fn test_module_desc_set_public_symbols(mut empty_module_desc: ModuleDesc) {
        let symbols = vec!["symbol1".to_string(), "symbol2".to_string()];
        empty_module_desc.set_public_symbols(symbols.clone());

        assert_eq!(empty_module_desc.public_symbols.as_ref().unwrap().len(), 2);
        assert_eq!(empty_module_desc.public_symbols.as_ref().unwrap(), &symbols);
    }

    #[rstest]
    fn test_module_desc_extend_public_symbols(mut empty_module_desc: ModuleDesc) {
        let symbols1 = vec!["symbol1".to_string()];
        let symbols2 = vec!["symbol2".to_string()];

        empty_module_desc.set_public_symbols(symbols1.clone());
        empty_module_desc.extend_public_symbols(symbols2.clone());

        let symbols = empty_module_desc.public_symbols.as_ref().unwrap();
        assert_eq!(symbols.len(), 2);
        assert!(symbols.contains(&"symbol1".to_string()));
        assert!(symbols.contains(&"symbol2".to_string()));
    }
}
