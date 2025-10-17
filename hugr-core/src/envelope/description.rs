//! Description of the contents of a HUGR envelope used for debugging and error reporting.
use semver::Version;

use crate::{HugrView, Node, envelope::EnvelopeHeader, ops::OpType};

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(transparent)]
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

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct PackageDesc {
    #[serde(serialize_with = "header_serialize")]
    pub(super) header: EnvelopeHeader,
    modules: PartialVec<ModuleDesc>,
    packaged_extensions: PartialVec<ExtensionDesc>,
}

fn header_serialize<S>(header: &EnvelopeHeader, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&header.to_string())
}

impl PackageDesc {
    pub(super) fn new(header: EnvelopeHeader) -> Self {
        Self {
            header,
            ..Default::default()
        }
    }
    pub(crate) fn set_n_modules(&mut self, n: usize) {
        self.modules.set_len(n);
    }

    pub fn header(&self) -> String {
        self.header.to_string()
    }
    pub fn n_modules(&self) -> usize {
        self.modules.len()
    }
    pub(crate) fn set_module(&mut self, index: usize, module: impl Into<ModuleDesc>) {
        self.modules.set_index(index, module.into());
    }
    pub(crate) fn set_packaged_extension(&mut self, index: usize, ext: impl Into<ExtensionDesc>) {
        self.packaged_extensions.set_index(index, ext.into());
    }
    pub(crate) fn set_n_packaged_extensions(&mut self, n: usize) {
        self.packaged_extensions.set_len(n);
    }
    pub fn n_packaged_extensions(&self) -> usize {
        self.packaged_extensions.len()
    }

    pub fn generator(&self) -> Option<String> {
        let generators: Vec<String> = self
            .modules
            .vec
            .iter()
            .flatten()
            .flat_map(|m| m.generator.clone())
            .collect();
        if generators.is_empty() {
            return None;
        }

        Some(generators.join(", "))
    }
    pub fn modules(&self) -> impl Iterator<Item = &Option<ModuleDesc>> {
        self.modules.vec.iter()
    }

    pub fn packaged_extensions(&self) -> impl Iterator<Item = &Option<ExtensionDesc>> {
        self.packaged_extensions.vec.iter()
    }
}

#[derive(derive_more::Display, Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[display("Extension {name} v{version}")]
pub struct ExtensionDesc {
    /// Name of the extension.
    pub name: String,
    /// Version of the extension.
    pub version: Version,
}

impl ExtensionDesc {
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
pub struct Entrypoint {
    pub node: Node,
    #[serde(serialize_with = "op_serialize")]
    pub optype: OpType,
}

impl Entrypoint {
    pub fn new(node: Node, optype: OpType) -> Self {
        Self { node, optype }
    }
}

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
pub struct ModuleDesc {
    /// Number of nodes in the module.
    pub num_nodes: Option<usize>,
    /// The entrypoint node and the corresponding operation type.
    pub entrypoint: Option<Entrypoint>,
    /// Extensions used in the module computed while resolving, expected to be a subset of `used_extensions_metadata`.
    pub used_extensions_resolved: Option<Vec<ExtensionDesc>>,
    /// Generator specified in the module metadata.
    pub generator: Option<String>,
    /// Generator specified used extensions in the module metadata.
    pub used_extensions_generator: Option<Vec<ExtensionDesc>>,
    /// Public symbols defined in the module.
    pub public_symbols: Option<Vec<String>>,
}

impl ModuleDesc {
    pub fn set_num_nodes(&mut self, num_nodes: usize) {
        self.num_nodes = Some(num_nodes);
    }
    pub fn set_entrypoint(&mut self, node: Node, optype: OpType) {
        self.entrypoint = Some(Entrypoint::new(node, optype));
    }
    pub fn set_generator(&mut self, generator: impl Into<String>) {
        self.generator = Some(generator.into());
    }
    pub fn set_used_extensions_generator(
        &mut self,
        used_extensions_metadata: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        self.used_extensions_generator = Some(used_extensions_metadata.into_iter().collect());
    }
    pub fn extend_used_extensions_metadata(
        &mut self,
        exts: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        extend_option_vec(&mut self.used_extensions_generator, exts);
    }
    pub fn set_used_extensions_resolved(
        &mut self,
        used_extensions_resolved: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        self.used_extensions_resolved = Some(used_extensions_resolved.into_iter().collect());
    }
    pub fn extend_used_extensions_resolved(
        &mut self,
        exts: impl IntoIterator<Item = ExtensionDesc>,
    ) {
        extend_option_vec(&mut self.used_extensions_resolved, exts);
    }
    pub fn set_public_symbols(&mut self, symbols: impl IntoIterator<Item = String>) {
        self.public_symbols = Some(symbols.into_iter().collect());
    }
    pub fn extend_public_symbols(&mut self, symbols: impl IntoIterator<Item = String>) {
        extend_option_vec(&mut self.public_symbols, symbols);
    }

    pub fn load_generator(&mut self, hugr: &impl HugrView) {
        if let Some(val) = hugr.get_metadata(hugr.module_root(), crate::envelope::GENERATOR_KEY) {
            self.set_generator(format_generator(val));
        }
    }

    pub fn load_used_extensions_generator(&mut self, hugr: &impl HugrView) {
        let Some(exts) = hugr.get_metadata(hugr.module_root(), USED_EXTENSIONS_KEY) else {
            return; // No used extensions metadata, nothing to check
        };
        let Some(used_exts): Option<Vec<ExtensionDesc>> = serde_json::from_value(exts.clone()).ok()
        else {
            // TODO don't fail silently
            return;
        };
        self.set_used_extensions_generator(used_exts);
    }

    pub fn load_used_extensions_resolved(&mut self, hugr: &impl HugrView) {
        self.set_used_extensions_resolved(
            hugr.extensions()
                .iter()
                .map(|ext| ExtensionDesc::new(&ext.name, ext.version.clone())),
        )
    }

    pub fn load_public_symbols(&mut self, hugr: &impl HugrView) {
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

    pub fn load_entrypoint(&mut self, hugr: &impl HugrView<Node = Node>) {
        let node = hugr.entrypoint();
        self.set_entrypoint(node, hugr.get_optype(node).clone());
    }

    pub fn load_num_nodes(&mut self, hugr: &impl HugrView) {
        self.set_num_nodes(hugr.num_nodes());
    }
    fn load_from_hugr(&mut self, hugr: &impl HugrView<Node = Node>) {
        self.load_num_nodes(hugr);
        self.load_entrypoint(hugr);
        self.load_generator(hugr);
        self.load_used_extensions_generator(hugr);
        self.load_used_extensions_resolved(hugr);
        self.load_public_symbols(hugr);
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

// TODO centralise all core metadata keys in one place.

/// Key used to store the name of the generator that produced the envelope.
pub const GENERATOR_KEY: &str = "core.generator";
/// Key used to store the list of used extensions in the metadata of a HUGR.
pub const USED_EXTENSIONS_KEY: &str = "core.used_extensions";

/// Format a generator value from the metadata.
pub(crate) fn format_generator(json_val: &serde_json::Value) -> String {
    match json_val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Object(obj) => {
            if let (Some(name), version) = (
                obj.get("name").and_then(|v| v.as_str()),
                obj.get("version").and_then(|v| v.as_str()),
            ) {
                if let Some(version) = version {
                    // Expected format: {"name": "generator", "version": "1.0.0"}
                    format!("{name}-v{version}")
                } else {
                    name.to_string()
                }
            } else {
                // just print the whole object as a string
                json_val.to_string()
            }
        }
        // Raw JSON string fallback
        _ => json_val.to_string(),
    }
}
