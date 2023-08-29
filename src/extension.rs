//! Extensions
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use smol_str::SmolStr;
use thiserror::Error;

use crate::ops;
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::types::type_param::{check_type_arg, TypeArgError};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::CustomType;

mod infer;
pub use infer::{infer_extensions, ExtensionSolution, InferExtensionError};

mod op_def;
pub use op_def::{CustomSignatureFunc, OpDef};
mod type_def;
pub use type_def::{TypeDef, TypeDefBound};
pub mod prelude;
pub mod validate;

pub use prelude::PRELUDE;

/// Extension Registries store extensions to be looked up e.g. during validation.
pub struct ExtensionRegistry(BTreeMap<SmolStr, Extension>);

impl ExtensionRegistry {
    /// Makes a new (empty) registry.
    pub const fn new() -> Self {
        Self(BTreeMap::new())
    }
}

/// An Extension Registry containing no extensions.
pub const EMPTY_REG: ExtensionRegistry = ExtensionRegistry::new();

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SignatureError {
    /// Name mismatch
    #[error("Definition name ({0}) and instantiation name ({1}) do not match.")]
    NameMismatch(SmolStr, SmolStr),
    /// Extension mismatch
    #[error("Definition extension ({0:?}) and instantiation extension ({1:?}) do not match.")]
    ExtensionMismatch(ExtensionId, ExtensionId),
    /// When the type arguments of the node did not match the params declared by the OpDef
    #[error("Type arguments of node did not match params declared by definition: {0}")]
    TypeArgMismatch(#[from] TypeArgError),
    /// Invalid type arguments
    #[error("Invalid type arguments for operation")]
    InvalidTypeArgs,
}

/// Concrete instantiations of types and operations defined in extensions.
trait CustomConcrete {
    fn def_name(&self) -> &SmolStr;
    fn type_args(&self) -> &[TypeArg];
    fn parent_extension(&self) -> &ExtensionId;
}

impl CustomConcrete for OpaqueOp {
    fn def_name(&self) -> &SmolStr {
        self.name()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_extension(&self) -> &ExtensionId {
        self.extension()
    }
}

impl CustomConcrete for CustomType {
    fn def_name(&self) -> &SmolStr {
        self.name()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_extension(&self) -> &ExtensionId {
        self.extension()
    }
}

/// Type-parametrised functionality shared between [`TypeDef`] and [`OpDef`].
trait TypeParametrised {
    /// The concrete object built by binding type arguments to parameters
    type Concrete: CustomConcrete;
    /// The extension-unique name.
    fn name(&self) -> &SmolStr;
    /// Type parameters.
    fn params(&self) -> &[TypeParam];
    /// The parent extension.
    fn extension(&self) -> &ExtensionId;
    /// Check provided type arguments are valid against parameters.
    fn check_args_impl(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        if args.len() != self.params().len() {
            return Err(SignatureError::TypeArgMismatch(
                TypeArgError::WrongNumberArgs(args.len(), self.params().len()),
            ));
        }
        for (a, p) in args.iter().zip(self.params().iter()) {
            check_type_arg(a, p).map_err(SignatureError::TypeArgMismatch)?;
        }
        Ok(())
    }

    /// Check custom instance is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    fn check_concrete_impl(&self, custom: &Self::Concrete) -> Result<(), SignatureError> {
        if self.extension() != custom.parent_extension() {
            return Err(SignatureError::ExtensionMismatch(
                self.extension().clone(),
                custom.parent_extension().clone(),
            ));
        }
        if self.name() != custom.def_name() {
            return Err(SignatureError::NameMismatch(
                self.name().clone(),
                custom.def_name().clone(),
            ));
        }

        self.check_args_impl(custom.type_args())?;

        Ok(())
    }
}

/// A constant value provided by a extension.
/// Must be an instance of a type available to the extension.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExtensionValue {
    extension: ExtensionId,
    name: SmolStr,
    typed_value: ops::Const,
}

impl ExtensionValue {
    /// Returns a reference to the typed value of this [`ExtensionValue`].
    pub fn typed_value(&self) -> &ops::Const {
        &self.typed_value
    }

    /// Returns a reference to the name of this [`ExtensionValue`].
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Returns a reference to the extension this [`ExtensionValue`] belongs to.
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }
}

/// A unique identifier for a extension.
///
/// The actual [`Extension`] is stored externally.
pub type ExtensionId = SmolStr;

/// A extension is a set of capabilities required to execute a graph.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Extension {
    /// Unique identifier for the extension.
    pub name: ExtensionId,
    /// Other extensions defining types used by this extension.
    /// That is, an upper-bound on the types that can be returned by
    /// computing the signature of any operation in this extension,
    /// for any possible [TypeArg].
    pub extension_reqs: ExtensionSet,
    /// Types defined by this extension.
    types: HashMap<SmolStr, TypeDef>,
    /// Static values defined by this extension.
    values: HashMap<SmolStr, ExtensionValue>,
    /// Operation declarations with serializable definitions.
    // Note: serde will serialize this because we configure with `features=["rc"]`.
    // That will clone anything that has multiple references, but each
    // OpDef should appear exactly once in this map (keyed by its name),
    // and the other references to the OpDef are from ExternalOp's in the Hugr
    // (which are serialized as OpaqueOp's i.e. Strings).
    operations: HashMap<SmolStr, Arc<op_def::OpDef>>,
}

impl Extension {
    /// Creates a new extension with the given name.
    pub fn new(name: ExtensionId) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    /// Creates a new extension with the given name and requirements.
    pub fn new_with_reqs(name: ExtensionId, extension_reqs: ExtensionSet) -> Self {
        Self {
            name,
            extension_reqs,
            ..Default::default()
        }
    }

    /// Allows read-only access to the operations in this Extension
    pub fn get_op(&self, op_name: &str) -> Option<&Arc<op_def::OpDef>> {
        self.operations.get(op_name)
    }

    /// Allows read-only access to the types in this Extension
    pub fn get_type(&self, type_name: &str) -> Option<&type_def::TypeDef> {
        self.types.get(type_name)
    }

    /// Allows read-only access to the values in this Extension
    pub fn get_value(&self, type_name: &str) -> Option<&ExtensionValue> {
        self.values.get(type_name)
    }

    /// Returns the name of the extension.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Iterator over the operations of this [`Extension`].
    pub fn operations(&self) -> impl Iterator<Item = (&SmolStr, &Arc<OpDef>)> {
        self.operations.iter()
    }

    /// Iterator over the types of this [`Extension`].
    pub fn types(&self) -> impl Iterator<Item = (&SmolStr, &TypeDef)> {
        self.types.iter()
    }

    /// Add a named static value to the extension.
    pub fn add_value(
        &mut self,
        name: impl Into<SmolStr>,
        typed_value: ops::Const,
    ) -> Result<&mut ExtensionValue, ExtensionBuildError> {
        let extension_value = ExtensionValue {
            extension: self.name().into(),
            name: name.into(),
            typed_value,
        };
        match self.values.entry(extension_value.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::OpDefExists(extension_value.name)),
            Entry::Vacant(ve) => Ok(ve.insert(extension_value)),
        }
    }

    /// Instantiate an [`ExtensionOp`] which references an [`OpDef`] in this extension.
    pub fn instantiate_extension_op(
        &self,
        op_name: &str,
        args: impl Into<Vec<TypeArg>>,
    ) -> Result<ExtensionOp, SignatureError> {
        let op_def = self.get_op(op_name).expect("Op not found.");
        ExtensionOp::new(op_def.clone(), args)
    }
}

impl PartialEq for Extension {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ExtensionBuildError {
    /// Existing [`OpDef`]
    #[error("Extension already has an op called {0}.")]
    OpDefExists(SmolStr),
    /// Existing [`TypeDef`]
    #[error("Extension already has an type called {0}.")]
    TypeDefExists(SmolStr),
}

/// A set of extensions identified by their unique [`ExtensionId`].
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSet(HashSet<ExtensionId>);

impl ExtensionSet {
    /// Creates a new empty extension set.
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    /// Creates a new extension set from some extensions.
    pub fn new_from_extensions(extensions: impl Into<HashSet<ExtensionId>>) -> Self {
        Self(extensions.into())
    }

    /// Adds a extension to the set.
    pub fn insert(&mut self, extension: &ExtensionId) {
        self.0.insert(extension.clone());
    }

    /// Returns `true` if the set contains the given extension.
    pub fn contains(&self, extension: &ExtensionId) -> bool {
        self.0.contains(extension)
    }

    /// Returns `true` if the set is a subset of `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns `true` if the set is a superset of `other`.
    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    /// Create a extension set with a single element.
    pub fn singleton(extension: &ExtensionId) -> Self {
        let mut set = Self::new();
        set.insert(extension);
        set
    }

    /// Returns the union of two extension sets.
    pub fn union(mut self, other: &Self) -> Self {
        self.0.extend(other.0.iter().cloned());
        self
    }

    /// The things in other which are in not in self
    pub fn missing_from(&self, other: &Self) -> Self {
        ExtensionSet(HashSet::from_iter(other.0.difference(&self.0).cloned()))
    }

    /// Iterate over the contained ExtensionIds
    pub fn iter(&self) -> impl Iterator<Item = &ExtensionId> {
        self.0.iter()
    }
}

impl Display for ExtensionSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl FromIterator<ExtensionId> for ExtensionSet {
    fn from_iter<I: IntoIterator<Item = ExtensionId>>(iter: I) -> Self {
        Self(HashSet::from_iter(iter))
    }
}
