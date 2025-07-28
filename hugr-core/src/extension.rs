//! Extensions
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use itertools::Itertools;
use resolution::{ExtensionResolutionError, WeakExtensionRegistry};
pub use semver::Version;
use serde::{Deserialize, Deserializer, Serialize};
use std::cell::UnsafeCell;
use std::collections::btree_map;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::{io, mem};

use derive_more::Display;
use thiserror::Error;

use crate::hugr::IdentList;
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::ops::{OpName, OpNameRef};
use crate::types::RowVariable;
use crate::types::type_param::{TermTypeError, TypeArg, TypeParam};
use crate::types::{CustomType, TypeBound, TypeName};
use crate::types::{Signature, TypeNameRef};

mod const_fold;
mod op_def;
pub mod prelude;
pub mod resolution;
pub mod simple_op;
mod type_def;

pub use const_fold::{ConstFold, ConstFoldResult, Folder, fold_out_row};
pub use op_def::{
    CustomSignatureFunc, CustomValidator, LowerFunc, OpDef, SignatureFromArgs, SignatureFunc,
    ValidateJustArgs, ValidateTypeArgs, deserialize_lower_funcs,
};
pub use prelude::{PRELUDE, PRELUDE_REGISTRY};
pub use type_def::{TypeDef, TypeDefBound};

#[cfg(feature = "declarative")]
pub mod declarative;

/// Extension Registries store extensions to be looked up e.g. during validation.
#[derive(Debug, Display, Default)]
#[display("ExtensionRegistry[{}]", exts.keys().join(", "))]
pub struct ExtensionRegistry {
    /// The extensions in the registry.
    exts: BTreeMap<ExtensionId, Arc<Extension>>,
    /// A flag indicating whether the current set of extensions has been
    /// validated.
    ///
    /// This is used to avoid re-validating the extensions every time the
    /// registry is validated, and is set to `false` whenever a new extension is
    /// added.
    valid: AtomicBool,
}

impl PartialEq for ExtensionRegistry {
    fn eq(&self, other: &Self) -> bool {
        self.exts == other.exts
    }
}

impl Clone for ExtensionRegistry {
    fn clone(&self) -> Self {
        Self {
            exts: self.exts.clone(),
            valid: self.valid.load(Ordering::Relaxed).into(),
        }
    }
}

impl ExtensionRegistry {
    /// Create a new empty extension registry.
    pub fn new(extensions: impl IntoIterator<Item = Arc<Extension>>) -> Self {
        let mut res = Self::default();
        for ext in extensions {
            res.register_updated(ext);
        }
        res
    }

    /// Load an `ExtensionRegistry` serialized as json.
    ///
    /// After deserialization, updates all the internal `Weak<Extension>`
    /// references to point to the newly created [`Arc`]s in the registry,
    /// or extensions in the `additional_extensions` parameter.
    pub fn load_json(
        reader: impl io::Read,
        other_extensions: &ExtensionRegistry,
    ) -> Result<Self, ExtensionRegistryLoadError> {
        let extensions: Vec<Extension> = serde_json::from_reader(reader)?;
        // After deserialization, we need to update all the internal
        // `Weak<Extension>` references.
        Ok(ExtensionRegistry::new_with_extension_resolution(
            extensions,
            &other_extensions.into(),
        )?)
    }

    /// Gets the Extension with the given name
    pub fn get(&self, name: &str) -> Option<&Arc<Extension>> {
        self.exts.get(name)
    }

    /// Returns `true` if the registry contains an extension with the given name.
    pub fn contains(&self, name: &str) -> bool {
        self.exts.contains_key(name)
    }

    /// Validate the set of extensions.
    pub fn validate(&self) -> Result<(), ExtensionRegistryError> {
        if self.valid.load(Ordering::Relaxed) {
            return Ok(());
        }
        for ext in self.exts.values() {
            ext.validate()
                .map_err(|e| ExtensionRegistryError::InvalidSignature(ext.name().clone(), e))?;
        }
        self.valid.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// Registers a new extension to the registry.
    ///
    /// Returns a reference to the registered extension if successful.
    pub fn register(
        &mut self,
        extension: impl Into<Arc<Extension>>,
    ) -> Result<(), ExtensionRegistryError> {
        let extension = extension.into();
        match self.exts.entry(extension.name().clone()) {
            btree_map::Entry::Occupied(prev) => Err(ExtensionRegistryError::AlreadyRegistered(
                extension.name().clone(),
                Box::new(prev.get().version().clone()),
                Box::new(extension.version().clone()),
            )),
            btree_map::Entry::Vacant(ve) => {
                ve.insert(extension);
                // Clear the valid flag so that the registry is re-validated.
                self.valid.store(false, Ordering::Relaxed);

                Ok(())
            }
        }
    }

    /// Registers a new extension to the registry, keeping the one most up to
    /// date if the extension already exists.
    ///
    /// If extension IDs match, the extension with the higher version is kept.
    /// If versions match, the original extension is kept. Returns a reference
    /// to the registered extension if successful.
    ///
    /// Takes an Arc to the extension. To avoid cloning Arcs unless necessary,
    /// see [`ExtensionRegistry::register_updated_ref`].
    pub fn register_updated(&mut self, extension: impl Into<Arc<Extension>>) {
        let extension = extension.into();
        match self.exts.entry(extension.name().clone()) {
            btree_map::Entry::Occupied(mut prev) => {
                if prev.get().version() < extension.version() {
                    *prev.get_mut() = extension;
                }
            }
            btree_map::Entry::Vacant(ve) => {
                ve.insert(extension);
            }
        }
        // Clear the valid flag so that the registry is re-validated.
        self.valid.store(false, Ordering::Relaxed);
    }

    /// Registers a new extension to the registry, keeping the one most up to
    /// date if the extension already exists.
    ///
    /// If extension IDs match, the extension with the higher version is kept.
    /// If versions match, the original extension is kept. Returns a reference
    /// to the registered extension if successful.
    ///
    /// Clones the Arc only when required. For no-cloning version see
    /// [`ExtensionRegistry::register_updated`].
    pub fn register_updated_ref(&mut self, extension: &Arc<Extension>) {
        match self.exts.entry(extension.name().clone()) {
            btree_map::Entry::Occupied(mut prev) => {
                if prev.get().version() < extension.version() {
                    *prev.get_mut() = extension.clone();
                }
            }
            btree_map::Entry::Vacant(ve) => {
                ve.insert(extension.clone());
            }
        }
        // Clear the valid flag so that the registry is re-validated.
        self.valid.store(false, Ordering::Relaxed);
    }

    /// Returns the number of extensions in the registry.
    pub fn len(&self) -> usize {
        self.exts.len()
    }

    /// Returns `true` if the registry contains no extensions.
    pub fn is_empty(&self) -> bool {
        self.exts.is_empty()
    }

    /// Returns an iterator over the extensions in the registry.
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.exts.values()
    }

    /// Returns an iterator over the extensions ids in the registry.
    pub fn ids(&self) -> impl Iterator<Item = &ExtensionId> {
        self.exts.keys()
    }

    /// Delete an extension from the registry and return it if it was present.
    pub fn remove_extension(&mut self, name: &ExtensionId) -> Option<Arc<Extension>> {
        // Clear the valid flag so that the registry is re-validated.
        self.valid.store(false, Ordering::Relaxed);

        self.exts.remove(name)
    }

    /// Constructs a new `ExtensionRegistry` from a list of [`Extension`]s while
    /// giving you a [`WeakExtensionRegistry`] to the allocation. This allows
    /// you to add [`Weak`] self-references to the [`Extension`]s while
    /// constructing them, before wrapping them in [`Arc`]s.
    ///
    /// This is similar to [`Arc::new_cyclic`], but for `ExtensionRegistries`.
    ///
    /// Calling [`Weak::upgrade`] on a weak reference in the
    /// [`WeakExtensionRegistry`] inside your closure will return an extension
    /// with no internal (op / type / value) definitions.
    //
    // It may be possible to implement this safely using `Arc::new_cyclic`
    // directly, but the callback type does not allow for returning extra
    // data so it seems unlikely.
    pub fn new_cyclic<F, E>(
        extensions: impl IntoIterator<Item = Extension>,
        init: F,
    ) -> Result<Self, E>
    where
        F: FnOnce(Vec<Extension>, &WeakExtensionRegistry) -> Result<Vec<Extension>, E>,
    {
        let extensions = extensions.into_iter().collect_vec();

        // Unsafe internally-mutable wrapper around an extension. Important:
        // `repr(transparent)` ensures the layout is identical to `Extension`,
        // so it can be safely transmuted.
        #[repr(transparent)]
        struct ExtensionCell {
            ext: UnsafeCell<Extension>,
        }

        // Create the arcs with internal mutability, and collect weak references
        // over immutable references.
        //
        // This is safe as long as the cell mutation happens when we can guarantee
        // that the weak references are not used.
        let (arcs, weaks): (Vec<Arc<ExtensionCell>>, Vec<Weak<Extension>>) = extensions
            .iter()
            .map(|ext| {
                // Create a new arc with an empty extension sharing the name and version of the original,
                // but with no internal definitions.
                //
                // `UnsafeCell` is not sync, but we are not writing to it while the weak references are
                // being used.
                #[allow(clippy::arc_with_non_send_sync)]
                let arc = Arc::new(ExtensionCell {
                    ext: UnsafeCell::new(Extension::new(ext.name().clone(), ext.version().clone())),
                });

                // SAFETY: `ExtensionCell` is `repr(transparent)`, so it has the same layout as `Extension`.
                let weak_arc: Weak<Extension> = unsafe { mem::transmute(Arc::downgrade(&arc)) };
                (arc, weak_arc)
            })
            .unzip();

        let mut weak_registry = WeakExtensionRegistry::default();
        for (ext, weak) in extensions.iter().zip(weaks) {
            weak_registry.register(ext.name().clone(), weak);
        }

        // Actual initialization here
        // Upgrading the weak references at any point here will access the empty extensions in the arcs.
        let extensions = init(extensions, &weak_registry)?;

        // We're done.
        let arcs: Vec<Arc<Extension>> = arcs
            .into_iter()
            .zip(extensions)
            .map(|(arc, ext)| {
                // Replace the dummy extensions with the updated ones.
                // SAFETY: The cell is only mutated when the weak references are not used.
                unsafe { *arc.ext.get() = ext };
                // Pretend the UnsafeCells never existed.
                // SAFETY: `ExtensionCell` is `repr(transparent)`, so it has the same layout as `Extension`.
                unsafe { mem::transmute::<Arc<ExtensionCell>, Arc<Extension>>(arc) }
            })
            .collect();
        Ok(ExtensionRegistry::new(arcs))
    }
}

impl IntoIterator for ExtensionRegistry {
    type Item = Arc<Extension>;

    type IntoIter = std::collections::btree_map::IntoValues<ExtensionId, Arc<Extension>>;

    fn into_iter(self) -> Self::IntoIter {
        self.exts.into_values()
    }
}

impl<'a> IntoIterator for &'a ExtensionRegistry {
    type Item = &'a Arc<Extension>;

    type IntoIter = std::collections::btree_map::Values<'a, ExtensionId, Arc<Extension>>;

    fn into_iter(self) -> Self::IntoIter {
        self.exts.values()
    }
}

impl<'a> Extend<&'a Arc<Extension>> for ExtensionRegistry {
    fn extend<T: IntoIterator<Item = &'a Arc<Extension>>>(&mut self, iter: T) {
        for ext in iter {
            self.register_updated_ref(ext);
        }
    }
}

impl Extend<Arc<Extension>> for ExtensionRegistry {
    fn extend<T: IntoIterator<Item = Arc<Extension>>>(&mut self, iter: T) {
        for ext in iter {
            self.register_updated(ext);
        }
    }
}

/// Encode/decode `ExtensionRegistry` as a list of extensions.
///
/// Any `Weak<Extension>` references inside the registry will be left unresolved.
/// Prefer using [`ExtensionRegistry::load_json`] when deserializing.
impl<'de> Deserialize<'de> for ExtensionRegistry {
    fn deserialize<D>(deserializer: D) -> Result<ExtensionRegistry, D::Error>
    where
        D: Deserializer<'de>,
    {
        let extensions: Vec<Arc<Extension>> = Vec::deserialize(deserializer)?;
        Ok(ExtensionRegistry::new(extensions))
    }
}

impl Serialize for ExtensionRegistry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let extensions: Vec<Arc<Extension>> = self.exts.values().cloned().collect();
        extensions.serialize(serializer)
    }
}

/// An Extension Registry containing no extensions.
pub static EMPTY_REG: ExtensionRegistry = ExtensionRegistry {
    exts: BTreeMap::new(),
    valid: AtomicBool::new(true),
};

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum SignatureError {
    /// Name mismatch
    #[error("Definition name ({0}) and instantiation name ({1}) do not match.")]
    NameMismatch(TypeName, TypeName),
    /// Extension mismatch
    #[error("Definition extension ({0}) and instantiation extension ({1}) do not match.")]
    ExtensionMismatch(ExtensionId, ExtensionId),
    /// When the type arguments of the node did not match the params declared by the `OpDef`
    #[error("Type arguments of node did not match params declared by definition: {0}")]
    TypeArgMismatch(#[from] TermTypeError),
    /// Invalid type arguments
    #[error("Invalid type arguments for operation")]
    InvalidTypeArgs,
    /// The weak [`Extension`] reference for a custom type has been dropped.
    #[error(
        "Type '{typ}' is defined in extension '{missing}', but the extension reference has been dropped."
    )]
    MissingTypeExtension { typ: TypeName, missing: ExtensionId },
    /// The Extension was found in the registry, but did not contain the Type(Def) referenced in the Signature
    #[error("Extension '{exn}' did not contain expected TypeDef '{typ}'")]
    ExtensionTypeNotFound { exn: ExtensionId, typ: TypeName },
    /// The bound recorded for a `CustomType` doesn't match what the `TypeDef` would compute
    #[error("Bound on CustomType ({actual}) did not match TypeDef ({expected})")]
    WrongBound {
        actual: TypeBound,
        expected: TypeBound,
    },
    /// A Type Variable's cache of its declared kind is incorrect
    #[error("Type Variable claims to be {cached} but actual declaration {actual}")]
    TypeVarDoesNotMatchDeclaration {
        actual: Box<TypeParam>,
        cached: Box<TypeParam>,
    },
    /// A type variable that was used has not been declared
    #[error("Type variable {idx} was not declared ({num_decls} in scope)")]
    FreeTypeVar { idx: usize, num_decls: usize },
    /// A row variable was found outside of a variable-length row
    #[error("Expected a single type, but found row variable {var}")]
    RowVarWhereTypeExpected { var: RowVariable },
    /// The result of the type application stored in a [Call]
    /// is not what we get by applying the type-args to the polymorphic function
    ///
    /// [Call]: crate::ops::dataflow::Call
    #[error(
        "Incorrect result of type application in Call - cached {cached} but expected {expected}"
    )]
    CallIncorrectlyAppliesType {
        cached: Box<Signature>,
        expected: Box<Signature>,
    },
    /// The result of the type application stored in a [`LoadFunction`]
    /// is not what we get by applying the type-args to the polymorphic function
    ///
    /// [`LoadFunction`]: crate::ops::dataflow::LoadFunction
    #[error(
        "Incorrect result of type application in LoadFunction - cached {cached} but expected {expected}"
    )]
    LoadFunctionIncorrectlyAppliesType {
        cached: Box<Signature>,
        expected: Box<Signature>,
    },

    /// Extension declaration specifies a binary compute signature function, but none
    /// was loaded.
    #[error("Binary compute signature function not loaded.")]
    MissingComputeFunc,

    /// Extension declaration specifies a binary compute signature function, but none
    /// was loaded.
    #[error("Binary validate signature function not loaded.")]
    MissingValidateFunc,
}

/// Concrete instantiations of types and operations defined in extensions.
trait CustomConcrete {
    /// The identifier type for the concrete object.
    type Identifier;
    /// A generic identifier to the element.
    ///
    /// This may either refer to a [`TypeName`] or an [`OpName`].
    fn def_name(&self) -> &Self::Identifier;
    /// The concrete type arguments for the instantiation.
    fn type_args(&self) -> &[TypeArg];
    /// Extension required by the instantiation.
    fn parent_extension(&self) -> &ExtensionId;
}

impl CustomConcrete for OpaqueOp {
    type Identifier = OpName;

    fn def_name(&self) -> &Self::Identifier {
        self.unqualified_id()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_extension(&self) -> &ExtensionId {
        self.extension()
    }
}

impl CustomConcrete for CustomType {
    type Identifier = TypeName;

    fn def_name(&self) -> &TypeName {
        // Casts the `TypeName` to a generic string.
        self.name()
    }

    fn type_args(&self) -> &[TypeArg] {
        self.args()
    }

    fn parent_extension(&self) -> &ExtensionId {
        self.extension()
    }
}

/// A unique identifier for a extension.
///
/// The actual [`Extension`] is stored externally.
pub type ExtensionId = IdentList;

/// A extension is a set of capabilities required to execute a graph.
///
/// These are normally defined once and shared across multiple graphs and
/// operations wrapped in [`Arc`]s inside [`ExtensionRegistry`].
///
/// # Example
///
/// The following example demonstrates how to define a new extension with a
/// custom operation and a custom type.
///
/// When using `arc`s, the extension can only be modified at creation time. The
/// defined operations and types keep a [`Weak`] reference to their extension. We provide a
/// helper method [`Extension::new_arc`] to aid their definition.
///
/// ```
/// # use hugr_core::types::Signature;
/// # use hugr_core::extension::{Extension, ExtensionId, Version};
/// # use hugr_core::extension::{TypeDefBound};
/// Extension::new_arc(
///     ExtensionId::new_unchecked("my.extension"),
///     Version::new(0, 1, 0),
///     |ext, extension_ref| {
///         // Add a custom type definition
///         ext.add_type(
///             "MyType".into(),
///             vec![], // No type parameters
///             "Some type".into(),
///             TypeDefBound::any(),
///             extension_ref,
///         );
///         // Add a custom operation
///         ext.add_op(
///             "MyOp".into(),
///             "Some operation".into(),
///             Signature::new_endo(vec![]),
///             extension_ref,
///         );
///     },
/// );
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Extension {
    /// Extension version, follows semver.
    pub version: Version,
    /// Unique identifier for the extension.
    pub name: ExtensionId,
    /// Types defined by this extension.
    types: BTreeMap<TypeName, TypeDef>,
    /// Operation declarations with serializable definitions.
    // Note: serde will serialize this because we configure with `features=["rc"]`.
    // That will clone anything that has multiple references, but each
    // OpDef should appear exactly once in this map (keyed by its name),
    // and the other references to the OpDef are from ExternalOp's in the Hugr
    // (which are serialized as OpaqueOp's i.e. Strings).
    operations: BTreeMap<OpName, Arc<op_def::OpDef>>,
}

impl Extension {
    /// Creates a new extension with the given name.
    ///
    /// In most cases extensions are contained inside an [`Arc`] so that they
    /// can be shared across hugr instances and operation definitions.
    ///
    /// See [`Extension::new_arc`] for a more ergonomic way to create boxed
    /// extensions.
    #[must_use]
    pub fn new(name: ExtensionId, version: Version) -> Self {
        Self {
            name,
            version,
            types: Default::default(),
            operations: Default::default(),
        }
    }

    /// Creates a new extension wrapped in an [`Arc`].
    ///
    /// The closure lets us use a weak reference to the arc while the extension
    /// is being built. This is necessary for calling [`Extension::add_op`] and
    /// [`Extension::add_type`].
    pub fn new_arc(
        name: ExtensionId,
        version: Version,
        init: impl FnOnce(&mut Extension, &Weak<Extension>),
    ) -> Arc<Self> {
        Arc::new_cyclic(|extension_ref| {
            let mut ext = Self::new(name, version);
            init(&mut ext, extension_ref);
            ext
        })
    }

    /// Creates a new extension wrapped in an [`Arc`], using a fallible
    /// initialization function.
    ///
    /// The closure lets us use a weak reference to the arc while the extension
    /// is being built. This is necessary for calling [`Extension::add_op`] and
    /// [`Extension::add_type`].
    pub fn try_new_arc<E>(
        name: ExtensionId,
        version: Version,
        init: impl FnOnce(&mut Extension, &Weak<Extension>) -> Result<(), E>,
    ) -> Result<Arc<Self>, E> {
        // Annoying hack around not having `Arc::try_new_cyclic` that can return
        // a Result.
        // https://github.com/rust-lang/rust/issues/75861#issuecomment-980455381
        //
        // When there is an error, we store it in `error` and return it at the
        // end instead of the partially-initialized extension.
        let mut error = None;
        let ext = Arc::new_cyclic(|extension_ref| {
            let mut ext = Self::new(name, version);
            match init(&mut ext, extension_ref) {
                Ok(()) => ext,
                Err(e) => {
                    error = Some(e);
                    ext
                }
            }
        });
        match error {
            Some(e) => Err(e),
            None => Ok(ext),
        }
    }

    /// Allows read-only access to the operations in this Extension
    #[must_use]
    pub fn get_op(&self, name: &OpNameRef) -> Option<&Arc<op_def::OpDef>> {
        self.operations.get(name)
    }

    /// Allows read-only access to the types in this Extension
    #[must_use]
    pub fn get_type(&self, type_name: &TypeNameRef) -> Option<&type_def::TypeDef> {
        self.types.get(type_name)
    }

    /// Returns the name of the extension.
    #[must_use]
    pub fn name(&self) -> &ExtensionId {
        &self.name
    }

    /// Returns the version of the extension.
    #[must_use]
    pub fn version(&self) -> &Version {
        &self.version
    }

    /// Iterator over the operations of this [`Extension`].
    pub fn operations(&self) -> impl Iterator<Item = (&OpName, &Arc<OpDef>)> {
        self.operations.iter()
    }

    /// Iterator over the types of this [`Extension`].
    pub fn types(&self) -> impl Iterator<Item = (&TypeName, &TypeDef)> {
        self.types.iter()
    }

    /// Instantiate an [`ExtensionOp`] which references an [`OpDef`] in this extension.
    pub fn instantiate_extension_op(
        &self,
        name: &OpNameRef,
        args: impl Into<Vec<TypeArg>>,
    ) -> Result<ExtensionOp, SignatureError> {
        let op_def = self.get_op(name).expect("Op not found.");
        ExtensionOp::new(op_def.clone(), args)
    }

    /// Validates the operation definitions in the register.
    fn validate(&self) -> Result<(), SignatureError> {
        // We should validate TypeParams of TypeDefs too - https://github.com/CQCL/hugr/issues/624
        for op_def in self.operations.values() {
            op_def.validate()?;
        }
        Ok(())
    }
}

impl PartialEq for Extension {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.version == other.version
    }
}

/// An error that can occur in defining an extension registry.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExtensionRegistryError {
    /// Extension already defined.
    #[error(
        "The registry already contains an extension with id {0} and version {1}. New extension has version {2}."
    )]
    AlreadyRegistered(ExtensionId, Box<Version>, Box<Version>),
    /// A registered extension has invalid signatures.
    #[error("The extension {0} contains an invalid signature, {1}.")]
    InvalidSignature(ExtensionId, #[source] SignatureError),
}

/// An error that can occur while loading an extension registry.
#[derive(Debug, Error)]
#[non_exhaustive]
#[error("Extension registry load error")]
pub enum ExtensionRegistryLoadError {
    /// Deserialization error.
    #[error(transparent)]
    SerdeError(#[from] serde_json::Error),
    /// Error when resolving internal extension references.
    #[error(transparent)]
    ExtensionResolutionError(Box<ExtensionResolutionError>),
}

impl From<ExtensionResolutionError> for ExtensionRegistryLoadError {
    fn from(error: ExtensionResolutionError) -> Self {
        Self::ExtensionResolutionError(Box::new(error))
    }
}

/// An error that can occur in building a new extension.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExtensionBuildError {
    /// Existing [`OpDef`]
    #[error("Extension already has an op called {0}.")]
    OpDefExists(OpName),
    /// Existing [`TypeDef`]
    #[error("Extension already has an type called {0}.")]
    TypeDefExists(TypeName),
}

/// A set of extensions identified by their unique [`ExtensionId`].
#[derive(
    Clone, Debug, Display, Default, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
#[display("[{}]", _0.iter().join(", "))]
pub struct ExtensionSet(BTreeSet<ExtensionId>);

impl ExtensionSet {
    /// Creates a new empty extension set.
    #[must_use]
    pub const fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Adds a extension to the set.
    pub fn insert(&mut self, extension: ExtensionId) {
        self.0.insert(extension.clone());
    }

    /// Returns `true` if the set contains the given extension.
    #[must_use]
    pub fn contains(&self, extension: &ExtensionId) -> bool {
        self.0.contains(extension)
    }

    /// Returns `true` if the set is a subset of `other`.
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }

    /// Returns `true` if the set is a superset of `other`.
    #[must_use]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.0.is_superset(&other.0)
    }

    /// Create a extension set with a single element.
    #[must_use]
    pub fn singleton(extension: ExtensionId) -> Self {
        let mut set = Self::new();
        set.insert(extension);
        set
    }

    /// Returns the union of two extension sets.
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        self.0.extend(other.0);
        self
    }

    /// Returns the union of an arbitrary collection of [`ExtensionSet`]s
    pub fn union_over(sets: impl IntoIterator<Item = Self>) -> Self {
        // `union` clones the receiver, which we do not need to do here
        let mut res = ExtensionSet::new();
        for s in sets {
            res.0.extend(s.0);
        }
        res
    }

    /// The things in other which are in not in self
    #[must_use]
    pub fn missing_from(&self, other: &Self) -> Self {
        ExtensionSet::from_iter(other.0.difference(&self.0).cloned())
    }

    /// Iterate over the contained `ExtensionIds`
    pub fn iter(&self) -> impl Iterator<Item = &ExtensionId> {
        self.0.iter()
    }

    /// True if this set contains no [`ExtensionId`]s
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<ExtensionId> for ExtensionSet {
    fn from(id: ExtensionId) -> Self {
        Self::singleton(id)
    }
}

impl IntoIterator for ExtensionSet {
    type Item = ExtensionId;
    type IntoIter = std::collections::btree_set::IntoIter<ExtensionId>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a ExtensionSet {
    type Item = &'a ExtensionId;
    type IntoIter = std::collections::btree_set::Iter<'a, ExtensionId>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromIterator<ExtensionId> for ExtensionSet {
    fn from_iter<I: IntoIterator<Item = ExtensionId>>(iter: I) -> Self {
        Self(BTreeSet::from_iter(iter))
    }
}

/// Extension tests.
#[cfg(test)]
pub mod test {
    // We re-export this here because mod op_def is private.
    pub use super::op_def::test::SimpleOpDef;

    use super::*;

    impl Extension {
        /// Create a new extension for testing, with a 0 version.
        pub(crate) fn new_test_arc(
            name: ExtensionId,
            init: impl FnOnce(&mut Extension, &Weak<Extension>),
        ) -> Arc<Self> {
            Self::new_arc(name, Version::new(0, 0, 0), init)
        }

        /// Create a new extension for testing, with a 0 version.
        pub(crate) fn try_new_test_arc(
            name: ExtensionId,
            init: impl FnOnce(
                &mut Extension,
                &Weak<Extension>,
            ) -> Result<(), Box<dyn std::error::Error>>,
        ) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
            Self::try_new_arc(name, Version::new(0, 0, 0), init)
        }
    }

    #[test]
    fn test_register_update() {
        // Two registers that should remain the same.
        // We use them to test both `register_updated` and `register_updated_ref`.
        let mut reg = ExtensionRegistry::default();
        let mut reg_ref = ExtensionRegistry::default();

        let ext_1_id = ExtensionId::new("ext1").unwrap();
        let ext_2_id = ExtensionId::new("ext2").unwrap();
        let ext1 = Arc::new(Extension::new(ext_1_id.clone(), Version::new(1, 0, 0)));
        let ext1_1 = Arc::new(Extension::new(ext_1_id.clone(), Version::new(1, 1, 0)));
        let ext1_2 = Arc::new(Extension::new(ext_1_id.clone(), Version::new(0, 2, 0)));
        let ext2 = Arc::new(Extension::new(ext_2_id, Version::new(1, 0, 0)));

        reg.register(ext1.clone()).unwrap();
        reg_ref.register(ext1.clone()).unwrap();
        assert_eq!(&reg, &reg_ref);

        // normal registration fails
        assert_eq!(
            reg.register(ext1_1.clone()),
            Err(ExtensionRegistryError::AlreadyRegistered(
                ext_1_id.clone(),
                Box::new(Version::new(1, 0, 0)),
                Box::new(Version::new(1, 1, 0))
            ))
        );

        // register with update works
        reg_ref.register_updated_ref(&ext1_1);
        reg.register_updated(ext1_1.clone());
        assert_eq!(reg.get("ext1").unwrap().version(), &Version::new(1, 1, 0));
        assert_eq!(&reg, &reg_ref);

        // register with lower version does not change version
        reg_ref.register_updated_ref(&ext1_2);
        reg.register_updated(ext1_2.clone());
        assert_eq!(reg.get("ext1").unwrap().version(), &Version::new(1, 1, 0));
        assert_eq!(&reg, &reg_ref);

        reg.register(ext2.clone()).unwrap();
        assert_eq!(reg.get("ext2").unwrap().version(), &Version::new(1, 0, 0));
        assert_eq!(reg.len(), 2);

        assert!(reg.remove_extension(&ext_1_id).unwrap().version() == &Version::new(1, 1, 0));
        assert_eq!(reg.len(), 1);
    }

    mod proptest {

        use ::proptest::{collection::hash_set, prelude::*};

        use super::super::{ExtensionId, ExtensionSet};

        impl Arbitrary for ExtensionSet {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;

            fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
                hash_set(any::<ExtensionId>(), 0..3)
                    .prop_map(|extensions| extensions.into_iter().collect::<ExtensionSet>())
                    .boxed()
            }
        }
    }
}
