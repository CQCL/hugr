//! Extensions
//!
//! TODO: YAML declaration and parsing. This should be similar to a plugin
//! system (outside the `types` module), which also parses nested [`OpDef`]s.

use std::collections::btree_map;
use std::collections::hash_map;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use thiserror::Error;

use crate::hugr::IdentList;
use crate::ops::constant::{ValueName, ValueNameRef};
use crate::ops::custom::{ExtensionOp, OpaqueOp};
use crate::ops::{self, OpName, OpNameRef};
use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
use crate::types::RowVariable;
use crate::types::{check_typevar_decl, CustomType, Substitution, TypeBound, TypeName};
use crate::types::{Signature, TypeNameRef};

mod op_def;
pub use op_def::{
    CustomSignatureFunc, CustomValidator, LowerFunc, OpDef, SignatureFromArgs, SignatureFunc,
    ValidateJustArgs, ValidateTypeArgs,
};
mod type_def;
pub use type_def::{TypeDef, TypeDefBound};
mod const_fold;
pub mod prelude;
pub mod simple_op;
pub use const_fold::{ConstFold, ConstFoldResult, Folder};
pub use prelude::{PRELUDE, PRELUDE_REGISTRY};

#[cfg(feature = "declarative")]
pub mod declarative;

/// Extension Registries store extensions to be looked up e.g. during validation.
#[derive(Clone, Debug)]
pub struct ExtensionRegistry(BTreeMap<ExtensionId, Extension>);

impl ExtensionRegistry {
    /// Gets the Extension with the given name
    pub fn get(&self, name: &str) -> Option<&Extension> {
        self.0.get(name)
    }

    /// Returns `true` if the registry contains an extension with the given name.
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// Makes a new ExtensionRegistry, validating all the extensions in it
    pub fn try_new(
        value: impl IntoIterator<Item = Extension>,
    ) -> Result<Self, ExtensionRegistryError> {
        let mut exts = BTreeMap::new();
        for ext in value.into_iter() {
            let prev = exts.insert(ext.name.clone(), ext);
            if let Some(prev) = prev {
                return Err(ExtensionRegistryError::AlreadyRegistered(
                    prev.name().clone(),
                ));
            };
        }
        // Note this potentially asks extensions to validate themselves against other extensions that
        // may *not* be valid themselves yet. It'd be better to order these respecting dependencies,
        // or at least to validate the types first - which we don't do at all yet:
        // TODO https://github.com/CQCL/hugr/issues/624. However, parametrized types could be
        // cyclically dependent, so there is no perfect solution, and this is at least simple.
        let res = ExtensionRegistry(exts);
        for ext in res.0.values() {
            ext.validate(&res)
                .map_err(|e| ExtensionRegistryError::InvalidSignature(ext.name().clone(), e))?;
        }
        Ok(res)
    }

    /// Registers a new extension to the registry.
    ///
    /// Returns a reference to the registered extension if successful.
    pub fn register(&mut self, extension: Extension) -> Result<&Extension, ExtensionRegistryError> {
        match self.0.entry(extension.name().clone()) {
            btree_map::Entry::Occupied(_) => Err(ExtensionRegistryError::AlreadyRegistered(
                extension.name().clone(),
            )),
            btree_map::Entry::Vacant(ve) => Ok(ve.insert(extension)),
        }
    }

    /// Returns the number of extensions in the registry.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the registry contains no extensions.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator over the extensions in the registry.
    pub fn iter(&self) -> impl Iterator<Item = (&ExtensionId, &Extension)> {
        self.0.iter()
    }
}

impl IntoIterator for ExtensionRegistry {
    type Item = (ExtensionId, Extension);

    type IntoIter = <BTreeMap<ExtensionId, Extension> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// An Extension Registry containing no extensions.
pub const EMPTY_REG: ExtensionRegistry = ExtensionRegistry(BTreeMap::new());

/// An error that can occur in computing the signature of a node.
/// TODO: decide on failure modes
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum SignatureError {
    /// Name mismatch
    #[error("Definition name ({0}) and instantiation name ({1}) do not match.")]
    NameMismatch(TypeName, TypeName),
    /// Extension mismatch
    #[error("Definition extension ({0:?}) and instantiation extension ({1:?}) do not match.")]
    ExtensionMismatch(ExtensionId, ExtensionId),
    /// When the type arguments of the node did not match the params declared by the OpDef
    #[error("Type arguments of node did not match params declared by definition: {0}")]
    TypeArgMismatch(#[from] TypeArgError),
    /// Invalid type arguments
    #[error("Invalid type arguments for operation")]
    InvalidTypeArgs,
    /// The Extension Registry did not contain an Extension referenced by the Signature
    #[error("Extension '{0}' not found")]
    ExtensionNotFound(ExtensionId),
    /// The Extension was found in the registry, but did not contain the Type(Def) referenced in the Signature
    #[error("Extension '{exn}' did not contain expected TypeDef '{typ}'")]
    ExtensionTypeNotFound { exn: ExtensionId, typ: TypeName },
    /// The bound recorded for a CustomType doesn't match what the TypeDef would compute
    #[error("Bound on CustomType ({actual}) did not match TypeDef ({expected})")]
    WrongBound {
        actual: TypeBound,
        expected: TypeBound,
    },
    /// A Type Variable's cache of its declared kind is incorrect
    #[error("Type Variable claims to be {cached:?} but actual declaration {actual:?}")]
    TypeVarDoesNotMatchDeclaration {
        actual: TypeParam,
        cached: TypeParam,
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
        cached: Signature,
        expected: Signature,
    },
    /// The result of the type application stored in a [LoadFunction]
    /// is not what we get by applying the type-args to the polymorphic function
    ///
    /// [LoadFunction]: crate::ops::dataflow::LoadFunction
    #[error(
        "Incorrect result of type application in LoadFunction - cached {cached} but expected {expected}"
    )]
    LoadFunctionIncorrectlyAppliesType {
        cached: Signature,
        expected: Signature,
    },
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

    fn def_name(&self) -> &OpName {
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

/// A constant value provided by a extension.
/// Must be an instance of a type available to the extension.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExtensionValue {
    extension: ExtensionId,
    name: ValueName,
    typed_value: ops::Value,
}

impl ExtensionValue {
    /// Returns a reference to the typed value of this [`ExtensionValue`].
    pub fn typed_value(&self) -> &ops::Value {
        &self.typed_value
    }

    /// Returns a reference to the name of this [`ExtensionValue`].
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns a reference to the extension this [`ExtensionValue`] belongs to.
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }
}

/// A unique identifier for a extension.
///
/// The actual [`Extension`] is stored externally.
pub type ExtensionId = IdentList;

/// A extension is a set of capabilities required to execute a graph.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Extension {
    /// Unique identifier for the extension.
    pub name: ExtensionId,
    /// Other extensions defining types used by this extension.
    /// That is, an upper-bound on the types that can be returned by
    /// computing the signature of any operation in this extension,
    /// for any possible [TypeArg].
    pub extension_reqs: ExtensionSet,
    /// Types defined by this extension.
    types: HashMap<TypeName, TypeDef>,
    /// Static values defined by this extension.
    values: HashMap<ValueName, ExtensionValue>,
    /// Operation declarations with serializable definitions.
    // Note: serde will serialize this because we configure with `features=["rc"]`.
    // That will clone anything that has multiple references, but each
    // OpDef should appear exactly once in this map (keyed by its name),
    // and the other references to the OpDef are from ExternalOp's in the Hugr
    // (which are serialized as OpaqueOp's i.e. Strings).
    operations: HashMap<OpName, Arc<op_def::OpDef>>,
}

impl Extension {
    /// Creates a new extension with the given name.
    pub fn new(name: ExtensionId) -> Self {
        Self::new_with_reqs(name, ExtensionSet::default())
    }

    /// Creates a new extension with the given name and requirements.
    pub fn new_with_reqs(name: ExtensionId, extension_reqs: impl Into<ExtensionSet>) -> Self {
        Self {
            name,
            extension_reqs: extension_reqs.into(),
            types: Default::default(),
            values: Default::default(),
            operations: Default::default(),
        }
    }

    /// Allows read-only access to the operations in this Extension
    pub fn get_op(&self, name: &OpNameRef) -> Option<&Arc<op_def::OpDef>> {
        self.operations.get(name)
    }

    /// Allows read-only access to the types in this Extension
    pub fn get_type(&self, type_name: &TypeNameRef) -> Option<&type_def::TypeDef> {
        self.types.get(type_name)
    }

    /// Allows read-only access to the values in this Extension
    pub fn get_value(&self, value_name: &ValueNameRef) -> Option<&ExtensionValue> {
        self.values.get(value_name)
    }

    /// Returns the name of the extension.
    pub fn name(&self) -> &ExtensionId {
        &self.name
    }

    /// Iterator over the operations of this [`Extension`].
    pub fn operations(&self) -> impl Iterator<Item = (&OpName, &Arc<OpDef>)> {
        self.operations.iter()
    }

    /// Iterator over the types of this [`Extension`].
    pub fn types(&self) -> impl Iterator<Item = (&TypeName, &TypeDef)> {
        self.types.iter()
    }

    /// Add a named static value to the extension.
    pub fn add_value(
        &mut self,
        name: impl Into<ValueName>,
        typed_value: ops::Value,
    ) -> Result<&mut ExtensionValue, ExtensionBuildError> {
        let extension_value = ExtensionValue {
            extension: self.name.clone(),
            name: name.into(),
            typed_value,
        };
        match self.values.entry(extension_value.name.clone()) {
            hash_map::Entry::Occupied(_) => {
                Err(ExtensionBuildError::ValueExists(extension_value.name))
            }
            hash_map::Entry::Vacant(ve) => Ok(ve.insert(extension_value)),
        }
    }

    /// Instantiate an [`ExtensionOp`] which references an [`OpDef`] in this extension.
    pub fn instantiate_extension_op(
        &self,
        name: &OpNameRef,
        args: impl Into<Vec<TypeArg>>,
        ext_reg: &ExtensionRegistry,
    ) -> Result<ExtensionOp, SignatureError> {
        let op_def = self.get_op(name).expect("Op not found.");
        ExtensionOp::new(op_def.clone(), args, ext_reg)
    }

    // Validates against a registry, which we can assume includes this extension itself.
    // (TODO deal with the registry itself containing invalid extensions!)
    fn validate(&self, all_exts: &ExtensionRegistry) -> Result<(), SignatureError> {
        // We should validate TypeParams of TypeDefs too - https://github.com/CQCL/hugr/issues/624
        for op_def in self.operations.values() {
            op_def.validate(all_exts)?;
        }
        Ok(())
    }
}

impl PartialEq for Extension {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// An error that can occur in defining an extension registry.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ExtensionRegistryError {
    /// Extension already defined.
    #[error("The registry already contains an extension with id {0}.")]
    AlreadyRegistered(ExtensionId),
    /// A registered extension has invalid signatures.
    #[error("The extension {0} contains an invalid signature, {1}.")]
    InvalidSignature(ExtensionId, #[source] SignatureError),
}

/// An error that can occur in building a new extension.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ExtensionBuildError {
    /// Existing [`OpDef`]
    #[error("Extension already has an op called {0}.")]
    OpDefExists(OpName),
    /// Existing [`TypeDef`]
    #[error("Extension already has an type called {0}.")]
    TypeDefExists(TypeName),
    /// Existing [`ExtensionValue`]
    #[error("Extension already has an extension value called {0}.")]
    ValueExists(ValueName),
}

/// A set of extensions identified by their unique [`ExtensionId`].
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSet(BTreeSet<ExtensionId>);

/// A special ExtensionId which indicates that the delta of a non-Function
/// container node should be computed by extension inference. See [`infer_extensions`]
/// which lists the container nodes to which this can be applied.
///
/// [`infer_extensions`]: crate::hugr::Hugr::infer_extensions
pub const TO_BE_INFERRED: ExtensionId = ExtensionId::new_unchecked(".TO_BE_INFERRED");

impl ExtensionSet {
    /// Creates a new empty extension set.
    pub const fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Adds a extension to the set.
    pub fn insert(&mut self, extension: &ExtensionId) {
        self.0.insert(extension.clone());
    }

    /// Adds a type var (which must have been declared as a [TypeParam::Extensions]) to this set
    pub fn insert_type_var(&mut self, idx: usize) {
        // Represent type vars as string representation of variable index.
        // This is not a legal IdentList or ExtensionId so should not conflict.
        self.0
            .insert(ExtensionId::new_unchecked(idx.to_string().as_str()));
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

    /// An ExtensionSet containing a single type variable
    /// (which must have been declared as a [TypeParam::Extensions])
    pub fn type_var(idx: usize) -> Self {
        let mut set = Self::new();
        set.insert_type_var(idx);
        set
    }

    /// Returns the union of two extension sets.
    pub fn union(mut self, other: Self) -> Self {
        self.0.extend(other.0);
        self
    }

    /// Returns the union of an arbitrary collection of [ExtensionSet]s
    pub fn union_over(sets: impl IntoIterator<Item = Self>) -> Self {
        // `union` clones the receiver, which we do not need to do here
        let mut res = ExtensionSet::new();
        for s in sets {
            res.0.extend(s.0)
        }
        res
    }

    /// The things in other which are in not in self
    pub fn missing_from(&self, other: &Self) -> Self {
        ExtensionSet::from_iter(other.0.difference(&self.0).cloned())
    }

    /// Iterate over the contained ExtensionIds
    pub fn iter(&self) -> impl Iterator<Item = &ExtensionId> {
        self.0.iter()
    }

    /// True if this set contains no [ExtensionId]s
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn validate(&self, params: &[TypeParam]) -> Result<(), SignatureError> {
        self.iter()
            .filter_map(as_typevar)
            .try_for_each(|var_idx| check_typevar_decl(params, var_idx, &TypeParam::Extensions))
    }

    pub(crate) fn substitute(&self, t: &Substitution) -> Self {
        Self::from_iter(self.0.iter().flat_map(|e| match as_typevar(e) {
            None => vec![e.clone()],
            Some(i) => match t.apply_var(i, &TypeParam::Extensions) {
                TypeArg::Extensions{es} => es.iter().cloned().collect::<Vec<_>>(),
                _ => panic!("value for type var was not extension set - type scheme should be validated first"),
            },
        }))
    }
}

impl From<ExtensionId> for ExtensionSet {
    fn from(id: ExtensionId) -> Self {
        Self::singleton(&id)
    }
}

fn as_typevar(e: &ExtensionId) -> Option<usize> {
    // Type variables are represented as radix-10 numbers, which are illegal
    // as standard ExtensionIds. Hence if an ExtensionId starts with a digit,
    // we assume it must be a type variable, and fail fast if it isn't.
    match e.chars().next() {
        Some(c) if c.is_ascii_digit() => Some(str::parse(e).unwrap()),
        _ => None,
    }
}

impl Display for ExtensionSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl FromIterator<ExtensionId> for ExtensionSet {
    fn from_iter<I: IntoIterator<Item = ExtensionId>>(iter: I) -> Self {
        Self(BTreeSet::from_iter(iter))
    }
}

#[cfg(test)]
pub mod test {
    // We re-export this here because mod op_def is private.
    pub use super::op_def::test::SimpleOpDef;

    mod proptest {

        use ::proptest::{collection::hash_set, prelude::*};

        use super::super::{ExtensionId, ExtensionSet};

        impl Arbitrary for ExtensionSet {
            type Parameters = ();
            type Strategy = BoxedStrategy<Self>;

            fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
                (
                    hash_set(0..10usize, 0..3),
                    hash_set(any::<ExtensionId>(), 0..3),
                )
                    .prop_map(|(vars, extensions)| {
                        ExtensionSet::union_over(
                            std::iter::once(extensions.into_iter().collect::<ExtensionSet>())
                                .chain(vars.into_iter().map(ExtensionSet::type_var)),
                        )
                    })
                    .boxed()
            }
        }
    }
}
