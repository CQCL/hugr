//! Opaque types, used to represent a user-defined [`Type`].
//!
//! [`Type`]: super::Type
use smol_str::SmolStr;
use std::fmt::{self, Display};

use crate::extension::{ExtensionId, ExtensionRegistry, SignatureError, TypeDef};

use super::{
    type_param::{TypeArg, TypeParam},
    TypeBound, TypeTransformer,
};

/// An opaque type element. Contains the unique identifier of its definition.
#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    extension: ExtensionId,
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::extension::TypeDef
    id: SmolStr,
    /// Arguments that fit the [`TypeParam`]s declared by the typedef
    ///
    /// [`TypeParam`]: super::type_param::TypeParam
    args: Vec<TypeArg>,
    /// The [TypeBound] describing what can be done to instances of this type
    bound: TypeBound,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(
        id: impl Into<SmolStr>,
        args: impl Into<Vec<TypeArg>>,
        extension: ExtensionId,
        bound: TypeBound,
    ) -> Self {
        Self {
            id: id.into(),
            args: args.into(),
            extension,
            bound,
        }
    }

    /// Creates a new opaque type (constant version, no type arguments)
    pub const fn new_simple(id: SmolStr, extension: ExtensionId, bound: TypeBound) -> Self {
        Self {
            id,
            args: vec![],
            extension,
            bound,
        }
    }

    /// Returns the bound of this [`CustomType`].
    pub const fn bound(&self) -> TypeBound {
        self.bound
    }

    pub(super) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        // Check the args are individually ok
        self.args
            .iter()
            .try_for_each(|a| a.validate(extension_registry, var_decls))?;
        // And check they fit into the TypeParams declared by the TypeDef
        let def = self.get_type_def(extension_registry)?;
        def.check_custom(self)
    }

    fn get_type_def<'a>(
        &self,
        extension_registry: &'a ExtensionRegistry,
    ) -> Result<&'a TypeDef, SignatureError> {
        let ex = extension_registry.get(&self.extension);
        // Even if OpDef's (+binaries) are not available, the part of the Extension definition
        // describing the TypeDefs can easily be passed around (serialized), so should be available.
        let ex = ex.ok_or(SignatureError::ExtensionNotFound(self.extension.clone()))?;
        ex.get_type(&self.id)
            .ok_or(SignatureError::ExtensionTypeNotFound {
                exn: self.extension.clone(),
                typ: self.id.clone(),
            })
    }

    pub(super) fn transform(&self, tr: &impl TypeTransformer) -> Self {
        let args = self
            .args
            .iter()
            .map(|arg| arg.transform(tr))
            .collect::<Vec<_>>();
        let bound = self
            .get_type_def(tr.extension_registry())
            .expect("validate should rule this out")
            .bound(&args);
        assert!(self.bound.contains(bound));
        Self {
            args,
            bound,
            ..self.clone()
        }
    }

    /// unique name of the type.
    pub fn name(&self) -> &SmolStr {
        &self.id
    }

    /// Type arguments.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Parent extension.
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }
}

impl Display for CustomType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?})", self.id, self.args)
    }
}
