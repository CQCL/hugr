//! Opaque types, used to represent a user-defined [`Type`].
//!
//! [`Type`]: super::Type
use std::fmt::{self, Display};

use crate::extension::{ExtensionId, ExtensionRegistry, SignatureError, TypeDef};

use super::{
    type_param::{TypeArg, TypeParam},
    Substitution, TypeBound,
};
use super::{Type, TypeName};

/// An opaque type element. Contains the unique identifier of its definition.
#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    extension: ExtensionId,
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::extension::TypeDef
    id: TypeName,
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
        id: impl Into<TypeName>,
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
    pub const fn new_simple(id: TypeName, extension: ExtensionId, bound: TypeBound) -> Self {
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

    pub(super) fn substitute(&self, tr: &Substitution) -> Self {
        let args = self
            .args
            .iter()
            .map(|arg| arg.substitute(tr))
            .collect::<Vec<_>>();
        let bound = self
            .get_type_def(tr.extension_registry())
            .expect("validate should rule this out")
            .bound(&args);
        debug_assert!(self.bound.contains(bound));
        Self {
            args,
            bound,
            ..self.clone()
        }
    }

    /// unique name of the type.
    pub fn name(&self) -> &TypeName {
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
        if self.args.is_empty() {
            write!(f, "{}", self.id)
        } else {
            write!(f, "{}({:?})", self.id, self.args)
        }
    }
}

impl From<CustomType> for Type {
    fn from(value: CustomType) -> Self {
        Self::new_extension(value)
    }
}

#[cfg(test)]
pub mod test {

    pub mod proptest {
        use crate::extension::ExtensionId;
        use crate::proptest::any_nonempty_string;
        use crate::proptest::RecursionDepth;
        use crate::types::type_param::TypeArg;
        use crate::types::{CustomType, TypeBound};
        use ::proptest::collection::vec;
        use ::proptest::prelude::*;

        #[derive(Default)]
        pub struct CustomTypeArbitraryParameters(RecursionDepth, Option<TypeBound>);

        impl From<RecursionDepth> for CustomTypeArbitraryParameters {
            fn from(v: RecursionDepth) -> Self {
                Self::new(v)
            }
        }

        impl CustomTypeArbitraryParameters {
            pub fn new(depth: RecursionDepth) -> Self {
                Self(depth, None)
            }

            pub fn with_bound(mut self, bound: TypeBound) -> Self {
                self.1 = Some(bound);
                self
            }
        }

        impl Arbitrary for CustomType {
            type Parameters = CustomTypeArbitraryParameters;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(
                CustomTypeArbitraryParameters(depth, mb_bound): Self::Parameters,
            ) -> Self::Strategy {
                let bound = mb_bound.map_or(any::<TypeBound>().boxed(), |x| Just(x).boxed());
                let args = if depth.leaf() {
                    Just(vec![]).boxed()
                } else {
                    // a TypeArg may contain a CustomType, so we descend here
                    vec(any_with::<TypeArg>(depth.descend()), 0..3).boxed()
                };
                (any_nonempty_string(), args, any::<ExtensionId>(), bound)
                    .prop_map(|(id, args, extension, bound)| Self::new(id, args, extension, bound))
                    .boxed()
            }
        }
    }
}
