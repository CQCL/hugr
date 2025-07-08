//! Opaque types, used to represent a user-defined [`Type`].
//!
//! [`Type`]: super::Type
use std::fmt::{self, Display};
use std::sync::{Arc, Weak};

use crate::Extension;
use crate::extension::{ExtensionId, SignatureError, TypeDef};

use super::{
    Substitution, TypeBound,
    type_param::{TypeArg, TypeParam},
};
use super::{Type, TypeName};

/// An opaque type element. Contains the unique identifier of its definition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// The identifier for the extension owning this type.
    extension: ExtensionId,
    /// A weak reference to the extension defining this type.
    #[serde(skip)]
    extension_ref: Weak<Extension>,
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::extension::TypeDef
    id: TypeName,
    /// Arguments that fit the [`TypeParam`]s declared by the typedef
    ///
    /// [`TypeParam`]: super::type_param::TypeParam
    args: Vec<TypeArg>,
    /// The [`TypeBound`] describing what can be done to instances of this type
    bound: TypeBound,
}

impl std::hash::Hash for CustomType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.extension.hash(state);
        self.id.hash(state);
        self.args.hash(state);
        self.bound.hash(state);
    }
}

impl PartialEq for CustomType {
    fn eq(&self, other: &Self) -> bool {
        self.extension == other.extension
            && self.id == other.id
            && self.args == other.args
            && self.bound == other.bound
    }
}

impl Eq for CustomType {}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(
        id: impl Into<TypeName>,
        args: impl Into<Vec<TypeArg>>,
        extension: ExtensionId,
        bound: TypeBound,
        extension_ref: &Weak<Extension>,
    ) -> Self {
        Self {
            id: id.into(),
            args: args.into(),
            extension,
            bound,
            extension_ref: extension_ref.clone(),
        }
    }

    /// Returns the bound of this [`CustomType`].
    #[must_use]
    pub const fn bound(&self) -> TypeBound {
        self.bound
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        // Check the args are individually ok
        self.args.iter().try_for_each(|a| a.validate(var_decls))?;
        // And check they fit into the TypeParams declared by the TypeDef
        let ext = self.get_extension()?;
        let def = self.get_type_def(&ext)?;
        def.check_custom(self)
    }

    pub(super) fn get_type_def<'a>(
        &self,
        ext: &'a Arc<Extension>,
    ) -> Result<&'a TypeDef, SignatureError> {
        ext.get_type(&self.id)
            .ok_or(SignatureError::ExtensionTypeNotFound {
                exn: self.extension.clone(),
                typ: self.id.clone(),
            })
    }

    pub(super) fn get_extension(&self) -> Result<Arc<Extension>, SignatureError> {
        self.extension_ref
            .upgrade()
            .ok_or(SignatureError::MissingTypeExtension {
                missing: self.extension.clone(),
                typ: self.name().clone(),
            })
    }

    pub(super) fn substitute(&self, tr: &Substitution) -> Self {
        let args = self
            .args
            .iter()
            .map(|arg| arg.substitute(tr))
            .collect::<Vec<_>>();
        let ext = self.get_extension().unwrap_or_else(|e| panic!("{}", e));
        let bound = self
            .get_type_def(&ext)
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
    #[must_use]
    pub fn name(&self) -> &TypeName {
        &self.id
    }

    /// Type arguments.
    #[must_use]
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns a mutable reference to the type arguments.
    pub(crate) fn args_mut(&mut self) -> &mut Vec<TypeArg> {
        &mut self.args
    }

    /// Parent extension.
    #[must_use]
    pub fn extension(&self) -> &ExtensionId {
        &self.extension
    }

    /// Returns a weak reference to the extension defining this type.
    #[must_use]
    pub fn extension_ref(&self) -> Weak<Extension> {
        self.extension_ref.clone()
    }

    /// Update the internal extension reference with a new weak pointer.
    pub fn update_extension(&mut self, extension_ref: Weak<Extension>) {
        self.extension_ref = extension_ref;
    }
}

impl Display for CustomType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)?;
        if !self.args.is_empty() {
            write!(f, "(")?;
            crate::utils::display_list(&self.args, f)?;
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl From<CustomType> for Type {
    fn from(value: CustomType) -> Self {
        Self::new_extension(value)
    }
}

#[cfg(test)]
mod test {

    pub mod proptest {
        use std::sync::Weak;

        use crate::extension::ExtensionId;
        use crate::proptest::RecursionDepth;
        use crate::proptest::any_nonempty_string;
        use crate::types::proptest_utils::any_serde_type_arg;
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
                    vec(any_serde_type_arg(depth.descend()), 0..3).boxed()
                };
                (any_nonempty_string(), args, any::<ExtensionId>(), bound)
                    .prop_map(|(id, args, extension, bound)| {
                        Self::new(id, args, extension, bound, &Weak::default())
                    })
                    .boxed()
            }
        }
    }
}
