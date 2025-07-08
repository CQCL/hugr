use std::collections::btree_map::Entry;
use std::sync::Weak;

use super::{CustomConcrete, ExtensionBuildError};
use super::{Extension, ExtensionId, SignatureError};

use crate::types::{CustomType, TypeName, least_upper_bound};

use crate::types::type_param::{TypeArg, check_term_types};

use crate::types::type_param::TypeParam;

use crate::types::TypeBound;

/// The type bound of a [`TypeDef`]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "b")]
#[allow(missing_docs)]
pub enum TypeDefBound {
    /// Defined by an explicit bound.
    Explicit { bound: TypeBound },
    /// Derived as the least upper bound of the marked parameters.
    FromParams { indices: Vec<usize> },
}

impl From<TypeBound> for TypeDefBound {
    fn from(bound: TypeBound) -> Self {
        Self::Explicit { bound }
    }
}

impl TypeDefBound {
    /// Create a new [`TypeDefBound::Explicit`] with the `Any` bound.
    #[must_use]
    pub fn any() -> Self {
        TypeDefBound::Explicit {
            bound: TypeBound::Linear,
        }
    }

    /// Create a new [`TypeDefBound::Explicit`] with the `Copyable` bound.
    #[must_use]
    pub fn copyable() -> Self {
        TypeDefBound::Explicit {
            bound: TypeBound::Copyable,
        }
    }

    /// Create a new [`TypeDefBound::FromParams`] with the given indices.
    #[must_use]
    pub fn from_params(indices: Vec<usize>) -> Self {
        TypeDefBound::FromParams { indices }
    }
}

/// A declaration of an opaque type.
/// Note this does not provide any way to create instances
/// - typically these are operations also provided by the Extension.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TypeDef {
    /// The unique Extension owning this `TypeDef` (of which this `TypeDef` is a member)
    extension: ExtensionId,
    /// A weak reference to the extension defining this operation.
    #[serde(skip)]
    extension_ref: Weak<Extension>,
    /// The unique name of the type
    name: TypeName,
    /// Declaration of type parameters. The `TypeDef` must be instantiated
    /// with the same number of [`TypeArg`]'s to make an actual type.
    ///
    /// [`TypeArg`]: crate::types::type_param::TypeArg
    params: Vec<TypeParam>,
    /// Human readable description of the type definition.
    description: String,
    /// The definition of the type bound of this definition.
    bound: TypeDefBound,
}

impl TypeDef {
    /// Check provided type arguments are valid against parameters.
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        check_term_types(args, &self.params).map_err(SignatureError::TypeArgMismatch)
    }

    /// Check [`CustomType`] is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    pub fn check_custom(&self, custom: &CustomType) -> Result<(), SignatureError> {
        if self.extension_id() != custom.parent_extension() {
            return Err(SignatureError::ExtensionMismatch(
                self.extension_id().clone(),
                custom.parent_extension().clone(),
            ));
        }
        if self.name() != custom.def_name() {
            return Err(SignatureError::NameMismatch(
                self.name().clone(),
                custom.def_name().clone(),
            ));
        }

        check_term_types(custom.type_args(), &self.params)?;

        let calc_bound = self.bound(custom.args());
        if calc_bound == custom.bound() {
            Ok(())
        } else {
            Err(SignatureError::WrongBound {
                expected: calc_bound,
                actual: custom.bound(),
            })
        }
    }

    /// Instantiate a concrete [`CustomType`] by providing type arguments.
    ///
    /// # Errors
    ///
    /// This function will return an error if the provided arguments are not
    /// valid instances of the type parameters.
    pub fn instantiate(&self, args: impl Into<Vec<TypeArg>>) -> Result<CustomType, SignatureError> {
        let args = args.into();
        check_term_types(&args, &self.params)?;
        let bound = self.bound(&args);
        Ok(CustomType::new(
            self.name().clone(),
            args,
            self.extension_id().clone(),
            bound,
            &self.extension_ref,
        ))
    }
    /// The [`TypeBound`] of the definition.
    #[must_use]
    pub fn bound(&self, args: &[TypeArg]) -> TypeBound {
        match &self.bound {
            TypeDefBound::Explicit { bound } => *bound,
            TypeDefBound::FromParams { indices } => {
                let args: Vec<_> = args.iter().collect();
                if indices.is_empty() {
                    // Assume most general case
                    return TypeBound::Linear;
                }
                least_upper_bound(indices.iter().map(|i| {
                    let ta = args.get(*i);
                    match ta {
                        Some(TypeArg::Runtime(s)) => s.least_upper_bound(),
                        _ => panic!("TypeArg index does not refer to a type."),
                    }
                }))
            }
        }
    }

    /// The static parameters to the `TypeDef`; a [`TypeArg`] appropriate for each
    /// must be provided to produce an actual type.
    #[must_use]
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    /// The type name of the definition.
    #[must_use]
    pub fn name(&self) -> &TypeName {
        &self.name
    }

    /// Returns a reference to the extension id of this [`TypeDef`].
    #[must_use]
    pub fn extension_id(&self) -> &ExtensionId {
        &self.extension
    }

    /// Returns a weak reference to the extension defining this type.
    #[must_use]
    pub fn extension(&self) -> Weak<Extension> {
        self.extension_ref.clone()
    }

    /// Returns a mutable reference to the weak extension pointer in the type def.
    pub(super) fn extension_mut(&mut self) -> &mut Weak<Extension> {
        &mut self.extension_ref
    }
}

impl Extension {
    /// Add an exported type to the extension.
    ///
    /// This method requires a [`Weak`] reference to the [`std::sync::Arc`] containing the
    /// extension being defined. The intended way to call this method is inside
    /// the closure passed to [`Extension::new_arc`] when defining the extension.
    ///
    /// # Example
    ///
    /// ```
    /// # use hugr_core::types::Signature;
    /// # use hugr_core::extension::{Extension, ExtensionId, Version};
    /// # use hugr_core::extension::{TypeDefBound};
    /// Extension::new_arc(
    ///     ExtensionId::new_unchecked("my.extension"),
    ///     Version::new(0, 1, 0),
    ///     |ext, extension_ref| {
    ///         ext.add_type(
    ///             "MyType".into(),
    ///             vec![], // No type parameters
    ///             "Some type".into(),
    ///             TypeDefBound::any(),
    ///             extension_ref,
    ///         );
    ///     },
    /// );
    /// ```
    pub fn add_type(
        &mut self,
        name: TypeName,
        params: Vec<TypeParam>,
        description: String,
        bound: TypeDefBound,
        extension_ref: &Weak<Extension>,
    ) -> Result<&TypeDef, ExtensionBuildError> {
        let ty = TypeDef {
            extension: self.name.clone(),
            extension_ref: extension_ref.clone(),
            name,
            params,
            description,
            bound,
        };
        match self.types.entry(ty.name.clone()) {
            Entry::Occupied(_) => Err(ExtensionBuildError::TypeDefExists(ty.name)),
            Entry::Vacant(ve) => Ok(ve.insert(ty)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::extension::SignatureError;
    use crate::extension::prelude::{qb_t, usize_t};
    use crate::std_extensions::arithmetic::float_types::float64_type;
    use crate::types::type_param::{TermTypeError, TypeParam};
    use crate::types::{Signature, Type, TypeBound};

    use super::{TypeDef, TypeDefBound};

    #[test]
    fn test_instantiate_typedef() {
        let def = TypeDef {
            name: "MyType".into(),
            params: vec![TypeParam::RuntimeType(TypeBound::Copyable)],
            extension: "MyRsrc".try_into().unwrap(),
            // Dummy extension. Will return `None` when trying to upgrade it into an `Arc`.
            extension_ref: Default::default(),
            description: "Some parametrised type".into(),
            bound: TypeDefBound::FromParams { indices: vec![0] },
        };
        let typ = Type::new_extension(
            def.instantiate(vec![
                Type::new_function(Signature::new(vec![], vec![])).into(),
            ])
            .unwrap(),
        );
        assert_eq!(typ.least_upper_bound(), TypeBound::Copyable);
        let typ2 = Type::new_extension(def.instantiate([usize_t().into()]).unwrap());
        assert_eq!(typ2.least_upper_bound(), TypeBound::Copyable);

        // And some bad arguments...firstly, wrong kind of TypeArg:
        assert_eq!(
            def.instantiate([qb_t().into()]),
            Err(SignatureError::TypeArgMismatch(
                TermTypeError::TypeMismatch {
                    term: Box::new(qb_t().into()),
                    type_: Box::new(TypeBound::Copyable.into())
                }
            ))
        );
        // Too few arguments:
        assert_eq!(
            def.instantiate([]).unwrap_err(),
            SignatureError::TypeArgMismatch(TermTypeError::WrongNumberArgs(0, 1))
        );
        // Too many arguments:
        assert_eq!(
            def.instantiate([float64_type().into(), float64_type().into(),])
                .unwrap_err(),
            SignatureError::TypeArgMismatch(TermTypeError::WrongNumberArgs(2, 1))
        );
    }
}
