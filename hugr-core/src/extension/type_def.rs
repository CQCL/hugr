use std::collections::hash_map::Entry;

use super::{CustomConcrete, ExtensionBuildError};
use super::{Extension, ExtensionId, SignatureError};

use crate::types::{least_upper_bound, CustomType, TypeName};

use crate::types::type_param::{check_type_args, TypeArg};

use crate::types::type_param::TypeParam;

use crate::types::TypeBound;

/// The type bound of a [`TypeDef`]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeDefBound {
    /// Defined by an explicit bound.
    Explicit(TypeBound),
    /// Derived as the least upper bound of the marked parameters.
    FromParams(Vec<usize>),
}

impl From<TypeBound> for TypeDefBound {
    fn from(bound: TypeBound) -> Self {
        Self::Explicit(bound)
    }
}

/// A declaration of an opaque type.
/// Note this does not provide any way to create instances
/// - typically these are operations also provided by the Extension.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TypeDef {
    /// The unique Extension owning this TypeDef (of which this TypeDef is a member)
    extension: ExtensionId,
    /// The unique name of the type
    name: TypeName,
    /// Declaration of type parameters. The TypeDef must be instantiated
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
        check_type_args(args, &self.params).map_err(SignatureError::TypeArgMismatch)
    }

    /// Check [`CustomType`] is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    pub fn check_custom(&self, custom: &CustomType) -> Result<(), SignatureError> {
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

        check_type_args(custom.type_args(), &self.params)?;

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
        check_type_args(&args, &self.params)?;
        let bound = self.bound(&args);
        Ok(CustomType::new(
            self.name().clone(),
            args,
            self.extension().clone(),
            bound,
        ))
    }
    /// The [`TypeBound`] of the definition.
    pub fn bound(&self, args: &[TypeArg]) -> TypeBound {
        match &self.bound {
            TypeDefBound::Explicit(bound) => *bound,
            TypeDefBound::FromParams(indices) => {
                let args: Vec<_> = args.iter().collect();
                if indices.is_empty() {
                    // Assume most general case
                    return TypeBound::Any;
                }
                least_upper_bound(indices.iter().map(|i| {
                    let ta = args.get(*i);
                    match ta {
                        Some(TypeArg::Type { ty: s }) => s.least_upper_bound(),
                        _ => panic!("TypeArg index does not refer to a type."),
                    }
                }))
            }
        }
    }

    /// The static parameters to the TypeDef; a [TypeArg] appropriate for each
    /// must be provided to produce an actual type.
    pub fn params(&self) -> &[TypeParam] {
        &self.params
    }

    fn name(&self) -> &TypeName {
        &self.name
    }

    fn extension(&self) -> &ExtensionId {
        &self.extension
    }
}

impl Extension {
    /// Add an exported type to the extension.
    pub fn add_type(
        &mut self,
        name: TypeName,
        params: Vec<TypeParam>,
        description: String,
        bound: TypeDefBound,
    ) -> Result<&TypeDef, ExtensionBuildError> {
        let ty = TypeDef {
            extension: self.name.clone(),
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
    use crate::extension::prelude::{QB_T, USIZE_T};
    use crate::extension::SignatureError;
    use crate::std_extensions::arithmetic::float_types::FLOAT64_TYPE;
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{Signature, Type, TypeBound};

    use super::{TypeDef, TypeDefBound};

    #[test]
    fn test_instantiate_typedef() {
        let def = TypeDef {
            name: "MyType".into(),
            params: vec![TypeParam::Type {
                b: TypeBound::Copyable,
            }],
            extension: "MyRsrc".try_into().unwrap(),
            description: "Some parametrised type".into(),
            bound: TypeDefBound::FromParams(vec![0]),
        };
        let typ = Type::new_extension(
            def.instantiate(vec![TypeArg::Type {
                ty: Type::new_function(Signature::new(vec![], vec![])),
            }])
            .unwrap(),
        );
        assert_eq!(typ.least_upper_bound(), TypeBound::Copyable);
        let typ2 = Type::new_extension(def.instantiate([USIZE_T.into()]).unwrap());
        assert_eq!(typ2.least_upper_bound(), TypeBound::Eq);

        // And some bad arguments...firstly, wrong kind of TypeArg:
        assert_eq!(
            def.instantiate([TypeArg::Type { ty: QB_T }]),
            Err(SignatureError::TypeArgMismatch(
                TypeArgError::TypeMismatch {
                    arg: TypeArg::Type { ty: QB_T },
                    param: TypeBound::Copyable.into()
                }
            ))
        );
        // Too few arguments:
        assert_eq!(
            def.instantiate([]).unwrap_err(),
            SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(0, 1))
        );
        // Too many arguments:
        assert_eq!(
            def.instantiate([
                TypeArg::Type { ty: FLOAT64_TYPE },
                TypeArg::Type { ty: FLOAT64_TYPE },
            ])
            .unwrap_err(),
            SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(2, 1))
        );
    }
}
