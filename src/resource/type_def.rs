use std::collections::hash_map::Entry;

use super::ResourceBuildError;
use super::{Resource, ResourceId, SignatureError, TypeParametrised};

use crate::types::{least_upper_bound, CustomType};

use crate::types::type_param::TypeArg;

use crate::types::type_param::TypeParam;

use smol_str::SmolStr;

use crate::types::TypeBound;

/// The type bound of a [`TypeDef`]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeDefBound {
    /// Defined by an explicit tag.
    Explicit(Option<TypeBound>),
    /// Derived as the tag containing all marked type parameters.
    FromParams(Vec<usize>),
}

impl From<TypeBound> for TypeDefBound {
    fn from(bound: TypeBound) -> Self {
        Self::Explicit(Some(bound))
    }
}

/// A declaration of an opaque type.
/// Note this does not provide any way to create instances
/// - typically these are operations also provided by the Resource.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TypeDef {
    /// The unique Resource owning this TypeDef (of which this TypeDef is a member)
    resource: ResourceId,
    /// The unique name of the type
    name: SmolStr,
    /// Declaration of type parameters. The TypeDef must be instantiated
    /// with the same number of [`TypeArg`]'s to make an actual type.
    ///
    /// [`TypeArg`]: crate::types::type_param::TypeArg
    params: Vec<TypeParam>,
    /// Human readable description of the type definition.
    description: String,
    /// The definition of the type tag of this definition.
    tag: TypeDefBound,
}

impl TypeDef {
    /// Check provided type arguments are valid against parameters.
    pub fn check_args(&self, args: &[TypeArg]) -> Result<(), SignatureError> {
        self.check_args_impl(args)
    }

    /// Check [`CustomType`] is a valid instantiation of this definition.
    ///
    /// # Errors
    ///
    /// This function will return an error if the type of the instance does not
    /// match the definition.
    pub fn check_custom(&self, custom: &CustomType) -> Result<(), SignatureError> {
        self.check_concrete_impl(custom)
    }

    /// Instantiate a concrete [`CustomType`] by providing type arguments.
    ///
    /// # Errors
    ///
    /// This function will return an error if the provided arguments are not
    /// valid instances of the type parameters.
    pub fn instantiate_concrete(
        &self,
        args: impl Into<Vec<TypeArg>>,
    ) -> Result<CustomType, SignatureError> {
        let args = args.into();
        self.check_args_impl(&args)?;
        let bound = self.bound(&args);
        Ok(CustomType::new(
            self.name().clone(),
            args,
            self.resource().clone(),
            bound,
        ))
    }
    /// The [`TypeBound`] of the definition.
    pub fn bound(&self, args: &[TypeArg]) -> Option<TypeBound> {
        match &self.tag {
            TypeDefBound::Explicit(bound) => *bound,
            TypeDefBound::FromParams(indices) => {
                let args: Vec<_> = args.iter().collect();
                if indices.is_empty() {
                    // Assume most general case
                    return None;
                }
                least_upper_bound(indices.iter().map(|i| {
                    args.get(*i).and_then(|ta| match ta {
                        TypeArg::Type(s) => todo!(),
                        _ => panic!("TypeArg index does not refer to a type."),
                    })
                }))
            }
        }
    }
}

impl TypeParametrised for TypeDef {
    type Concrete = CustomType;

    fn params(&self) -> &[TypeParam] {
        &self.params
    }

    fn name(&self) -> &SmolStr {
        &self.name
    }

    fn resource(&self) -> &ResourceId {
        &self.resource
    }
}

impl Resource {
    /// Add an exported type to the resource.
    pub fn add_type(
        &mut self,
        name: SmolStr,
        params: Vec<TypeParam>,
        description: String,
        tag: TypeDefBound,
    ) -> Result<&TypeDef, ResourceBuildError> {
        let ty = TypeDef {
            resource: self.name().into(),
            name,
            params,
            description,
            tag,
        };
        match self.types.entry(ty.name.clone()) {
            Entry::Occupied(_) => Err(ResourceBuildError::OpDefExists(ty.name)),
            Entry::Vacant(ve) => Ok(ve.insert(ty)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::resource::SignatureError;
    use crate::types::test::{ANY_T, COPYABLE_T, EQ_T};
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{AbstractSignature, Type, TypeBound};

    use super::{TypeDef, TypeDefBound};

    #[test]
    fn test_instantiate_typedef() {
        let def = TypeDef {
            name: "MyType".into(),
            params: vec![TypeParam::Type(Some(TypeBound::Copyable))],
            resource: "MyRsrc".into(),
            description: "Some parameterised type".into(),
            tag: TypeDefBound::FromParams(vec![0]),
        };
        let typ = Type::new_extension(
            def.instantiate_concrete(vec![TypeArg::Type(Type::graph(AbstractSignature::new_df(
                vec![],
                vec![],
            )))])
            .unwrap(),
        );
        assert_eq!(typ.least_upper_bound(), Some(TypeBound::Copyable));
        let typ2 = Type::new_extension(def.instantiate_concrete([TypeArg::Type(EQ_T)]).unwrap());
        assert_eq!(typ2.least_upper_bound(), Some(TypeBound::Eq));

        // And some bad arguments...firstly, wrong kind of TypeArg:
        assert_eq!(
            def.instantiate_concrete([TypeArg::Type(ANY_T)]),
            Err(SignatureError::TypeArgMismatch(TypeArgError::TypeMismatch(
                TypeArg::Type(ANY_T),
                TypeParam::Type(Some(TypeBound::Copyable))
            )))
        );
        // Too few arguments:
        assert_eq!(
            def.instantiate_concrete([]).unwrap_err(),
            SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(0, 1))
        );
        // Too many arguments:
        assert_eq!(
            def.instantiate_concrete([TypeArg::Type(COPYABLE_T), TypeArg::Type(COPYABLE_T),])
                .unwrap_err(),
            SignatureError::TypeArgMismatch(TypeArgError::WrongNumberArgs(2, 1))
        );
    }
}
