use std::collections::hash_map::Entry;

use super::ResourceBuildError;
use super::{Resource, ResourceId, SignatureError, TypeParametrised};

use crate::types::CustomType;

use crate::types::type_param::TypeArg;

use crate::types::type_param::TypeParam;

use smol_str::SmolStr;

use crate::types::TypeTag;

/// The type tag of a [`TypeDef`]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TypeDefTag {
    /// Defined by an explicit tag.
    Explicit(TypeTag),
    /// Derived as the tag containing all marked type parameters.
    FromParams(Vec<usize>),
}

impl From<TypeTag> for TypeDefTag {
    fn from(tag: TypeTag) -> Self {
        Self::Explicit(tag)
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
    tag: TypeDefTag,
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
        let tag = self.tag(&args);
        Ok(CustomType::new(
            self.name().clone(),
            args,
            self.resource().clone(),
            tag,
        ))
    }
    /// The [`TypeTag`] of the definition.
    pub fn tag(&self, args: &[TypeArg]) -> TypeTag {
        match &self.tag {
            TypeDefTag::Explicit(tag) => *tag,
            TypeDefTag::FromParams(indices) => {
                let args: Vec<_> = args.iter().collect();
                if indices.is_empty() {
                    // Assume most general case
                    return TypeTag::Simple;
                }
                indices
                    .iter()
                    .map(|i| {
                        args.get(*i)
                            .and_then(|ta| ta.tag_of_type())
                            .expect("TypeParam index invalid or param does not have a TypeTag.")
                    })
                    .fold(TypeTag::Hashable, TypeTag::union)
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
        tag: TypeDefTag,
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
    use crate::types::type_param::{TypeArg, TypeArgError, TypeParam};
    use crate::types::{
        AbstractSignature, ClassicType, HashableType, PrimType, SimpleType, TypeTag,
    };

    use super::{TypeDef, TypeDefTag};

    #[test]
    fn test_instantiate_typedef() {
        let def = TypeDef {
            name: "MyType".into(),
            params: vec![TypeParam::Type(TypeTag::Classic)],
            resource: "MyRsrc".into(),
            description: "Some parameterised type".into(),
            tag: TypeDefTag::FromParams(vec![0]),
        };
        let typ: SimpleType = def
            .instantiate_concrete(vec![TypeArg::Type(
                ClassicType::Graph(Box::new(AbstractSignature::new_df(vec![], vec![]))).into(),
            )])
            .unwrap()
            .into();
        assert_eq!(typ.tag(), TypeTag::Classic);
        let typ2: SimpleType = def
            .instantiate_concrete([TypeArg::Type(HashableType::String.into())])
            .unwrap()
            .into();
        assert_eq!(typ2.tag(), TypeTag::Hashable);

        // And some bad arguments...firstly, wrong kind of TypeArg:
        assert_eq!(
            def.instantiate_concrete([TypeArg::Type(SimpleType::Qubit)]),
            Err(SignatureError::TypeArgMismatch(TypeArgError::TypeMismatch(
                TypeArg::Type(SimpleType::Qubit),
                TypeParam::Type(TypeTag::Classic)
            )))
        );
        // Too few arguments:
        assert_eq!(
            def.instantiate_concrete([]).unwrap_err(),
            SignatureError::WrongNumberTypeArgs(0, 1)
        );
        // Too many arguments:
        assert_eq!(
            def.instantiate_concrete([
                TypeArg::Type(ClassicType::F64.into()),
                TypeArg::Type(ClassicType::F64.into()),
            ])
            .unwrap_err(),
            SignatureError::WrongNumberTypeArgs(2, 1)
        );
    }
}
