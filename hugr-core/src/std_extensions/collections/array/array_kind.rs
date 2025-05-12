use std::sync::Arc;

use crate::std_extensions::collections::array::op_builder::GenericArrayOpBuilder;
use crate::{
    Extension, Wire,
    builder::{BuildError, Dataflow},
    extension::{ExtensionId, SignatureError, TypeDef},
    ops::constant::ValueName,
    types::{CustomType, Type, TypeArg, TypeName},
};

/// Trait capturing a concrete array implementation in an extension.
///
/// Array operations are generically defined over this trait so the different
/// array extensions can share parts of their implementation. See for example
/// [`GenericArrayOpDef`] or [`GenericArrayValue`]
///
/// Currently the available kinds of array are [`Array`] (the default one) and
/// [`ValueArray`].
///
/// [`GenericArrayOpDef`]: super::GenericArrayOpDef
/// [`GenericArrayValue`]: super::GenericArrayValue
/// [`Array`]: super::Array
/// [`ValueArray`]: crate::std_extensions::collections::value_array::ValueArray
pub trait ArrayKind:
    Clone
    + Copy
    + std::fmt::Debug
    + std::fmt::Display
    + Eq
    + PartialEq
    + Default
    + Send
    + Sync
    + 'static
{
    /// Identifier of the extension containing the array.
    const EXTENSION_ID: ExtensionId;

    /// Name of the array type.
    const TYPE_NAME: TypeName;

    /// Name of the array value.
    const VALUE_NAME: ValueName;

    /// Returns the extension containing the array.
    fn extension() -> &'static Arc<Extension>;

    /// Returns the definition for the array type.
    fn type_def() -> &'static TypeDef;

    /// Instantiates an array [`CustomType`] from its definition given a size and
    /// element type argument.
    fn instantiate_custom_ty(
        array_def: &TypeDef,
        size: impl Into<TypeArg>,
        element_ty: impl Into<TypeArg>,
    ) -> Result<CustomType, SignatureError> {
        array_def.instantiate(vec![size.into(), element_ty.into()])
    }

    /// Instantiates an array type from its definition given a size and element
    /// type argument.
    fn instantiate_ty(
        array_def: &TypeDef,
        size: impl Into<TypeArg>,
        element_ty: impl Into<TypeArg>,
    ) -> Result<Type, SignatureError> {
        Self::instantiate_custom_ty(array_def, size, element_ty).map(Into::into)
    }

    /// Instantiates an array [`CustomType`] given a size and element type argument.
    fn custom_ty(size: impl Into<TypeArg>, element_ty: impl Into<TypeArg>) -> CustomType {
        Self::instantiate_custom_ty(Self::type_def(), size, element_ty)
            .expect("array parameters are valid")
    }

    /// Instantiate a new array type given a size argument and element type.
    ///
    /// This method is equivalent to [`ArrayKind::ty_parametric`], but uses concrete
    /// arguments types to ensure no errors are possible.
    #[must_use]
    fn ty(size: u64, element_ty: Type) -> Type {
        Self::custom_ty(size, element_ty).into()
    }

    /// Instantiate a new array type given the size and element type parameters.
    ///
    /// This is a generic version of [`ArrayKind::ty`].
    fn ty_parametric(
        size: impl Into<TypeArg>,
        element_ty: impl Into<TypeArg>,
    ) -> Result<Type, SignatureError> {
        Self::instantiate_ty(Self::type_def(), size, element_ty)
    }

    /// Adds a operation to a dataflow graph that clones an array of copyable values.
    ///
    /// The default implementation uses the array clone operation.
    fn build_clone<D: Dataflow>(
        builder: &mut D,
        elem_ty: Type,
        size: u64,
        arr: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        builder.add_generic_array_clone::<Self>(elem_ty, size, arr)
    }

    /// Adds a operation to a dataflow graph that clones an array of copyable values.
    ///
    /// The default implementation uses the array clone operation.
    fn build_discard<D: Dataflow>(
        builder: &mut D,
        elem_ty: Type,
        size: u64,
        arr: Wire,
    ) -> Result<(), BuildError> {
        builder.add_generic_array_discard::<Self>(elem_ty, size, arr)
    }
}
