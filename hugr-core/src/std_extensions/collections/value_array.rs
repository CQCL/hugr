//! A version of the standard fixed-length array extension where arrays of copyable types
//! are copyable themselves.
//!
//! Supports all regular array operations apart from `clone` and `discard`.

use std::sync::{Arc, LazyLock};

use delegate::delegate;

use crate::builder::{BuildError, Dataflow};
use crate::extension::resolution::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::simple_op::{HasConcrete, MakeOpDef};
use crate::extension::{ExtensionId, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{CustomConst, ValueName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, Type, TypeBound, TypeName};
use crate::{Extension, Wire};

use super::array::op_builder::GenericArrayOpBuilder;
use super::array::{
    Array, ArrayKind, FROM, GenericArrayConvert, GenericArrayConvertDef, GenericArrayOp,
    GenericArrayOpDef, GenericArrayRepeat, GenericArrayRepeatDef, GenericArrayScan,
    GenericArrayScanDef, GenericArrayValue, INTO,
};

/// Reported unique name of the value array type.
pub const VALUE_ARRAY_TYPENAME: TypeName = TypeName::new_inline("value_array");
/// Reported unique name of the value array value.
pub const VALUE_ARRAY_VALUENAME: TypeName = TypeName::new_inline("value_array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_static_unchecked("collections.value_array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 1);

/// A fixed-length collection of values.
///
/// A value array inherits its linearity from its elements.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct ValueArray;

impl ArrayKind for ValueArray {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = VALUE_ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = VALUE_ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&VALUE_ARRAY_TYPENAME).unwrap()
    }

    fn build_clone<D: Dataflow>(
        _builder: &mut D,
        _elem_ty: Type,
        _size: u64,
        arr: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        Ok((arr, arr))
    }

    fn build_discard<D: Dataflow>(
        _builder: &mut D,
        _elem_ty: Type,
        _size: u64,
        _arr: Wire,
    ) -> Result<(), BuildError> {
        Ok(())
    }
}

/// Value array operation definitions.
pub type VArrayOpDef = GenericArrayOpDef<ValueArray>;
/// Value array repeat operation definition.
pub type VArrayRepeatDef = GenericArrayRepeatDef<ValueArray>;
/// Value array scan operation definition.
pub type VArrayScanDef = GenericArrayScanDef<ValueArray>;
/// Value array to default array conversion operation definition.
pub type VArrayToArrayDef = GenericArrayConvertDef<ValueArray, INTO, Array>;
/// Value array from default array conversion operation definition.
pub type VArrayFromArrayDef = GenericArrayConvertDef<ValueArray, FROM, Array>;

/// Value array operations.
pub type VArrayOp = GenericArrayOp<ValueArray>;
/// The value array repeat operation.
pub type VArrayRepeat = GenericArrayRepeat<ValueArray>;
/// The value array scan operation.
pub type VArrayScan = GenericArrayScan<ValueArray>;
/// The value array to default array conversion operation.
pub type VArrayToArray = GenericArrayConvert<ValueArray, INTO, Array>;
/// The value array from default array conversion operation.
pub type VArrayFromArray = GenericArrayConvert<ValueArray, FROM, Array>;

/// A value array extension value.
pub type VArrayValue = GenericArrayValue<ValueArray>;

/// Extension for value array operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                VALUE_ARRAY_TYPENAME,
                vec![TypeParam::max_nat_type(), TypeBound::Linear.into()],
                "Fixed-length value array".into(),
                // Value arrays are copyable iff their elements are
                TypeDefBound::from_params(vec![1]),
                extension_ref,
            )
            .unwrap();

        VArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
        VArrayRepeatDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        VArrayScanDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        VArrayToArrayDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        VArrayFromArrayDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
    })
});

#[typetag::serde(name = "VArrayValue")]
impl CustomConst for VArrayValue {
    delegate! {
        to self {
            fn name(&self) -> ValueName;
            fn validate(&self) -> Result<(), CustomCheckFailure>;
            fn update_extensions(
                &mut self,
                extensions: &WeakExtensionRegistry,
            ) -> Result<(), ExtensionResolutionError>;
            fn get_type(&self) -> Type;
        }
    }

    fn equal_consts(&self, other: &dyn CustomConst) -> bool {
        crate::ops::constant::downcast_equal_consts(self, other)
    }
}

/// Gets the [`TypeDef`] for value arrays. Note that instantiations are more easily
/// created via [`value_array_type`] and [`value_array_type_parametric`]
#[must_use]
pub fn value_array_type_def() -> &'static TypeDef {
    ValueArray::type_def()
}

/// Instantiate a new value array type given a size argument and element type.
///
/// This method is equivalent to [`value_array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
#[must_use]
pub fn value_array_type(size: u64, element_ty: Type) -> Type {
    ValueArray::ty(size, element_ty)
}

/// Instantiate a new value array type given the size and element type parameters.
///
/// This is a generic version of [`value_array_type`].
pub fn value_array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    ValueArray::ty_parametric(size, element_ty)
}

/// Trait for building value array operations in a dataflow graph.
pub trait VArrayOpBuilder: GenericArrayOpBuilder {
    /// Adds a new array operation to the dataflow graph and return the wire
    /// representing the new array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `values` - An iterator over the values to initialize the array with.
    ///
    /// # Errors
    ///
    /// If building the operation fails.
    ///
    /// # Returns
    ///
    /// The wire representing the new array.
    fn add_new_value_array(
        &mut self,
        elem_ty: Type,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.add_new_generic_array::<ValueArray>(elem_ty, values)
    }

    /// Adds an array get operation to the dataflow graph.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    /// * `index` - The wire representing the index to get.
    ///
    /// # Errors
    ///
    /// If building the operation fails.
    ///
    /// # Returns
    ///
    /// * The wire representing the value at the specified index in the array
    /// * The wire representing the array
    fn add_value_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_get::<ValueArray>(elem_ty, size, input, index)
    }

    /// Adds an array set operation to the dataflow graph.
    ///
    /// This operation sets the value at a specified index in the array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    /// * `index` - The wire representing the index to set.
    /// * `value` - The wire representing the value to set at the specified index.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    ///
    /// # Returns
    ///
    /// The wire representing the updated array after the set operation.
    fn add_value_array_set(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_set::<ValueArray>(elem_ty, size, input, index, value)
    }

    /// Adds an array swap operation to the dataflow graph.
    ///
    /// This operation swaps the values at two specified indices in the array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    /// * `index1` - The wire representing the first index to swap.
    /// * `index2` - The wire representing the second index to swap.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    ///
    /// # Returns
    ///
    /// The wire representing the updated array after the swap operation.
    fn add_value_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op =
            GenericArrayOpDef::<ValueArray>::swap.instantiate(&[size.into(), elem_ty.into()])?;
        let [out] = self
            .add_dataflow_op(op, vec![input, index1, index2])?
            .outputs_arr();
        Ok(out)
    }

    /// Adds an array pop-left operation to the dataflow graph.
    ///
    /// This operation removes the leftmost element from the array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    ///
    /// # Returns
    ///
    /// The wire representing the Option<elemty, array<SIZE-1, elemty>>
    fn add_array_pop_left(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_left::<ValueArray>(elem_ty, size, input)
    }

    /// Adds an array pop-right operation to the dataflow graph.
    ///
    /// This operation removes the rightmost element from the array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    ///
    /// # Returns
    ///
    /// The wire representing the Option<elemty, array<SIZE-1, elemty>>
    fn add_array_pop_right(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_right::<ValueArray>(elem_ty, size, input)
    }

    /// Adds an operation to discard an empty array from the dataflow graph.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `input` - The wire representing the array.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    fn add_array_discard_empty(&mut self, elem_ty: Type, input: Wire) -> Result<(), BuildError> {
        self.add_generic_array_discard_empty::<ValueArray>(elem_ty, input)
    }
}

impl<D: Dataflow> VArrayOpBuilder for D {}
