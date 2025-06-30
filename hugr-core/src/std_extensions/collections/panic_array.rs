//! A version of the standard fixed-length array extension that includes unsafe
//! operations that may panic.

use std::sync::Arc;

use delegate::delegate;
use lazy_static::lazy_static;

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
    Array, ArrayKind, FROM, GenericArrayClone, GenericArrayCloneDef, GenericArrayConvert,
    GenericArrayConvertDef, GenericArrayDiscard, GenericArrayDiscardDef, GenericArrayOp,
    GenericArrayOpDef, GenericArrayRepeat, GenericArrayRepeatDef, GenericArrayScan,
    GenericArrayScanDef, GenericArrayValue, INTO,
};

/// Reported unique name of the panic array type.
pub const PANIC_ARRAY_TYPENAME: TypeName = TypeName::new_inline("panic_array");
/// Reported unique name of the panic array value.
pub const PANIC_ARRAY_VALUENAME: TypeName = TypeName::new_inline("panic_array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.panic_array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 1);

/// A linear, unsafe, fixed-length collection of values.
///
/// Panic arrays are linear, even if their elements are copyable.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct PanicArray;

impl ArrayKind for PanicArray {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = PANIC_ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = PANIC_ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&PANIC_ARRAY_TYPENAME).unwrap()
    }
}

/// Panic array operation definitions.
pub type PArrayOpDef = GenericArrayOpDef<PanicArray>;
/// Panic array clone operation definition.
pub type PArrayCloneDef = GenericArrayCloneDef<PanicArray>;
/// Panic array discard operation definition.
pub type PArrayDiscardDef = GenericArrayDiscardDef<PanicArray>;
/// Panic array repeat operation definition.
pub type PArrayRepeatDef = GenericArrayRepeatDef<PanicArray>;
/// Panic array scan operation definition.
pub type PArrayScanDef = GenericArrayScanDef<PanicArray>;
/// Panic array to default array conversion operation definition.
pub type PArrayToArrayDef = GenericArrayConvertDef<PanicArray, INTO, Array>;
/// Panic array from default array conversion operation definition.
pub type PArrayFromArrayDef = GenericArrayConvertDef<PanicArray, FROM, Array>;

/// Panic array operations.
pub type PArrayOp = GenericArrayOp<PanicArray>;
/// The array clone operation.
pub type PArrayClone = GenericArrayClone<PanicArray>;
/// The array discard operation.
pub type PArrayDiscard = GenericArrayDiscard<PanicArray>;
/// The array repeat operation.
pub type PArrayRepeat = GenericArrayRepeat<PanicArray>;
/// The array scan operation.
pub type PArrayScan = GenericArrayScan<PanicArray>;
/// The panic array to default array conversion operation.
pub type PArrayToArray = GenericArrayConvert<PanicArray, INTO, Array>;
/// The panic array from default array conversion operation.
pub type PArrayFromArray = GenericArrayConvert<PanicArray, FROM, Array>;

/// A panic array extension value.
pub type PArrayValue = GenericArrayValue<PanicArray>;

lazy_static! {
    /// Extension for panic array operations.
    pub static ref EXTENSION: Arc<Extension> = {
        Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
            extension.add_type(
                    PANIC_ARRAY_TYPENAME,
                    vec![ TypeParam::max_nat_type(), TypeBound::Any.into()],
                    "Fixed-length panic array".into(),
                    // Panic array is linear, even if the elements are copyable.
                    TypeDefBound::any(),
                    extension_ref,
                )
                .unwrap();

            PArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
            PArrayCloneDef::new().add_to_extension(extension, extension_ref).unwrap();
            PArrayDiscardDef::new().add_to_extension(extension, extension_ref).unwrap();
            PArrayRepeatDef::new().add_to_extension(extension, extension_ref).unwrap();
            PArrayScanDef::new().add_to_extension(extension, extension_ref).unwrap();
            PArrayToArrayDef::new().add_to_extension(extension, extension_ref).unwrap();
            PArrayFromArrayDef::new().add_to_extension(extension, extension_ref).unwrap();
        })
    };
}

#[typetag::serde(name = "PArrayValue")]
impl CustomConst for PArrayValue {
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

/// Gets the [`TypeDef`] for panic arrays. Note that instantiations are more easily
/// created via [`panic_array_type`] and [`panic_array_type_parametric`]
#[must_use]
pub fn panic_array_type_def() -> &'static TypeDef {
    PanicArray::type_def()
}

/// Instantiate a new panic array type given a size argument and element type.
///
/// This method is equivalent to [`panic_array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
#[must_use]
pub fn panic_array_type(size: u64, element_ty: Type) -> Type {
    PanicArray::ty(size, element_ty)
}

/// Instantiate a new panic array type given the size and element type parameters.
///
/// This is a generic version of [`panic_array_type`].
pub fn panic_array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    PanicArray::ty_parametric(size, element_ty)
}

/// Trait for building panic array operations in a dataflow graph.
pub trait PArrayOpBuilder: GenericArrayOpBuilder {
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
    fn add_new_panic_array(
        &mut self,
        elem_ty: Type,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.add_new_generic_array::<PanicArray>(elem_ty, values)
    }
    /// Adds an array unpack operation to the dataflow graph.
    ///
    /// This operation unpacks an array into individual elements.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array to unpack.
    ///
    /// # Errors
    ///
    /// If building the operation fails.
    ///
    /// # Returns
    ///
    /// A vector of wires representing the individual elements from the array.
    fn add_panic_array_unpack(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        self.add_generic_array_unpack::<PanicArray>(elem_ty, size, input)
    }
    /// Adds an array clone operation to the dataflow graph and return the wires
    /// representing the original and cloned array.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    ///
    /// # Errors
    ///
    /// If building the operation fails.
    ///
    /// # Returns
    ///
    /// The wires representing the original and cloned array.
    fn add_panic_array_clone(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_clone::<PanicArray>(elem_ty, size, input)
    }

    /// Adds an array discard operation to the dataflow graph.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    ///
    /// # Errors
    ///
    /// If building the operation fails.
    fn add_panic_array_discard(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_generic_array_discard::<PanicArray>(elem_ty, size, input)
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
    fn add_panic_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_get::<PanicArray>(elem_ty, size, input, index)
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
    fn add_panic_array_set(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_set::<PanicArray>(elem_ty, size, input, index, value)
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
    fn add_panic_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op =
            GenericArrayOpDef::<PanicArray>::swap.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_panic_array_pop_left(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_left::<PanicArray>(elem_ty, size, input)
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
    fn add_panic_array_pop_right(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_right::<PanicArray>(elem_ty, size, input)
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
    fn add_panic_array_discard_empty(
        &mut self,
        elem_ty: Type,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_generic_array_discard_empty::<Array>(elem_ty, input)
    }
}

impl<D: Dataflow> PArrayOpBuilder for D {}
