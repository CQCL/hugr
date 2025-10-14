//! Fixed-length array type and operations extension.

mod array_clone;
mod array_conversion;
mod array_discard;
mod array_kind;
mod array_op;
mod array_repeat;
mod array_scan;
mod array_value;
pub mod op_builder;

use std::sync::{Arc, LazyLock};

use delegate::delegate;

use crate::builder::{BuildError, Dataflow};
use crate::extension::resolution::{ExtensionResolutionError, WeakExtensionRegistry};
use crate::extension::simple_op::{HasConcrete, MakeOpDef, MakeRegisteredOp};
use crate::extension::{ExtensionId, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{CustomConst, ValueName};
use crate::ops::{ExtensionOp, OpName};
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, Type, TypeBound, TypeName};
use crate::{Extension, Wire};

pub use array_clone::{ARRAY_CLONE_OP_ID, GenericArrayClone, GenericArrayCloneDef};
pub use array_conversion::{Direction, FROM, GenericArrayConvert, GenericArrayConvertDef, INTO};
pub use array_discard::{ARRAY_DISCARD_OP_ID, GenericArrayDiscard, GenericArrayDiscardDef};
pub use array_kind::ArrayKind;
pub use array_op::{GenericArrayOp, GenericArrayOpDef};
pub use array_repeat::{ARRAY_REPEAT_OP_ID, GenericArrayRepeat, GenericArrayRepeatDef};
pub use array_scan::{ARRAY_SCAN_OP_ID, GenericArrayScan, GenericArrayScanDef};
pub use array_value::GenericArrayValue;

use op_builder::GenericArrayOpBuilder;

/// Reported unique name of the array type.
pub const ARRAY_TYPENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the array value.
pub const ARRAY_VALUENAME: TypeName = TypeName::new_inline("array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.array");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 1, 1);

/// A linear, fixed-length collection of values.
///
/// Arrays are linear, even if their elements are copyable.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct Array;

impl ArrayKind for Array {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&ARRAY_TYPENAME).unwrap()
    }
}

/// Array operation definitions.
pub type ArrayOpDef = GenericArrayOpDef<Array>;
/// Array clone operation definition.
pub type ArrayCloneDef = GenericArrayCloneDef<Array>;
/// Array discard operation definition.
pub type ArrayDiscardDef = GenericArrayDiscardDef<Array>;
/// Array repeat operation definition.
pub type ArrayRepeatDef = GenericArrayRepeatDef<Array>;
/// Array scan operation definition.
pub type ArrayScanDef = GenericArrayScanDef<Array>;

/// Array operations.
pub type ArrayOp = GenericArrayOp<Array>;
/// The array clone operation.
pub type ArrayClone = GenericArrayClone<Array>;
/// The array discard operation.
pub type ArrayDiscard = GenericArrayDiscard<Array>;
/// The array repeat operation.
pub type ArrayRepeat = GenericArrayRepeat<Array>;
/// The array scan operation.
pub type ArrayScan = GenericArrayScan<Array>;

/// An array extension value.
pub type ArrayValue = GenericArrayValue<Array>;

/// Extension for array operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                ARRAY_TYPENAME,
                vec![TypeParam::max_nat_type(), TypeBound::Linear.into()],
                "Fixed-length array".into(),
                // Default array is linear, even if the elements are copyable
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();

        ArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
        ArrayCloneDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        ArrayDiscardDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        ArrayRepeatDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        ArrayScanDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
    })
});

impl ArrayValue {
    /// Name of the constructor for creating constant arrays.
    pub(crate) const CTR_NAME: &'static str = "collections.array.const";
}

#[typetag::serde(name = "ArrayValue")]
impl CustomConst for ArrayValue {
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

/// Gets the [`TypeDef`] for arrays. Note that instantiations are more easily
/// created via [`array_type`] and [`array_type_parametric`]
#[must_use]
pub fn array_type_def() -> &'static TypeDef {
    Array::type_def()
}

/// Instantiate a new array type given a size argument and element type.
///
/// This method is equivalent to [`array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
#[must_use]
pub fn array_type(size: u64, element_ty: Type) -> Type {
    Array::ty(size, element_ty)
}

/// Instantiate a new array type given the size and element type parameters.
///
/// This is a generic version of [`array_type`].
pub fn array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    Array::ty_parametric(size, element_ty)
}

/// Name of the operation in the prelude for creating new arrays.
pub const NEW_ARRAY_OP_ID: OpName = OpName::new_inline("new_array");

/// Initialize a new array op of element type `element_ty` of length `size`
#[must_use]
pub fn new_array_op(element_ty: Type, size: u64) -> ExtensionOp {
    let op = ArrayOpDef::new_array.to_concrete(element_ty, size);
    op.to_extension_op().unwrap()
}

/// Trait for building array operations in a dataflow graph.
pub trait ArrayOpBuilder: GenericArrayOpBuilder {
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
    fn add_new_array(
        &mut self,
        elem_ty: Type,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.add_new_generic_array::<Array>(elem_ty, values)
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
    fn add_array_unpack(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        self.add_generic_array_unpack::<Array>(elem_ty, size, input)
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
    fn add_array_clone(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_clone::<Array>(elem_ty, size, input)
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
    fn add_array_discard(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_generic_array_discard::<Array>(elem_ty, size, input)
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
    fn add_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_get::<Array>(elem_ty, size, input, index)
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
    fn add_array_set(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_set::<Array>(elem_ty, size, input, index, value)
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
    fn add_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op = GenericArrayOpDef::<Array>::swap.instantiate(&[size.into(), elem_ty.into()])?;
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
        self.add_generic_array_pop_left::<Array>(elem_ty, size, input)
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
        self.add_generic_array_pop_right::<Array>(elem_ty, size, input)
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
        self.add_generic_array_discard_empty::<Array>(elem_ty, input)
    }
}

impl<D: Dataflow> ArrayOpBuilder for D {}

#[cfg(test)]
mod test {
    use crate::builder::{DFGBuilder, Dataflow, DataflowHugr, inout_sig};
    use crate::extension::prelude::qb_t;

    use super::{array_type, new_array_op};

    #[test]
    /// Test building a HUGR involving a `new_array` operation.
    fn test_new_array() {
        let mut b =
            DFGBuilder::new(inout_sig(vec![qb_t(), qb_t()], array_type(2, qb_t()))).unwrap();

        let [q1, q2] = b.input_wires_arr();

        let op = new_array_op(qb_t(), 2);

        let out = b.add_dataflow_op(op, [q1, q2]).unwrap();

        b.finish_hugr_with_outputs(out.outputs()).unwrap();
    }
}
