//! A version of the standard fixed-length array extension that includes unsafe
//! operations for borrowing and returning that may panic.

use std::sync::{self, Arc, LazyLock};

use delegate::delegate;

use crate::extension::{ExtensionId, SignatureError, TypeDef, TypeDefBound};
use crate::ops::constant::{CustomConst, ValueName};
use crate::type_row;
use crate::types::type_param::{TypeArg, TypeParam};
use crate::types::{CustomCheckFailure, Term, Type, TypeBound, TypeName};
use crate::{Extension, Wire};
use crate::{
    builder::{BuildError, Dataflow},
    extension::SignatureFunc,
};
use crate::{
    extension::simple_op::{HasConcrete, HasDef, MakeExtensionOp, MakeOpDef, MakeRegisteredOp},
    ops::ExtensionOp,
};
use crate::{
    extension::{
        OpDef,
        prelude::usize_t,
        resolution::{ExtensionResolutionError, WeakExtensionRegistry},
        simple_op::{OpLoadError, try_from_name},
    },
    ops::OpName,
    types::{FuncValueType, PolyFuncTypeRV},
};

use super::array::op_builder::GenericArrayOpBuilder;
use super::array::{
    Array, ArrayKind, FROM, GenericArrayClone, GenericArrayCloneDef, GenericArrayConvert,
    GenericArrayConvertDef, GenericArrayDiscard, GenericArrayDiscardDef, GenericArrayOp,
    GenericArrayOpDef, GenericArrayRepeat, GenericArrayRepeatDef, GenericArrayScan,
    GenericArrayScanDef, GenericArrayValue, INTO,
};

/// Reported unique name of the borrow array type.
pub const BORROW_ARRAY_TYPENAME: TypeName = TypeName::new_inline("borrow_array");
/// Reported unique name of the borrow array value.
pub const BORROW_ARRAY_VALUENAME: TypeName = TypeName::new_inline("borrow_array");
/// Reported unique name of the extension
pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("collections.borrow_arr");
/// Extension version.
pub const VERSION: semver::Version = semver::Version::new(0, 2, 0);

/// A linear, unsafe, fixed-length collection of values.
///
/// Borrow arrays are linear, even if their elements are copyable.
#[derive(Clone, Copy, Debug, derive_more::Display, Eq, PartialEq, Default)]
pub struct BorrowArray;

impl ArrayKind for BorrowArray {
    const EXTENSION_ID: ExtensionId = EXTENSION_ID;
    const TYPE_NAME: TypeName = BORROW_ARRAY_TYPENAME;
    const VALUE_NAME: ValueName = BORROW_ARRAY_VALUENAME;

    fn extension() -> &'static Arc<Extension> {
        &EXTENSION
    }

    fn type_def() -> &'static TypeDef {
        EXTENSION.get_type(&BORROW_ARRAY_TYPENAME).unwrap()
    }
}

/// Borrow array operation definitions.
pub type BArrayOpDef = GenericArrayOpDef<BorrowArray>;
/// Borrow array clone operation definition.
pub type BArrayCloneDef = GenericArrayCloneDef<BorrowArray>;
/// Borrow array discard operation definition.
pub type BArrayDiscardDef = GenericArrayDiscardDef<BorrowArray>;
/// Borrow array repeat operation definition.
pub type BArrayRepeatDef = GenericArrayRepeatDef<BorrowArray>;
/// Borrow array scan operation definition.
pub type BArrayScanDef = GenericArrayScanDef<BorrowArray>;
/// Borrow array to default array conversion operation definition.
pub type BArrayToArrayDef = GenericArrayConvertDef<BorrowArray, INTO, Array>;
/// Borrow array from default array conversion operation definition.
pub type BArrayFromArrayDef = GenericArrayConvertDef<BorrowArray, FROM, Array>;

/// Borrow array operations.
pub type BArrayOp = GenericArrayOp<BorrowArray>;
/// The borrow array clone operation.
pub type BArrayClone = GenericArrayClone<BorrowArray>;
/// The borrow array discard operation.
pub type BArrayDiscard = GenericArrayDiscard<BorrowArray>;
/// The borrow array repeat operation.
pub type BArrayRepeat = GenericArrayRepeat<BorrowArray>;
/// The borrow array scan operation.
pub type BArrayScan = GenericArrayScan<BorrowArray>;
/// The borrow array to default array conversion operation.
pub type BArrayToArray = GenericArrayConvert<BorrowArray, INTO, Array>;
/// The borrow array from default array conversion operation.
pub type BArrayFromArray = GenericArrayConvert<BorrowArray, FROM, Array>;

/// A borrow array extension value.
pub type BArrayValue = GenericArrayValue<BorrowArray>;

#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    PartialEq,
    Eq,
    strum::EnumIter,
    strum::IntoStaticStr,
    strum::EnumString,
)]
#[allow(non_camel_case_types, missing_docs)]
#[non_exhaustive]
pub enum BArrayUnsafeOpDef {
    /// `borrow<size, elem_ty>: borrow_array<size, elem_ty>, index -> elem_ty, borrow_array<size, elem_ty>`
    borrow,
    /// `return<size, elem_ty>: borrow_array<size, elem_ty>, index, elem_ty -> borrow_array<size, elem_ty>`
    #[strum(serialize = "return")]
    r#return,
    /// `discard_all_borrowed<size, elem_ty>: borrow_array<size, elem_ty> -> ()`
    discard_all_borrowed,
    /// `new_all_borrowed<size, elem_ty>: () -> borrow_array<size, elem_ty>`
    new_all_borrowed,
    /// is_borrowed<N, T>: borrow_array<N, T>, usize -> bool, borrow_array<N, T>
    is_borrowed,
}

impl BArrayUnsafeOpDef {
    /// Instantiate a new unsafe borrow array operation with the given element type and array size.
    #[must_use]
    pub fn to_concrete(self, elem_ty: Type, size: u64) -> BArrayUnsafeOp {
        BArrayUnsafeOp {
            def: self,
            elem_ty,
            size,
        }
    }

    fn signature_from_def(&self, def: &TypeDef, _: &sync::Weak<Extension>) -> SignatureFunc {
        let size_var = TypeArg::new_var_use(0, TypeParam::max_nat_type());
        let elem_ty_var = Type::new_var_use(1, TypeBound::Linear);
        let array_ty: Type = def
            .instantiate(vec![size_var, elem_ty_var.clone().into()])
            .unwrap()
            .into();

        let params = vec![TypeParam::max_nat_type(), TypeBound::Linear.into()];

        let usize_t: Type = usize_t();

        match self {
            Self::borrow => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(vec![array_ty.clone(), usize_t], vec![array_ty, elem_ty_var]),
            ),
            Self::r#return => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(
                    vec![array_ty.clone(), usize_t, elem_ty_var.clone()],
                    vec![array_ty],
                ),
            ),
            Self::discard_all_borrowed => {
                PolyFuncTypeRV::new(params, FuncValueType::new(vec![array_ty], type_row![]))
            }
            Self::new_all_borrowed => {
                PolyFuncTypeRV::new(params, FuncValueType::new(type_row![], vec![array_ty]))
            }
            Self::is_borrowed => PolyFuncTypeRV::new(
                params,
                FuncValueType::new(
                    vec![array_ty.clone(), usize_t],
                    vec![array_ty, crate::extension::prelude::bool_t()],
                ),
            ),
        }
        .into()
    }
}

impl MakeOpDef for BArrayUnsafeOpDef {
    fn opdef_id(&self) -> OpName {
        <&'static str>::from(self).into()
    }

    fn from_def(op_def: &OpDef) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        try_from_name(op_def.name(), op_def.extension_id())
    }

    fn init_signature(&self, extension_ref: &sync::Weak<Extension>) -> SignatureFunc {
        self.signature_from_def(
            EXTENSION.get_type(&BORROW_ARRAY_TYPENAME).unwrap(),
            extension_ref,
        )
    }

    fn extension_ref(&self) -> sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }

    fn extension(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn description(&self) -> String {
        match self {
            Self::borrow => {
                "Take an element from a borrow array (panicking if it was already taken before)"
            }
            Self::r#return => {
                "Put an element into a borrow array (panicking if there is an element already)"
            }
            Self::discard_all_borrowed => {
                "Discard a borrow array where all elements have been borrowed"
            }
            Self::new_all_borrowed => "Create a new borrow array that contains no elements",
            Self::is_borrowed => "Test whether an element in a borrow array has been borrowed",
        }
        .into()
    }

    // This method is re-defined here to avoid recursive loops initializing the extension.
    fn add_to_extension(
        &self,
        extension: &mut Extension,
        extension_ref: &sync::Weak<Extension>,
    ) -> Result<(), crate::extension::ExtensionBuildError> {
        let sig = self.signature_from_def(
            extension.get_type(&BORROW_ARRAY_TYPENAME).unwrap(),
            extension_ref,
        );
        let def = extension.add_op(self.opdef_id(), self.description(), sig, extension_ref)?;

        self.post_opdef(def);

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Concrete array operation.
pub struct BArrayUnsafeOp {
    /// The operation definition.
    pub def: BArrayUnsafeOpDef,
    /// The element type of the array.
    pub elem_ty: Type,
    /// The size of the array.
    pub size: u64,
}

impl MakeExtensionOp for BArrayUnsafeOp {
    fn op_id(&self) -> OpName {
        self.def.opdef_id()
    }

    fn from_extension_op(ext_op: &ExtensionOp) -> Result<Self, OpLoadError>
    where
        Self: Sized,
    {
        let def = BArrayUnsafeOpDef::from_def(ext_op.def())?;
        def.instantiate(ext_op.args())
    }

    fn type_args(&self) -> Vec<TypeArg> {
        vec![self.size.into(), self.elem_ty.clone().into()]
    }
}

impl HasDef for BArrayUnsafeOp {
    type Def = BArrayUnsafeOpDef;
}

impl HasConcrete for BArrayUnsafeOpDef {
    type Concrete = BArrayUnsafeOp;

    fn instantiate(&self, type_args: &[TypeArg]) -> Result<Self::Concrete, OpLoadError> {
        match type_args {
            [Term::BoundedNat(n), Term::Runtime(ty)] => Ok(self.to_concrete(ty.clone(), *n)),
            _ => Err(SignatureError::InvalidTypeArgs.into()),
        }
    }
}

impl MakeRegisteredOp for BArrayUnsafeOp {
    fn extension_id(&self) -> ExtensionId {
        EXTENSION_ID.clone()
    }

    fn extension_ref(&self) -> sync::Weak<Extension> {
        Arc::downgrade(&EXTENSION)
    }
}

/// Extension for borrow array operations.
pub static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(|| {
    Extension::new_arc(EXTENSION_ID, VERSION, |extension, extension_ref| {
        extension
            .add_type(
                BORROW_ARRAY_TYPENAME,
                vec![TypeParam::max_nat_type(), TypeBound::Linear.into()],
                "Fixed-length borrow array".into(),
                // Borrow array is linear, even if the elements are copyable.
                TypeDefBound::any(),
                extension_ref,
            )
            .unwrap();

        BArrayOpDef::load_all_ops(extension, extension_ref).unwrap();
        BArrayCloneDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        BArrayDiscardDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        BArrayRepeatDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        BArrayScanDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        BArrayToArrayDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();
        BArrayFromArrayDef::new()
            .add_to_extension(extension, extension_ref)
            .unwrap();

        BArrayUnsafeOpDef::load_all_ops(extension, extension_ref).unwrap();
    })
});

#[typetag::serde(name = "BArrayValue")]
impl CustomConst for BArrayValue {
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

/// Gets the [`TypeDef`] for borrow arrays. Note that instantiations are more easily
/// created via [`borrow_array_type`] and [`borrow_array_type_parametric`]
#[must_use]
pub fn borrow_array_type_def() -> &'static TypeDef {
    BorrowArray::type_def()
}

/// Instantiate a new borrow array type given a size argument and element type.
///
/// This method is equivalent to [`borrow_array_type_parametric`], but uses concrete
/// arguments types to ensure no errors are possible.
#[must_use]
pub fn borrow_array_type(size: u64, element_ty: Type) -> Type {
    BorrowArray::ty(size, element_ty)
}

/// Instantiate a new borrow array type given the size and element type parameters.
///
/// This is a generic version of [`borrow_array_type`].
pub fn borrow_array_type_parametric(
    size: impl Into<TypeArg>,
    element_ty: impl Into<TypeArg>,
) -> Result<Type, SignatureError> {
    BorrowArray::ty_parametric(size, element_ty)
}

/// Trait for building borrow array operations in a dataflow graph.
pub trait BArrayOpBuilder: GenericArrayOpBuilder {
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
    fn add_new_borrow_array(
        &mut self,
        elem_ty: Type,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        self.add_new_generic_array::<BorrowArray>(elem_ty, values)
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
    fn add_borrow_array_unpack(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        self.add_generic_array_unpack::<BorrowArray>(elem_ty, size, input)
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
    fn add_borrow_array_clone(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_clone::<BorrowArray>(elem_ty, size, input)
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
    fn add_borrow_array_discard(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_generic_array_discard::<BorrowArray>(elem_ty, size, input)
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
    fn add_borrow_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        self.add_generic_array_get::<BorrowArray>(elem_ty, size, input, index)
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
    fn add_borrow_array_set(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_set::<BorrowArray>(elem_ty, size, input, index, value)
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
    fn add_borrow_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op =
            GenericArrayOpDef::<BorrowArray>::swap.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_borrow_array_pop_left(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_left::<BorrowArray>(elem_ty, size, input)
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
    fn add_borrow_array_pop_right(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        self.add_generic_array_pop_right::<BorrowArray>(elem_ty, size, input)
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
    fn add_borrow_array_discard_empty(
        &mut self,
        elem_ty: Type,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_generic_array_discard_empty::<BorrowArray>(elem_ty, input)
    }

    /// Adds a borrow array borrow operation to the dataflow graph.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    /// * `index` - The wire representing the index to get.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The wire representing the updated array with the element marked as borrowed.
    /// * The wire representing the borrowed element at the specified index.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    fn add_borrow_array_borrow(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        let op = BArrayUnsafeOpDef::borrow.instantiate(&[size.into(), elem_ty.into()])?;
        let [arr, out] = self
            .add_dataflow_op(op.to_extension_op().unwrap(), vec![input, index])?
            .outputs_arr();
        Ok((arr, out))
    }

    /// Adds a borrow array put operation to the dataflow graph.
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
    fn add_borrow_array_return(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        let op = BArrayUnsafeOpDef::r#return.instantiate(&[size.into(), elem_ty.into()])?;
        let [arr] = self
            .add_dataflow_op(op.to_extension_op().unwrap(), vec![input, index, value])?
            .outputs_arr();
        Ok(arr)
    }

    /// Adds an operation to discard a borrow array where all elements have been borrowed.
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
    fn add_discard_all_borrowed(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        let op =
            BArrayUnsafeOpDef::discard_all_borrowed.instantiate(&[size.into(), elem_ty.into()])?;
        self.add_dataflow_op(op.to_extension_op().unwrap(), vec![input])?;
        Ok(())
    }

    /// Adds an operation to create a new empty borrowed array in the dataflow graph.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    fn add_new_all_borrowed(&mut self, elem_ty: Type, size: u64) -> Result<Wire, BuildError> {
        let op = BArrayUnsafeOpDef::new_all_borrowed.instantiate(&[size.into(), elem_ty.into()])?;
        let [arr] = self
            .add_dataflow_op(op.to_extension_op().unwrap(), vec![])?
            .outputs_arr();
        Ok(arr)
    }

    /// Adds an operation to test whether an element in a borrow array has been borrowed.
    ///
    /// # Arguments
    ///
    /// * `elem_ty` - The type of the elements in the array.
    /// * `size` - The size of the array.
    /// * `input` - The wire representing the array.
    /// * `index` - The wire representing the index to test.
    ///
    /// # Errors
    ///
    /// Returns an error if building the operation fails.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The wire representing the updated array.
    /// * The wire representing the boolean result (true if borrowed).
    fn add_is_borrowed(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        let op = BArrayUnsafeOpDef::is_borrowed.instantiate(&[size.into(), elem_ty.into()])?;
        let [arr, is_borrowed] = self
            .add_dataflow_op(op.to_extension_op().unwrap(), vec![input, index])?
            .outputs_arr();
        Ok((arr, is_borrowed))
    }
}

impl<D: Dataflow> BArrayOpBuilder for D {}

#[cfg(test)]
mod test {
    use strum::IntoEnumIterator;

    use crate::{
        builder::{DFGBuilder, Dataflow, DataflowHugr as _},
        extension::prelude::{ConstUsize, qb_t, usize_t},
        ops::OpType,
        std_extensions::collections::borrow_array::{
            BArrayOpBuilder, BArrayUnsafeOp, BArrayUnsafeOpDef, borrow_array_type,
        },
        types::Signature,
    };

    #[test]
    fn test_borrow_array_unsafe_ops() {
        for def in BArrayUnsafeOpDef::iter() {
            let op = def.to_concrete(qb_t(), 2);
            let optype: OpType = op.clone().into();
            let new_op: BArrayUnsafeOp = optype.cast().unwrap();
            assert_eq!(new_op, op);
        }
    }

    #[test]
    fn test_borrow_and_return() {
        let size = 22;
        let elem_ty = qb_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let _ = {
            let mut builder = DFGBuilder::new(Signature::new_endo(vec![arr_ty.clone()])).unwrap();
            let idx1 = builder.add_load_value(ConstUsize::new(11));
            let idx2 = builder.add_load_value(ConstUsize::new(11));
            let [arr] = builder.input_wires_arr();
            let (arr_with_take, el) = builder
                .add_borrow_array_borrow(elem_ty.clone(), size, arr, idx1)
                .unwrap();
            let arr_with_put = builder
                .add_borrow_array_return(elem_ty, size, arr_with_take, idx2, el)
                .unwrap();
            builder.finish_hugr_with_outputs([arr_with_put]).unwrap()
        };
    }

    #[test]
    fn test_discard_all_borrowed() {
        let size = 1;
        let elem_ty = qb_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let _ = {
            let mut builder =
                DFGBuilder::new(Signature::new(vec![arr_ty.clone()], vec![qb_t()])).unwrap();
            let idx = builder.add_load_value(ConstUsize::new(0));
            let [arr] = builder.input_wires_arr();
            let (arr_with_borrowed, el) = builder
                .add_borrow_array_borrow(elem_ty.clone(), size, arr, idx)
                .unwrap();
            builder
                .add_discard_all_borrowed(elem_ty, size, arr_with_borrowed)
                .unwrap();
            builder.finish_hugr_with_outputs([el]).unwrap()
        };
    }

    #[test]
    fn test_new_all_borrowed() {
        let size = 5;
        let elem_ty = usize_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());
        let _ = {
            let mut builder =
                DFGBuilder::new(Signature::new(vec![], vec![arr_ty.clone()])).unwrap();
            let arr = builder.add_new_all_borrowed(elem_ty.clone(), size).unwrap();
            let idx = builder.add_load_value(ConstUsize::new(3));
            let val = builder.add_load_value(ConstUsize::new(202));
            let arr_with_put = builder
                .add_borrow_array_return(elem_ty, size, arr, idx, val)
                .unwrap();
            builder.finish_hugr_with_outputs([arr_with_put]).unwrap()
        };
    }
    #[test]
    fn test_is_borrowed() {
        let size = 4;
        let elem_ty = qb_t();
        let arr_ty = borrow_array_type(size, elem_ty.clone());

        let mut builder =
            DFGBuilder::new(Signature::new(vec![arr_ty.clone()], vec![qb_t(), arr_ty])).unwrap();
        let idx = builder.add_load_value(ConstUsize::new(2));
        let [arr] = builder.input_wires_arr();
        // Borrow the element at index 2
        let (arr_with_borrowed, qb) = builder
            .add_borrow_array_borrow(elem_ty.clone(), size, arr, idx)
            .unwrap();
        let (arr_after_check, _is_borrowed) = builder
            .add_is_borrowed(elem_ty.clone(), size, arr_with_borrowed, idx)
            .unwrap();
        builder
            .finish_hugr_with_outputs([qb, arr_after_check])
            .unwrap();
    }
}
