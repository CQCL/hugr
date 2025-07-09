//! Builder trait for array operations in the dataflow graph.

use crate::std_extensions::collections::array::GenericArrayOpDef;
use crate::std_extensions::collections::borrow_array::BorrowArray;
use crate::std_extensions::collections::value_array::ValueArray;
use crate::{
    Wire,
    builder::{BuildError, Dataflow},
    extension::simple_op::HasConcrete as _,
    types::Type,
};
use itertools::Itertools as _;

use super::{Array, ArrayKind, GenericArrayClone, GenericArrayDiscard};

use crate::extension::prelude::{
    ConstUsize, UnwrapBuilder as _, either_type, option_type, usize_t,
};

/// Trait for building array operations in a dataflow graph that are generic
/// over the concrete array implementation.
pub trait GenericArrayOpBuilder: Dataflow {
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
    fn add_new_generic_array<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        values: impl IntoIterator<Item = Wire>,
    ) -> Result<Wire, BuildError> {
        let inputs = values.into_iter().collect_vec();
        let [out] = self
            .add_dataflow_op(
                GenericArrayOpDef::<AK>::new_array.to_concrete(elem_ty, inputs.len() as u64),
                inputs,
            )?
            .outputs_arr();
        Ok(out)
    }

    /// Adds an array unpack operation to the dataflow graph.
    ///
    /// This operation unpacks an array into individual elements.
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
    /// A vector of wires representing the individual elements of the array.
    fn add_generic_array_unpack<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Vec<Wire>, BuildError> {
        let op = GenericArrayOpDef::<AK>::unpack.instantiate(&[size.into(), elem_ty.into()])?;
        Ok(self.add_dataflow_op(op, vec![input])?.outputs().collect())
    }
    /// Adds an array clone operation to the dataflow graph and return the wires
    /// representing the originala and cloned array.
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
    fn add_generic_array_clone<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        let op = GenericArrayClone::<AK>::new(elem_ty, size).unwrap();
        let [arr1, arr2] = self.add_dataflow_op(op, vec![input])?.outputs_arr();
        Ok((arr1, arr2))
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
    fn add_generic_array_discard<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        let op = GenericArrayDiscard::<AK>::new(elem_ty, size).unwrap();
        let [] = self.add_dataflow_op(op, vec![input])?.outputs_arr();
        Ok(())
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
    fn add_generic_array_get<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        let op = GenericArrayOpDef::<AK>::get.instantiate(&[size.into(), elem_ty.into()])?;
        let [out, arr] = self.add_dataflow_op(op, vec![input, index])?.outputs_arr();
        Ok((out, arr))
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
    fn add_generic_array_set<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        let op = GenericArrayOpDef::<AK>::set.instantiate(&[size.into(), elem_ty.into()])?;
        let [out] = self
            .add_dataflow_op(op, vec![input, index, value])?
            .outputs_arr();
        Ok(out)
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
    fn add_generic_array_swap<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op = GenericArrayOpDef::<AK>::swap.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_generic_array_pop_left<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        let op = GenericArrayOpDef::<AK>::pop_left.instantiate(&[size.into(), elem_ty.into()])?;
        Ok(self.add_dataflow_op(op, vec![input])?.out_wire(0))
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
    fn add_generic_array_pop_right<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        let op = GenericArrayOpDef::<AK>::pop_right.instantiate(&[size.into(), elem_ty.into()])?;
        Ok(self.add_dataflow_op(op, vec![input])?.out_wire(0))
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
    fn add_generic_array_discard_empty<AK: ArrayKind>(
        &mut self,
        elem_ty: Type,
        input: Wire,
    ) -> Result<(), BuildError> {
        self.add_dataflow_op(
            GenericArrayOpDef::<AK>::discard_empty
                .instantiate(&[elem_ty.into()])
                .unwrap(),
            [input],
        )?;
        Ok(())
    }
}

impl<D: Dataflow> GenericArrayOpBuilder for D {}

/// Helper function to build a Hugr that contains all basic array operations.
///
/// Generic over the concrete array implementation.
pub fn build_all_array_ops_generic<B: Dataflow, AK: ArrayKind>(mut builder: B) -> B {
    let us0 = builder.add_load_value(ConstUsize::new(0));
    let us1 = builder.add_load_value(ConstUsize::new(1));
    let us2 = builder.add_load_value(ConstUsize::new(2));
    let arr = builder
        .add_new_generic_array::<AK>(usize_t(), [us1, us2])
        .unwrap();

    // Add array unpack operation
    let [_us1, _us2] = builder
        .add_generic_array_unpack::<AK>(usize_t(), 2, arr)
        .unwrap()
        .try_into()
        .unwrap();

    let arr = builder
        .add_new_generic_array::<AK>(usize_t(), [us1, us2])
        .unwrap();
    let [arr] = {
        let r = builder
            .add_generic_array_swap::<AK>(usize_t(), 2, arr, us0, us1)
            .unwrap();
        let res_sum_ty = {
            let array_type = AK::ty(2, usize_t());
            either_type(array_type.clone(), array_type)
        };
        builder.build_unwrap_sum(1, res_sum_ty, r).unwrap()
    };

    let ([elem_0], arr) = {
        let (r, arr) = builder
            .add_generic_array_get::<AK>(usize_t(), 2, arr, us0)
            .unwrap();
        (
            builder
                .build_unwrap_sum(1, option_type(usize_t()), r)
                .unwrap(),
            arr,
        )
    };

    let [_elem_1, arr] = {
        let r = builder
            .add_generic_array_set::<AK>(usize_t(), 2, arr, us1, elem_0)
            .unwrap();
        let res_sum_ty = {
            let row = vec![usize_t(), AK::ty(2, usize_t())];
            either_type(row.clone(), row)
        };
        builder.build_unwrap_sum(1, res_sum_ty, r).unwrap()
    };

    let [_elem_left, arr] = {
        let r = builder
            .add_generic_array_pop_left::<AK>(usize_t(), 2, arr)
            .unwrap();
        builder
            .build_unwrap_sum(1, option_type(vec![usize_t(), AK::ty(1, usize_t())]), r)
            .unwrap()
    };
    let [_elem_right, arr] = {
        let r = builder
            .add_generic_array_pop_right::<AK>(usize_t(), 1, arr)
            .unwrap();
        builder
            .build_unwrap_sum(1, option_type(vec![usize_t(), AK::ty(0, usize_t())]), r)
            .unwrap()
    };

    builder
        .add_generic_array_discard_empty::<AK>(usize_t(), arr)
        .unwrap();
    builder
}

/// Helper function to build a Hugr that contains all basic array operations.
pub fn build_all_array_ops<B: Dataflow>(builder: B) -> B {
    build_all_array_ops_generic::<B, Array>(builder)
}

/// Helper function to build a Hugr that contains all basic array operations.
pub fn build_all_value_array_ops<B: Dataflow>(builder: B) -> B {
    build_all_array_ops_generic::<B, ValueArray>(builder)
}

/// Helper function to build a Hugr that contains all basic array operations.
pub fn build_all_borrow_array_ops<B: Dataflow>(builder: B) -> B {
    build_all_array_ops_generic::<B, BorrowArray>(builder)
}

/// Testing utilities to generate Hugrs that contain array operations.
#[cfg(test)]
mod test {
    use crate::builder::{DFGBuilder, HugrBuilder};
    use crate::types::Signature;

    use super::*;

    #[test]
    fn all_array_ops() {
        let sig = Signature::new_endo(Type::EMPTY_TYPEROW);
        let builder = DFGBuilder::new(sig).unwrap();
        build_all_array_ops(builder).finish_hugr().unwrap();
    }

    #[test]
    fn all_value_array_ops() {
        let sig = Signature::new_endo(Type::EMPTY_TYPEROW);
        let builder = DFGBuilder::new(sig).unwrap();
        build_all_value_array_ops(builder).finish_hugr().unwrap();
    }

    #[test]
    fn all_borrow_array_ops() {
        let sig = Signature::new_endo(Type::EMPTY_TYPEROW);
        let builder = DFGBuilder::new(sig).unwrap();
        build_all_borrow_array_ops(builder).finish_hugr().unwrap();
    }
}
