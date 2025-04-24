//! Builder trait for array operations in the dataflow graph.

use crate::std_extensions::collections::array::{new_array_op, ArrayOpDef};
use crate::{
    builder::{BuildError, Dataflow},
    extension::simple_op::HasConcrete as _,
    types::Type,
    Wire,
};
use itertools::Itertools as _;

/// Trait for building array operations in a dataflow graph.
pub trait ArrayOpBuilder: Dataflow {
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
        let inputs = values.into_iter().collect_vec();
        let [out] = self
            .add_dataflow_op(new_array_op(elem_ty, inputs.len() as u64), inputs)?
            .outputs_arr();
        Ok(out)
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
    /// The wire representing the value at the specified index in the array.
    fn add_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<Wire, BuildError> {
        let op = ArrayOpDef::get.instantiate(&[size.into(), elem_ty.into()])?;
        let [out] = self.add_dataflow_op(op, vec![input, index])?.outputs_arr();
        Ok(out)
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
        let op = ArrayOpDef::set.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        let op = ArrayOpDef::swap.instantiate(&[size.into(), elem_ty.into()])?;
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
        let op = ArrayOpDef::pop_left.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_array_pop_right(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        let op = ArrayOpDef::pop_right.instantiate(&[size.into(), elem_ty.into()])?;
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
    fn add_array_discard_empty(&mut self, elem_ty: Type, input: Wire) -> Result<(), BuildError> {
        self.add_dataflow_op(
            ArrayOpDef::discard_empty
                .instantiate(&[elem_ty.into()])
                .unwrap(),
            [input],
        )?;
        Ok(())
    }
}

impl<D: Dataflow> ArrayOpBuilder for D {}

#[cfg(test)]
mod test {
    use crate::extension::prelude::PRELUDE_ID;
    use crate::extension::ExtensionSet;
    use crate::std_extensions::collections::array::{self, array_type};
    use crate::{
        builder::{DFGBuilder, HugrBuilder},
        extension::prelude::{either_type, option_type, usize_t, ConstUsize, UnwrapBuilder as _},
        types::Signature,
        Hugr,
    };
    use rstest::rstest;

    use super::*;

    #[rstest::fixture]
    #[default(DFGBuilder<Hugr>)]
    fn all_array_ops<B: Dataflow>(
        #[default(DFGBuilder::new(Signature::new_endo(Type::EMPTY_TYPEROW)
            .with_extension_delta(ExtensionSet::from_iter([
                PRELUDE_ID,
                array::EXTENSION_ID
        ]))).unwrap())]
        mut builder: B,
    ) -> B {
        let us0 = builder.add_load_value(ConstUsize::new(0));
        let us1 = builder.add_load_value(ConstUsize::new(1));
        let us2 = builder.add_load_value(ConstUsize::new(2));
        let arr = builder.add_new_array(usize_t(), [us1, us2]).unwrap();
        let [arr] = {
            let r = builder.add_array_swap(usize_t(), 2, arr, us0, us1).unwrap();
            let res_sum_ty = {
                let array_type = array_type(2, usize_t());
                either_type(array_type.clone(), array_type)
            };
            builder.build_unwrap_sum(1, res_sum_ty, r).unwrap()
        };

        let [elem_0] = {
            let r = builder.add_array_get(usize_t(), 2, arr, us0).unwrap();
            builder
                .build_unwrap_sum(1, option_type(usize_t()), r)
                .unwrap()
        };

        let [_elem_1, arr] = {
            let r = builder
                .add_array_set(usize_t(), 2, arr, us1, elem_0)
                .unwrap();
            let res_sum_ty = {
                let row = vec![usize_t(), array_type(2, usize_t())];
                either_type(row.clone(), row)
            };
            builder.build_unwrap_sum(1, res_sum_ty, r).unwrap()
        };

        let [_elem_left, arr] = {
            let r = builder.add_array_pop_left(usize_t(), 2, arr).unwrap();
            builder
                .build_unwrap_sum(1, option_type(vec![usize_t(), array_type(1, usize_t())]), r)
                .unwrap()
        };
        let [_elem_right, arr] = {
            let r = builder.add_array_pop_right(usize_t(), 1, arr).unwrap();
            builder
                .build_unwrap_sum(1, option_type(vec![usize_t(), array_type(0, usize_t())]), r)
                .unwrap()
        };

        builder.add_array_discard_empty(usize_t(), arr).unwrap();
        builder
    }

    #[rstest]
    fn build_all_ops(all_array_ops: DFGBuilder<Hugr>) {
        all_array_ops.finish_hugr().unwrap();
    }
}
