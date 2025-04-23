use hugr_core::std_extensions::collections::array::{
    Array, ArrayKind, GenericArrayClone, GenericArrayDiscard, GenericArrayOpDef,
};
use hugr_core::{
    builder::{BuildError, Dataflow},
    extension::simple_op::HasConcrete as _,
    types::Type,
    Wire,
};
use itertools::Itertools as _;

pub trait ArrayOpBuilder<AK: ArrayKind>: Dataflow {
    fn add_new_array(
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

    fn add_array_clone(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        let op = GenericArrayClone::<AK>::new(elem_ty, size).unwrap();
        let [arr1, arr2] = self.add_dataflow_op(op, vec![input])?.outputs_arr();
        Ok((arr1, arr2))
    }

    fn add_array_discard(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<(), BuildError> {
        let op = GenericArrayDiscard::<AK>::new(elem_ty, size).unwrap();
        let [] = self.add_dataflow_op(op, vec![input])?.outputs_arr();
        Ok(())
    }

    fn add_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<(Wire, Wire), BuildError> {
        // TODO Add an OpLoadError variant to BuildError.
        let op = GenericArrayOpDef::<AK>::get
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        let [out, arr] = self.add_dataflow_op(op, vec![input, index])?.outputs_arr();
        Ok((out, arr))
    }

    fn add_array_set(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
        value: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = GenericArrayOpDef::<AK>::set
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        let [out] = self
            .add_dataflow_op(op, vec![input, index, value])?
            .outputs_arr();
        Ok(out)
    }

    fn add_array_swap(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index1: Wire,
        index2: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = GenericArrayOpDef::<AK>::swap
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        let [out] = self
            .add_dataflow_op(op, vec![input, index1, index2])?
            .outputs_arr();
        Ok(out)
    }

    fn add_array_pop_left(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = GenericArrayOpDef::<AK>::pop_left
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        Ok(self.add_dataflow_op(op, vec![input])?.out_wire(0))
    }

    fn add_array_pop_right(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = GenericArrayOpDef::<AK>::pop_right
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        Ok(self.add_dataflow_op(op, vec![input])?.out_wire(0))
    }

    fn add_array_discard_empty(&mut self, elem_ty: Type, input: Wire) -> Result<(), BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        self.add_dataflow_op(
            GenericArrayOpDef::<AK>::discard_empty
                .instantiate(&[elem_ty.into()])
                .unwrap(),
            [input],
        )?;
        Ok(())
    }
}

impl<D: Dataflow> ArrayOpBuilder<Array> for D {}

#[cfg(test)]
pub mod test {
    use hugr_core::extension::prelude::PRELUDE_ID;
    use hugr_core::extension::ExtensionSet;
    use hugr_core::std_extensions::collections::array::{self, array_type};
    use hugr_core::{
        builder::{DFGBuilder, HugrBuilder},
        extension::prelude::{either_type, option_type, usize_t, ConstUsize, UnwrapBuilder as _},
        types::Signature,
        Hugr,
    };
    use rstest::rstest;

    use super::*;

    #[rstest::fixture]
    #[default(DFGBuilder<Hugr>)]
    pub fn all_array_ops<B: Dataflow>(
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

        let ([elem_0], arr) = {
            let (r, arr) = builder.add_array_get(usize_t(), 2, arr, us0).unwrap();
            (
                builder
                    .build_unwrap_sum(1, option_type(usize_t()), r)
                    .unwrap(),
                arr,
            )
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
