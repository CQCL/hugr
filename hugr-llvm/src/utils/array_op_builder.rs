use hugr_core::{
    builder::{BuildError, Dataflow},
    extension::{
        prelude::{new_array_op, ArrayOpDef},
        simple_op::HasConcrete as _,
    },
    types::Type,
    Wire,
};
use itertools::Itertools as _;

pub trait ArrayOpBuilder: Dataflow {
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

    fn add_array_get(
        &mut self,
        elem_ty: Type,
        size: u64,
        input: Wire,
        index: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError.
        let op = ArrayOpDef::get
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        let [out] = self.add_dataflow_op(op, vec![input, index])?.outputs_arr();
        Ok(out)
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
        let op = ArrayOpDef::set
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
        let op = ArrayOpDef::swap
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
        let op = ArrayOpDef::pop_left
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
        let op = ArrayOpDef::pop_right
            .instantiate(&[size.into(), elem_ty.into()])
            .unwrap();
        Ok(self.add_dataflow_op(op, vec![input])?.out_wire(0))
    }

    fn add_array_discard_empty(&mut self, elem_ty: Type, input: Wire) -> Result<(), BuildError> {
        // TODO Add an OpLoadError variant to BuildError
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
pub mod test {
    use hugr_core::{
        builder::{DFGBuilder, HugrBuilder},
        extension::{
            prelude::{
                array_type, either_type, option_type, ConstUsize, UnwrapBuilder as _, USIZE_T,
            },
            PRELUDE_REGISTRY,
        },
        types::Signature,
        Hugr,
    };
    use rstest::rstest;

    use super::*;

    #[rstest::fixture]
    #[default(DFGBuilder<Hugr>)]
    pub fn all_array_ops<B: Dataflow>(
        #[default(DFGBuilder::new(Signature::new_endo(Type::EMPTY_TYPEROW)).unwrap())]
        mut builder: B,
    ) -> B {
        let us0 = builder.add_load_value(ConstUsize::new(0));
        let us1 = builder.add_load_value(ConstUsize::new(1));
        let us2 = builder.add_load_value(ConstUsize::new(2));
        let arr = builder.add_new_array(USIZE_T, [us1, us2]).unwrap();
        let [arr] = {
            let r = builder.add_array_swap(USIZE_T, 2, arr, us0, us1).unwrap();
            let res_sum_ty = {
                let array_type = array_type(2, USIZE_T);
                either_type(array_type.clone(), array_type)
            };
            builder
                .build_unwrap_sum(&PRELUDE_REGISTRY, 1, res_sum_ty, r)
                .unwrap()
        };

        let [elem_0] = {
            let r = builder.add_array_get(USIZE_T, 2, arr, us0).unwrap();
            builder
                .build_unwrap_sum(&PRELUDE_REGISTRY, 1, option_type(USIZE_T), r)
                .unwrap()
        };

        let [_elem_1, arr] = {
            let r = builder.add_array_set(USIZE_T, 2, arr, us1, elem_0).unwrap();
            let res_sum_ty = {
                let row = vec![USIZE_T, array_type(2, USIZE_T)];
                either_type(row.clone(), row)
            };
            builder
                .build_unwrap_sum(&PRELUDE_REGISTRY, 1, res_sum_ty, r)
                .unwrap()
        };

        let [_elem_left, arr] = {
            let r = builder.add_array_pop_left(USIZE_T, 2, arr).unwrap();
            builder
                .build_unwrap_sum(
                    &PRELUDE_REGISTRY,
                    1,
                    option_type(vec![USIZE_T, array_type(1, USIZE_T)]),
                    r,
                )
                .unwrap()
        };
        let [_elem_right, arr] = {
            let r = builder.add_array_pop_right(USIZE_T, 1, arr).unwrap();
            builder
                .build_unwrap_sum(
                    &PRELUDE_REGISTRY,
                    1,
                    option_type(vec![USIZE_T, array_type(0, USIZE_T)]),
                    r,
                )
                .unwrap()
        };

        builder.add_array_discard_empty(USIZE_T, arr).unwrap();
        builder
    }

    #[rstest]
    fn build_all_ops(all_array_ops: DFGBuilder<Hugr>) {
        all_array_ops.finish_hugr(&PRELUDE_REGISTRY).unwrap();
    }
}
