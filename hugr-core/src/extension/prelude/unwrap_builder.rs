use std::iter;

use crate::{
    builder::{BuildError, BuildHandle, Dataflow, DataflowSubContainer, SubContainer},
    extension::{
        prelude::{ConstError, PANIC_OP_ID, PRELUDE_ID},
        ExtensionRegistry,
    },
    ops::handle::DataflowOpID,
    types::{SumType, Type, TypeArg, TypeRow},
    Wire,
};
use itertools::{zip_eq, Itertools as _};

/// Extend dataflow builders with methods for building unwrap operations.
pub trait UnwrapBuilder: Dataflow {
    /// Add a panic operation to the dataflow with the given error.
    fn add_panic(
        &mut self,
        reg: &ExtensionRegistry,
        err: ConstError,
        output_row: impl IntoIterator<Item = Type>,
        inputs: impl IntoIterator<Item = (Wire, Type)>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        let (input_wires, input_types): (Vec<_>, Vec<_>) = inputs.into_iter().unzip();
        let input_arg: TypeArg = input_types
            .into_iter()
            .map(<TypeArg as From<_>>::from)
            .collect_vec()
            .into();
        let output_arg: TypeArg = output_row
            .into_iter()
            .map(<TypeArg as From<_>>::from)
            .collect_vec()
            .into();
        let prelude = reg.get(&PRELUDE_ID).unwrap();
        let op = prelude.instantiate_extension_op(&PANIC_OP_ID, [input_arg, output_arg], reg)?;
        let err = self.add_load_value(err);
        self.add_dataflow_op(op, iter::once(err).chain(input_wires))
    }

    /// Build an unwrap operation for a sum type to extract the variant at the given tag
    /// or panic if the tag is not the expected value.
    fn build_unwrap_sum<const N: usize>(
        &mut self,
        reg: &ExtensionRegistry,
        tag: usize,
        sum_type: SumType,
        input: Wire,
    ) -> Result<[Wire; N], BuildError> {
        let variants: Vec<TypeRow> = (0..sum_type.num_variants())
            .map(|i| {
                let tr_rv = sum_type.get_variant(i).unwrap().to_owned();
                TypeRow::try_from(tr_rv)
            })
            .collect::<Result<_, _>>()?;

        // TODO don't panic if tag >= num_variants
        let output_row = variants.get(tag).unwrap();

        let mut conditional =
            self.conditional_builder((variants.clone(), input), [], output_row.clone())?;
        for (i, variant) in variants.iter().enumerate() {
            let mut case = conditional.case_builder(i)?;
            if i == tag {
                let outputs = case.input_wires();
                case.finish_with_outputs(outputs)?;
            } else {
                let output_row = output_row.iter().cloned();
                let inputs = zip_eq(case.input_wires(), variant.iter().cloned());
                let err =
                    ConstError::new(1, format!("Expected variant {} but got variant {}", tag, i));
                let outputs = case.add_panic(reg, err, output_row, inputs)?.outputs();
                case.finish_with_outputs(outputs)?;
            }
        }
        Ok(conditional.finish_sub_container()?.outputs_arr())
    }
}

impl<D: Dataflow> UnwrapBuilder for D {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{DFGBuilder, DataflowHugr},
        extension::{
            prelude::{option_type, BOOL_T},
            PRELUDE_REGISTRY,
        },
        types::Signature,
    };

    #[test]
    fn test_build_unwrap() {
        let mut builder =
            DFGBuilder::new(Signature::new(Type::from(option_type(BOOL_T)), BOOL_T).with_prelude())
                .unwrap();

        let [opt] = builder.input_wires_arr();

        let [res] = builder
            .build_unwrap_sum(&PRELUDE_REGISTRY, 1, option_type(BOOL_T), opt)
            .unwrap();
        builder.finish_prelude_hugr_with_outputs([res]).unwrap();
    }
}