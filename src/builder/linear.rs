use itertools::Itertools;

use crate::ops::OpType;

use super::{BuildError, BuildHandle, Dataflow, Wire};

pub struct LinearBuilder<'a, T, const N: usize> {
    wires: [Wire; N],
    builder: &'a mut T,
}

impl<'a, T: Dataflow, const N: usize> LinearBuilder<'a, T, N> {
    #[inline]
    pub fn append(
        &mut self,
        op: impl Into<OpType>,
        indices: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<&mut Self, BuildError> {
        self.append_and_consume(op, indices, [])
    }

    #[inline]
    pub fn append_and_consume(
        &mut self,
        op: impl Into<OpType>,
        linear_indices: impl IntoIterator<Item = usize> + Clone,
        non_linear_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<&mut Self, BuildError> {
        let output_wires =
            self.append_with_outputs(op, linear_indices.clone(), non_linear_wires)?;

        for (ind, wire) in linear_indices.into_iter().zip(output_wires.into_iter()) {
            self.wires[ind] = wire;
        }

        Ok(self)
    }

    pub fn append_with_outputs(
        &mut self,
        op: impl Into<OpType>,
        linear_indices: impl IntoIterator<Item = usize>,
        non_linear_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<Vec<Wire>, BuildError> {
        let input_wires = linear_indices
            .into_iter()
            .map(|index| self.wires[index])
            .chain(non_linear_wires.into_iter());

        let op = self.builder.add_dataflow_op(op, input_wires)?;

        Ok(op.outputs().collect_vec())
    }

    #[inline]
    pub fn finish(self) -> [Wire; N] {
        self.wires
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{build_main, QB},
            BuildError, Dataflow, Wire,
        },
        ops::LeafOp,
        type_row,
        types::Signature,
    };

    #[test]
    fn simple_linear() -> Result<(), BuildError> {
        let build_res = build_main(
            Signature::new_df(type_row![QB, QB], type_row![QB, QB]),
            |mut f_build| {
                let wires: [Wire; 2] = f_build.input_wires_arr();

                let mut linear = LinearBuilder {
                    wires,
                    builder: &mut f_build,
                };
                linear
                    .append(LeafOp::H, [0])?
                    .append(LeafOp::CX, [0, 1])?
                    .append(LeafOp::CX, [1, 0])?;

                let outs = linear.finish();
                f_build.finish_with_outputs(outs)
            },
        );

        assert_matches!(build_res, Ok(_));

        crate::utils::test::viz_dotstr(&build_res?.dot_string());
        Ok(())
    }
}
