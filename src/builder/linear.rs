use crate::ops::OpType;

use super::{nodehandle::Outputs, BuildError, BuildHandle, Dataflow, Wire};

/// Builder to build linear regions of dataflow graphs
/// Appends operations to an array of incoming wires
pub struct LinearBuilder<'a, T: ?Sized, const N: usize> {
    wires: [Wire; N],
    builder: &'a mut T,
}

impl<'a, T: Dataflow + ?Sized, const N: usize> LinearBuilder<'a, T, N> {
    /// Construct a new LinearBuilder from an array of incoming wires and the
    /// builder for the graph
    pub fn new(wires: [Wire; N], builder: &'a mut T) -> Self {
        Self { wires, builder }
    }

    #[inline]
    /// Append a linear op to the wires in the array with given `indices`.
    /// The outputs of the operation become the new wires at those indices.
    /// Returns a handle to self to allow chaining.
    pub fn append(
        &mut self,
        op: impl Into<OpType>,
        indices: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<&mut Self, BuildError> {
        self.append_and_consume(op, indices, [])
    }

    #[inline]
    /// Append a linear op to the wires in the array with given `linear_indices`
    /// and wire in the `non_linear_wires` as the remaining inputs.
    /// Returns a handle to self to allow chaining.
    pub fn append_and_consume(
        &mut self,
        op: impl Into<OpType>,
        linear_indices: impl IntoIterator<Item = usize> + Clone,
        non_linear_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<&mut Self, BuildError> {
        self.append_with_outputs(op, linear_indices, non_linear_wires)?;

        Ok(self)
    }

    /// Append a linear op to the wires in the array with given `linear_indices`
    /// and wire in the `non_linear_wires` as the remaining inputs.
    /// Returns any non-linear outputs.
    /// Assumes linear wires are the first inputs and first outputs.
    pub fn append_with_outputs(
        &mut self,
        op: impl Into<OpType>,
        linear_indices: impl IntoIterator<Item = usize> + Clone,
        non_linear_wires: impl IntoIterator<Item = Wire>,
    ) -> Result<Outputs, BuildError> {
        let input_wires = linear_indices
            .clone()
            .into_iter()
            .map(|index| self.wires[index])
            .chain(non_linear_wires.into_iter());

        let mut output_wires = self.builder.add_dataflow_op(op, input_wires)?.outputs();

        // zip will leave all the non-linear output_wires
        // assumes first len(linear_indices) wires are linear
        for (ind, wire) in linear_indices.into_iter().zip(&mut output_wires) {
            self.wires[ind] = wire;
        }

        Ok(output_wires)
    }

    #[inline]
    /// Finish building the linear region and return the dangling wires
    /// corresponding to the initially provided wires.
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
            test::{build_main, BIT, F64, QB},
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

        Ok(())
    }

    #[test]
    fn with_nonlinear_and_outputs() -> Result<(), BuildError> {
        let build_res = build_main(
            Signature::new_df(type_row![QB, QB, F64], type_row![QB, QB, BIT]),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

                let mut linear = f_build.as_linear([q0, q1]);

                let measure_out = linear
                    .append(LeafOp::CX, [0, 1])?
                    .append_and_consume(LeafOp::RzF64, [0], [angle])?
                    .append_with_outputs(LeafOp::Measure, [0], [])?;

                let out_qbs = linear.finish();
                f_build.finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            },
        );

        assert_matches!(build_res, Ok(_));

        Ok(())
    }
}
