use std::collections::HashMap;

use thiserror::Error;

use crate::hugr::CircuitUnit;

use crate::ops::OpType;

use super::{BuildError, Dataflow};
use crate::Wire;

/// Builder to build regions of dataflow graphs that look like Circuits,
/// where some inputs of operations directly correspond to some outputs.
/// Allows appending operations by indexing a vector of input wires.
#[derive(Debug, PartialEq)]
pub struct CircuitBuilder<'a, T: ?Sized> {
    wires: Vec<Wire>,
    builder: &'a mut T,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Error in [`CircuitBuilder`]
pub enum CircuitBuildError {
    /// Invalid index for stored wires.
    #[error("Invalid wire index.")]
    InvalidWireIndex,
}

impl<'a, T: Dataflow + ?Sized> CircuitBuilder<'a, T> {
    /// Construct a new [`CircuitBuilder`] from a vector of incoming wires and the
    /// builder for the graph
    pub fn new(wires: Vec<Wire>, builder: &'a mut T) -> Self {
        Self { wires, builder }
    }

    /// Number of wires tracked, upper bound of valid wire indices
    #[must_use]
    pub fn n_wires(&self) -> usize {
        self.wires.len()
    }

    #[inline]
    /// Append an op to the wires in the inner vector with given `indices`.
    /// The outputs of the operation become the new wires at those indices.
    /// Only valid for operations that have the same input type row as output
    /// type row.
    /// Returns a handle to self to allow chaining.
    pub fn append(
        &mut self,
        op: impl Into<OpType>,
        indices: impl IntoIterator<Item = usize> + Clone,
    ) -> Result<&mut Self, BuildError> {
        self.append_and_consume(op, indices)
    }

    #[inline]
    /// The same as [`CircuitBuilder::append_with_outputs`] except it assumes no outputs and
    /// instead returns a reference to self to allow chaining.
    pub fn append_and_consume<A: Into<CircuitUnit>>(
        &mut self,
        op: impl Into<OpType>,
        inputs: impl IntoIterator<Item = A>,
    ) -> Result<&mut Self, BuildError> {
        self.append_with_outputs(op, inputs)?;

        Ok(self)
    }

    /// Append an `op` with some inputs being the stored wires.
    /// Any inputs of the form [`CircuitUnit::Linear`] are used to index the
    /// stored wires.
    /// The outputs at those indices are used to replace the stored wire.
    /// The remaining outputs are returned.
    ///
    /// # Errors
    ///
    /// This function will return an error if an index is invalid.
    pub fn append_with_outputs<A: Into<CircuitUnit>>(
        &mut self,
        op: impl Into<OpType>,
        inputs: impl IntoIterator<Item = A>,
    ) -> Result<Vec<Wire>, BuildError> {
        // map of linear port offset to wire vector index
        let mut linear_inputs = HashMap::new();

        let input_wires: Option<Vec<Wire>> = inputs
            .into_iter()
            .map(Into::into)
            .enumerate()
            .map(|(input_port, a_w): (usize, CircuitUnit)| match a_w {
                CircuitUnit::Wire(wire) => Some(wire),
                CircuitUnit::Linear(wire_index) => {
                    linear_inputs.insert(input_port, wire_index);
                    self.wires.get(wire_index).copied()
                }
            })
            .collect();

        let input_wires = input_wires.ok_or(CircuitBuildError::InvalidWireIndex)?;

        let output_wires = self
            .builder
            .add_dataflow_op(
                op, // TODO: Add resource param
                input_wires,
            )?
            .outputs();
        let nonlinear_outputs: Vec<Wire> = output_wires
            .enumerate()
            .filter_map(|(output_port, wire)| {
                if let Some(wire_index) = linear_inputs.remove(&output_port) {
                    // output at output_port replaces input wire from same port
                    self.wires[wire_index] = wire;
                    None
                } else {
                    Some(wire)
                }
            })
            .collect();

        Ok(nonlinear_outputs)
    }

    #[inline]
    /// Finish building the circuit region and return the dangling wires
    /// that correspond to the initially provided wires.
    pub fn finish(self) -> Vec<Wire> {
        self.wires
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;

    use crate::{
        builder::{
            test::{build_main, BIT, NAT, QB},
            Dataflow, DataflowSubContainer, Wire,
        },
        ops::{custom::OpaqueOp, LeafOp},
        type_row,
        types::AbstractSignature,
    };

    #[test]
    fn simple_linear() {
        let build_res = build_main(
            AbstractSignature::new_df(type_row![QB, QB], type_row![QB, QB]).pure(),
            |mut f_build| {
                let wires = f_build.input_wires().collect();

                let mut linear = CircuitBuilder {
                    wires,
                    builder: &mut f_build,
                };

                assert_eq!(linear.n_wires(), 2);

                linear
                    .append(LeafOp::H, [0])?
                    .append(LeafOp::CX, [0, 1])?
                    .append(LeafOp::CX, [1, 0])?;

                let outs = linear.finish();
                f_build.finish_with_outputs(outs)
            },
        );

        assert_matches!(build_res, Ok(_));
    }

    #[test]
    fn with_nonlinear_and_outputs() {
        let my_custom_op = LeafOp::CustomOp(
            crate::ops::custom::ExternalOp::Opaque(OpaqueOp::new(
                "MissingRsrc".into(),
                "MyOp",
                "unknown op".to_string(),
                vec![],
                Some(AbstractSignature::new(vec![QB, NAT], vec![QB], vec![])),
            ))
            .into(),
        );
        let build_res = build_main(
            AbstractSignature::new_df(type_row![QB, QB, NAT], type_row![QB, QB, BIT]).pure(),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

                let mut linear = f_build.as_circuit(vec![q0, q1]);

                let measure_out = linear
                    .append(LeafOp::CX, [0, 1])?
                    .append_and_consume(
                        my_custom_op,
                        [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
                    )?
                    .append_with_outputs(LeafOp::Measure, [0])?;

                let out_qbs = linear.finish();
                f_build.finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            },
        );

        assert_matches!(build_res, Ok(_));
    }
}
