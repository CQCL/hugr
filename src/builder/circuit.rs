use std::collections::HashMap;

use thiserror::Error;

use crate::ops::OpType;

use super::{BuildError, Dataflow};
use crate::{CircuitUnit, Wire};

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
                op, // TODO: Add extension param
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
            test::{build_main, NAT, QB},
            Dataflow, DataflowSubContainer, Wire,
        },
        extension::{prelude::BOOL_T, ExtensionSet},
        ops::{custom::OpaqueOp, LeafOp},
        type_row,
        types::FunctionType,
        utils::test_quantum_extension::{cx_gate, h_gate, measure, EXTENSION_ID},
    };

    #[test]
    fn simple_linear() {
        let build_res = build_main(
            FunctionType::new_endo(type_row![QB, QB])
                .with_extension_delta(&ExtensionSet::singleton(&EXTENSION_ID))
                .into(),
            |mut f_build| {
                let mut wires: [Wire; 2] = f_build.input_wires_arr();
                [wires[1]] = f_build
                    .add_dataflow_op(
                        LeafOp::Lift {
                            type_row: vec![QB].into(),
                            new_extension: EXTENSION_ID,
                        },
                        [wires[1]],
                    )?
                    .outputs_arr();

                let mut linear = CircuitBuilder {
                    wires: Vec::from(wires),
                    builder: &mut f_build,
                };

                assert_eq!(linear.n_wires(), 2);

                linear
                    .append(h_gate(), [0])?
                    .append(cx_gate(), [0, 1])?
                    .append(cx_gate(), [1, 0])?;

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
                "MissingRsrc".try_into().unwrap(),
                "MyOp",
                "unknown op".to_string(),
                vec![],
                FunctionType::new(vec![QB, NAT], vec![QB]),
            ))
            .into(),
        );
        let build_res = build_main(
            FunctionType::new(type_row![QB, QB, NAT], type_row![QB, QB, BOOL_T])
                .with_extension_delta(&ExtensionSet::singleton(&EXTENSION_ID))
                .into(),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();
                let [angle] = f_build
                    .add_dataflow_op(
                        LeafOp::Lift {
                            type_row: vec![NAT].into(),
                            new_extension: EXTENSION_ID,
                        },
                        [angle],
                    )?
                    .outputs_arr();
                let mut linear = f_build.as_circuit(vec![q0, q1]);

                let measure_out = linear
                    .append(cx_gate(), [0, 1])?
                    .append_and_consume(
                        my_custom_op,
                        [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
                    )?
                    .append_with_outputs(measure(), [0])?;

                let out_qbs = linear.finish();
                f_build.finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            },
        );

        assert_matches!(build_res, Ok(_));
    }
}
