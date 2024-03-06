use std::collections::HashMap;
use std::mem;

use thiserror::Error;

use crate::ops::{OpName, OpType};

use super::{BuildError, Dataflow};
use crate::{CircuitUnit, Wire};

/// Builder to build regions of dataflow graphs that look like Circuits,
/// where some inputs of operations directly correspond to some outputs.
/// Allows appending operations by indexing a vector of input wires.
#[derive(Debug, PartialEq)]
pub struct CircuitBuilder<'a, T: ?Sized> {
    /// List of wires that are being tracked, identified by their index in the vector.
    ///
    /// Terminating wires may create holes in the vector, but the indices are stable.
    wires: Vec<Option<Wire>>,
    builder: &'a mut T,
}

#[derive(Debug, Clone, PartialEq, Error)]
/// Error in [`CircuitBuilder`]
pub enum CircuitBuildError {
    /// Invalid index for stored wires.
    #[error("Invalid wire index {invalid_index} while attempting to add operation {}.", .op.name())]
    InvalidWireIndex {
        /// The operation.
        op: OpType,
        /// The invalid indices.
        invalid_index: usize,
    },
    /// Some linear inputs had no corresponding output wire.
    #[error("The linear inputs {:?} had no corresponding output wire in operation {}.", .index.as_slice(), .op.name())]
    MismatchedLinearInputs {
        /// The operation.
        op: OpType,
        /// The index of the input that had no corresponding output wire.
        index: Vec<usize>,
    },
}

impl<'a, T: Dataflow + ?Sized> CircuitBuilder<'a, T> {
    /// Construct a new [`CircuitBuilder`] from a vector of incoming wires and the
    /// builder for the graph.
    pub fn new(wires: impl IntoIterator<Item = Wire>, builder: &'a mut T) -> Self {
        Self {
            wires: wires.into_iter().map(Some).collect(),
            builder,
        }
    }

    /// Returns the number of wires tracked.
    #[must_use]
    pub fn n_wires(&self) -> usize {
        self.wires.iter().flatten().count()
    }

    /// Returns the wire associated with the given index.
    #[must_use]
    pub fn tracked_wire(&self, index: usize) -> Option<Wire> {
        self.wires.get(index).copied().flatten()
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
        let op = op.into();

        let input_wires: Result<Vec<Wire>, usize> = inputs
            .into_iter()
            .map(Into::into)
            .enumerate()
            .map(|(input_port, a_w): (usize, CircuitUnit)| match a_w {
                CircuitUnit::Wire(wire) => Ok(wire),
                CircuitUnit::Linear(wire_index) => {
                    linear_inputs.insert(input_port, wire_index);
                    self.tracked_wire(wire_index).ok_or(wire_index)
                }
            })
            .collect();

        let input_wires =
            input_wires.map_err(|invalid_index| CircuitBuildError::InvalidWireIndex {
                op: op.clone(),
                invalid_index,
            })?;

        let output_wires = self
            .builder
            .add_dataflow_op(
                op.clone(), // TODO: Add extension param
                input_wires,
            )?
            .outputs();
        let nonlinear_outputs: Vec<Wire> = output_wires
            .enumerate()
            .filter_map(|(output_port, wire)| {
                if let Some(wire_index) = linear_inputs.remove(&output_port) {
                    // output at output_port replaces input wire from same port
                    self.wires[wire_index] = Some(wire);
                    None
                } else {
                    Some(wire)
                }
            })
            .collect();

        if !linear_inputs.is_empty() {
            return Err(CircuitBuildError::MismatchedLinearInputs {
                op,
                index: linear_inputs.values().copied().collect(),
            }
            .into());
        }

        Ok(nonlinear_outputs)
    }

    /// Add new tracked linear wires to the circuit, initialized via the given `op`.
    ///
    /// Any output from the operation will be tracked as a new linear wire.
    #[inline]
    pub fn add_ancilla(&mut self, op: impl Into<OpType>) -> Result<Vec<usize>, BuildError> {
        self.add_ancilla_with_inputs::<CircuitUnit>(op, [])
    }

    /// Add new tracked linear wires to the circuit, initialized via the given
    /// `op`.
    ///
    /// The operation may receive additional inputs. Any output without a
    /// matching linear input will be tracked as a new linear wire.
    pub fn add_ancilla_with_inputs<A: Into<CircuitUnit>>(
        &mut self,
        op: impl Into<OpType>,
        inputs: impl IntoIterator<Item = A>,
    ) -> Result<Vec<usize>, BuildError> {
        let wires = self.append_with_outputs(op, inputs)?;
        let mut new_indices = Vec::with_capacity(wires.len());
        for w in wires {
            self.wires.push(Some(w));
            new_indices.push(self.wires.len() - 1);
        }
        Ok(new_indices)
    }

    /// Discards ancillae with a consuming operation.
    #[inline]
    pub fn discard_ancilla(
        &mut self,
        op: impl Into<OpType>,
        ancilla: impl IntoIterator<Item = usize>,
    ) -> Result<&mut Self, BuildError> {
        self.discard_ancilla_with_outputs::<CircuitUnit>(op, ancilla)?;
        Ok(self)
    }

    /// Discards ancillae with a consuming operation.
    ///
    /// Returns the output wires of the operation.
    pub fn discard_ancilla_with_outputs<A: Into<CircuitUnit>>(
        &mut self,
        op: impl Into<OpType>,
        ancilla: impl IntoIterator<Item = usize>,
    ) -> Result<Vec<Wire>, BuildError> {
        let op = op.into();

        // Remove the ancillae from the list of tracked wires.
        let wires: Result<Vec<Wire>, usize> = ancilla
            .into_iter()
            .map(|i| self.wires.get_mut(i).and_then(mem::take).ok_or(i))
            .collect();
        let wires = wires.map_err(|invalid_index| CircuitBuildError::InvalidWireIndex {
            op: op.clone(),
            invalid_index,
        })?;

        self.append_with_outputs(op, wires)
    }

    #[inline]
    /// Finish building the circuit region and return the dangling wires
    /// that correspond to the initially provided wires.
    pub fn finish(self) -> Vec<Wire> {
        self.wires.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;

    use crate::utils::test_quantum_extension::{
        cx_gate, cz_gate, h_gate, measure, q_alloc, q_discard,
    };
    use crate::{
        builder::{
            test::{build_main, NAT, QB},
            Dataflow, DataflowSubContainer, Wire,
        },
        extension::prelude::BOOL_T,
        ops::{custom::OpaqueOp, LeafOp},
        type_row,
        types::FunctionType,
    };

    #[test]
    fn simple_linear() {
        let build_res = build_main(
            FunctionType::new(type_row![QB, QB], type_row![QB, QB]).into(),
            |mut f_build| {
                let wires = f_build.input_wires().map(Some).collect();

                let mut linear = CircuitBuilder {
                    wires,
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
            FunctionType::new(type_row![QB, QB, NAT], type_row![QB, QB, BOOL_T]).into(),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

                let mut linear = f_build.as_circuit([q0, q1]);

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

    #[test]
    fn ancillae() {
        let build_res = build_main(
            FunctionType::new(type_row![QB], type_row![QB]).into(),
            |mut f_build| {
                let mut circ = f_build.as_circuit(f_build.input_wires());

                assert_eq!(circ.n_wires(), 1);

                let [ancilla] = circ
                    .add_ancilla(q_alloc())?
                    .try_into()
                    .expect("Expected a single ancilla wire");

                assert_ne!(ancilla, 0);
                assert_eq!(circ.n_wires(), 2);

                circ.append(cz_gate(), [0, ancilla])?;
                let [_bit] = circ
                    .append_with_outputs(measure(), [0])?
                    .try_into()
                    .unwrap();

                // We could apply a classically controlled operation here
                // to complete a circuit that emulates a Hadamard gate.
                //
                //circ.append_and_consume(
                //    controlled_x(),
                //    [CircuitUnit::Linear(0), CircuitUnit::Wire(bit)],
                //)?;

                circ.discard_ancilla(q_discard(), [0])?;

                assert_eq!(circ.n_wires(), 1);

                let outs = circ.finish();

                assert_eq!(outs.len(), 1);

                f_build.finish_with_outputs(outs)
            },
        );

        assert_matches!(build_res, Ok(_));
    }
}
