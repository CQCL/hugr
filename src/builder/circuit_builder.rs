use std::collections::HashMap;

use thiserror::Error;

use crate::hugr::NodeType;
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

/// Enum for specifying a [`CircuitBuilder`] input wire using either an index to
/// the builder vector of wires, or an arbitrary other wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AppendWire {
    /// Arbitrary input wire.
    W(Wire),
    /// Index to CircuitBuilder vector of wires.
    I(usize),
}

impl From<usize> for AppendWire {
    fn from(value: usize) -> Self {
        AppendWire::I(value)
    }
}

impl From<Wire> for AppendWire {
    fn from(value: Wire) -> Self {
        AppendWire::W(value)
    }
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
    pub fn append_and_consume<A: Into<AppendWire>>(
        &mut self,
        op: impl Into<OpType>,
        inputs: impl IntoIterator<Item = A>,
    ) -> Result<&mut Self, BuildError> {
        self.append_with_outputs(op, inputs)?;

        Ok(self)
    }

    /// Append an `op` with some inputs being the stored wires.
    /// Any inputs of the form [`AppendWire::I`] are used to index the
    /// stored wires.
    /// The outputs at those indices are used to replace the stored wire.
    /// The remaining outputs are returned.
    ///
    /// # Errors
    ///
    /// This function will return an error if an index is invalid.
    pub fn append_with_outputs<A: Into<AppendWire>>(
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
            .map(|(input_port, a_w): (usize, AppendWire)| match a_w {
                AppendWire::W(wire) => Some(wire),
                AppendWire::I(wire_index) => {
                    linear_inputs.insert(input_port, wire_index);
                    self.wires.get(wire_index).copied()
                }
            })
            .collect();

        let input_wires = input_wires.ok_or(CircuitBuildError::InvalidWireIndex)?;

        let output_wires = self
            .builder
            .add_dataflow_op(
                NodeType::pure(op), // TODO: Add resource param
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
            test::{build_main, BIT, F64, QB},
            Dataflow, DataflowSubContainer, Wire,
        },
        ops::LeafOp,
        type_row,
        types::{Signature, SignatureTrait},
    };

    #[test]
    fn simple_linear() {
        let build_res = build_main(
            Signature::new_df(type_row![QB, QB], type_row![QB, QB]),
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
        use AppendWire::{I, W};
        let build_res = build_main(
            Signature::new_df(type_row![QB, QB, F64], type_row![QB, QB, BIT]),
            |mut f_build| {
                let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

                let mut linear = f_build.as_circuit(vec![q0, q1]);

                let measure_out = linear
                    .append(LeafOp::CX, [0, 1])?
                    .append_and_consume(LeafOp::RzF64, [I(0), W(angle)])?
                    .append_with_outputs(LeafOp::Measure, [0])?;

                let out_qbs = linear.finish();
                f_build.finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            },
        );

        assert_matches!(build_res, Ok(_));
    }
}
