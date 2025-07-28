use std::collections::HashMap;
use std::mem;

use thiserror::Error;

use crate::ops::{NamedOp, OpType, Value};
use crate::utils::collect_array;

use super::{BuildError, Dataflow};
use crate::{CircuitUnit, Wire};

/// Builder to build regions of dataflow graphs that look like Circuits,
/// where some inputs of operations directly correspond to some outputs.
///
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
#[non_exhaustive]
pub enum CircuitBuildError {
    /// Invalid index for stored wires.
    #[error("Invalid wire index {invalid_index} while attempting to add operation {}.", .op.as_ref().map(|op| op.name()).unwrap_or_default())]
    InvalidWireIndex {
        /// The operation.
        op: Option<Box<OpType>>,
        /// The invalid indices.
        invalid_index: usize,
    },
    /// Some linear inputs had no corresponding output wire.
    #[error("The linear inputs {:?} had no corresponding output wire in operation {}.", .index.as_slice(), .op.name())]
    MismatchedLinearInputs {
        /// The operation.
        op: Box<OpType>,
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

    /// Returns an iterator over the tracked linear units.
    pub fn tracked_units(&self) -> impl Iterator<Item = usize> + '_ {
        self.wires
            .iter()
            .enumerate()
            .filter_map(|(i, w)| w.map(|_| i))
    }

    /// Returns an array with the tracked linear units.
    ///
    /// # Panics
    ///
    /// If the number of outputs does not match `N`.
    #[must_use]
    pub fn tracked_units_arr<const N: usize>(&self) -> [usize; N] {
        collect_array(self.tracked_units())
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
    /// Returns an error on an invalid input unit.
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
                op: Some(Box::new(op.clone())),
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
                op: Box::new(op),
                index: linear_inputs.values().copied().collect(),
            }
            .into());
        }

        Ok(nonlinear_outputs)
    }

    /// Append an `op` with some inputs being the stored wires.
    /// Any inputs of the form [`CircuitUnit::Linear`] are used to index the
    /// stored wires.
    /// The outputs at those indices are used to replace the stored wire.
    /// The remaining outputs are returned as an array.
    ///
    /// # Errors
    ///
    /// Returns an error on an invalid input unit.
    ///
    /// # Panics
    ///
    /// If the number of outputs does not match `N`.
    pub fn append_with_outputs_arr<const N: usize, A: Into<CircuitUnit>>(
        &mut self,
        op: impl Into<OpType>,
        inputs: impl IntoIterator<Item = A>,
    ) -> Result<[Wire; N], BuildError> {
        let outputs = self.append_with_outputs(op, inputs)?;
        Ok(collect_array(outputs))
    }

    /// Adds a constant value to the circuit and loads it into a wire.
    pub fn add_constant(&mut self, value: impl Into<Value>) -> Wire {
        self.builder.add_load_value(value)
    }

    /// Add a wire to the list of tracked wires.
    ///
    /// Returns the new unit index.
    pub fn track_wire(&mut self, wire: Wire) -> usize {
        self.wires.push(Some(wire));
        self.wires.len() - 1
    }

    /// Stops tracking a linear unit, and returns the last wire corresponding to it.
    ///
    /// Returns the new unit index.
    ///
    /// # Errors
    ///
    /// Returns a [`CircuitBuildError::InvalidWireIndex`] if the index is invalid.
    pub fn untrack_wire(&mut self, index: usize) -> Result<Wire, CircuitBuildError> {
        self.wires
            .get_mut(index)
            .and_then(mem::take)
            .ok_or(CircuitBuildError::InvalidWireIndex {
                op: None,
                invalid_index: index,
            })
    }

    #[inline]
    /// Finish building the circuit region and return the dangling wires
    /// that correspond to the initially provided wires.
    #[must_use]
    pub fn finish(self) -> Vec<Wire> {
        self.wires.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cool_asserts::assert_matches;

    use crate::Extension;
    use crate::builder::{HugrBuilder, ModuleBuilder};
    use crate::extension::ExtensionId;
    use crate::extension::prelude::{qb_t, usize_t};
    use crate::std_extensions::arithmetic::float_types::ConstF64;
    use crate::utils::test_quantum_extension::{
        self, cx_gate, h_gate, measure, q_alloc, q_discard, rz_f64,
    };
    use crate::{
        builder::{DataflowSubContainer, test::build_main},
        extension::prelude::bool_t,
        types::Signature,
    };

    #[test]
    fn simple_linear() {
        let build_res = build_main(
            Signature::new(vec![qb_t(), qb_t()], vec![qb_t(), qb_t()]).into(),
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

                let angle = linear.add_constant(ConstF64::new(0.5));
                linear.append_and_consume(
                    rz_f64(),
                    [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
                )?;

                let outs = linear.finish();
                f_build.finish_with_outputs(outs)
            },
        );

        assert_matches!(build_res, Ok(_));
    }

    #[test]
    fn with_nonlinear_and_outputs() {
        let my_ext_name: ExtensionId = "MyExt".try_into().unwrap();
        let my_ext = Extension::new_test_arc(my_ext_name.clone(), |ext, extension_ref| {
            ext.add_op(
                "MyOp".into(),
                String::new(),
                Signature::new(vec![qb_t(), usize_t()], vec![qb_t()]),
                extension_ref,
            )
            .unwrap();
        });
        let my_custom_op = my_ext.instantiate_extension_op("MyOp", []).unwrap();

        let mut module_builder = ModuleBuilder::new();
        let mut f_build = module_builder
            .define_function(
                "main",
                Signature::new(
                    vec![qb_t(), qb_t(), usize_t()],
                    vec![qb_t(), qb_t(), bool_t()],
                ),
            )
            .unwrap();

        let [q0, q1, angle]: [Wire; 3] = f_build.input_wires_arr();

        let mut linear = f_build.as_circuit([q0, q1]);

        let measure_out = linear
            .append(cx_gate(), [0, 1])
            .unwrap()
            .append_and_consume(
                my_custom_op,
                [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
            )
            .unwrap()
            .append_with_outputs(measure(), [0])
            .unwrap();

        let out_qbs = linear.finish();
        f_build
            .finish_with_outputs(out_qbs.into_iter().chain(measure_out))
            .unwrap();

        let mut registry = test_quantum_extension::REG.clone();
        registry.register(my_ext).unwrap();
        let build_res = module_builder.finish_hugr();

        assert_matches!(build_res, Ok(_));
    }

    #[test]
    fn ancillae() {
        let build_res = build_main(Signature::new_endo(qb_t()).into(), |mut f_build| {
            let mut circ = f_build.as_circuit(f_build.input_wires());
            assert_eq!(circ.n_wires(), 1);

            let [q0] = circ.tracked_units_arr();
            let [ancilla] = circ.append_with_outputs_arr(q_alloc(), [] as [CircuitUnit; 0])?;
            let ancilla = circ.track_wire(ancilla);

            assert_ne!(ancilla, 0);
            assert_eq!(circ.n_wires(), 2);
            assert_eq!(circ.tracked_units_arr(), [q0, ancilla]);

            circ.append(cx_gate(), [q0, ancilla])?;
            let [_bit] = circ.append_with_outputs_arr(measure(), [q0])?;

            let q0 = circ.untrack_wire(q0)?;

            assert_eq!(circ.tracked_units_arr(), [ancilla]);

            circ.append_and_consume(q_discard(), [q0])?;

            let outs = circ.finish();

            assert_eq!(outs.len(), 1);

            f_build.finish_with_outputs(outs)
        });

        assert_matches!(build_res, Ok(_));
    }

    #[test]
    fn circuit_builder_errors() {
        let _build_res = build_main(
            Signature::new_endo(vec![qb_t(), qb_t()]).into(),
            |mut f_build| {
                let mut circ = f_build.as_circuit(f_build.input_wires());
                let [q0, q1] = circ.tracked_units_arr();
                let invalid_index = 0xff;

                // Passing an invalid linear index returns an error
                assert_matches!(
                    circ.append(cx_gate(), [q0, invalid_index]),
                    Err(BuildError::CircuitError(CircuitBuildError::InvalidWireIndex { op, invalid_index: idx }))
                    if op == Some(Box::new(cx_gate().into())) && idx == invalid_index,
                );

                // Untracking an invalid index returns an error
                assert_matches!(
                    circ.untrack_wire(invalid_index),
                    Err(CircuitBuildError::InvalidWireIndex { op: None, invalid_index: idx })
                    if idx == invalid_index,
                );

                // Passing a linear index to an operation without a corresponding output returns an error
                assert_matches!(
                    circ.append(q_discard(), [q1]),
                    Err(BuildError::CircuitError(CircuitBuildError::MismatchedLinearInputs { op, index }))
                    if *op == q_discard().into() && index == [q1],
                );

                let outs = circ.finish();

                assert_eq!(outs.len(), 2);

                f_build.finish_with_outputs(outs)
            },
        );

        // We do not test the build output, as the internal errors may have left
        // the hugr in an invalid state.
    }
}
