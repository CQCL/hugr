//! RootChecked methods specific to dataflow graphs.

use std::collections::BTreeMap;

use itertools::Itertools;
use thiserror::Error;

use crate::{
    IncomingPort, OutgoingPort, Port, PortIndex,
    hugr::HugrMut,
    ops::{
        OpParent, OpTrait, OpType,
        handle::{DataflowParentID, DfgID},
    },
    types::{NoRV, Signature, Type, TypeBase},
};

use super::RootChecked;

macro_rules! impl_dataflow_parent_methods {
    ($handle_type:ident) => {
        impl<H: HugrMut> RootChecked<H, $handle_type<H::Node>> {
            /// Get the input and output nodes of the DFG at the entrypoint node.
            pub fn get_io(&self) -> [H::Node; 2] {
                self.hugr()
                    .get_io(self.hugr().entrypoint())
                    .expect("valid DFG graph")
            }

            /// Rewire the inputs and outputs of the nested DFG to modify its signature.
            ///
            /// Reorder the outgoing resp. incoming wires at the input resp. output
            /// node of the DFG to modify the signature of the DFG HUGR. This will
            /// recursively update the signatures of all ancestors of the entrypoint.
            ///
            /// ### Arguments
            ///
            /// * `new_inputs`: The new input signature. After the map, the i-th input
            ///   wire will be connected to the ports connected to the
            ///   `new_inputs[i]`-th input of the old DFG.
            /// * `new_outputs`: The new output signature. After the map, the i-th
            ///   output wire will be connected to the ports connected to the
            ///   `new_outputs[i]`-th output of the old DFG.
            ///
            /// Returns an `InvalidSignature` error if the new_inputs and new_outputs
            /// map are not valid signatures.
            ///
            /// ### Panics
            ///
            /// Panics if the DFG is not trivially nested, i.e. if there is an ancestor
            /// DFG of the entrypoint that has more than one inner DFG.
            pub fn map_function_type(
                &mut self,
                new_inputs: &[usize],
                new_outputs: &[usize],
            ) -> Result<(), InvalidSignature> {
                let [inp, out] = self.get_io();
                let Self(hugr, _) = self;

                // Record the old connections from and to the input and output nodes
                let old_inputs_incoming = hugr
                    .node_outputs(inp)
                    .map(|p| hugr.linked_inputs(inp, p).collect_vec())
                    .collect_vec();
                let old_outputs_outgoing = hugr
                    .node_inputs(out)
                    .map(|p| hugr.linked_outputs(out, p).collect_vec())
                    .collect_vec();

                // The old signature types
                let old_inp_sig = hugr
                    .get_optype(inp)
                    .dataflow_signature()
                    .expect("input has signature");
                let old_inp_sig = old_inp_sig.output_types();
                let old_out_sig = hugr
                    .get_optype(out)
                    .dataflow_signature()
                    .expect("output has signature");
                let old_out_sig = old_out_sig.input_types();

                // Check if the signature map is valid
                check_valid_inputs(&old_inputs_incoming, old_inp_sig, new_inputs)?;
                check_valid_outputs(old_out_sig, new_outputs)?;

                // The new signature types
                let new_inp_sig = new_inputs
                    .iter()
                    .map(|&i| old_inp_sig[i].clone())
                    .collect_vec();
                let new_out_sig = new_outputs
                    .iter()
                    .map(|&i| old_out_sig[i].clone())
                    .collect_vec();
                let new_sig = Signature::new(new_inp_sig, new_out_sig);

                // Remove all edges of the input and output nodes
                disconnect_all(hugr, inp);
                disconnect_all(hugr, out);

                // Update the signatures of the IO and their ancestors
                let mut is_ancestor = false;
                let mut node = hugr.entrypoint();
                while matches!(hugr.get_optype(node), OpType::FuncDefn(_) | OpType::DFG(_)) {
                    let [inner_inp, inner_out] = hugr.get_io(node).expect("valid DFG graph");
                    for node in [node, inner_inp, inner_out] {
                        update_signature(hugr, node, &new_sig);
                    }
                    if is_ancestor {
                        update_inner_dfg_links(hugr, node);
                    }
                    if let Some(parent) = hugr.get_parent(node) {
                        node = parent;
                        is_ancestor = true;
                    } else {
                        break;
                    }
                }

                // Insert the new edges at the input
                let mut old_output_to_new_input = BTreeMap::<IncomingPort, OutgoingPort>::new();
                for (inp_pos, &old_pos) in new_inputs.iter().enumerate() {
                    for &(node, port) in &old_inputs_incoming[old_pos] {
                        if node != out {
                            hugr.connect(inp, inp_pos, node, port);
                        } else {
                            old_output_to_new_input.insert(port, inp_pos.into());
                        }
                    }
                }

                // Insert the new edges at the output
                for (out_pos, &old_pos) in new_outputs.iter().enumerate() {
                    for &(node, port) in &old_outputs_outgoing[old_pos] {
                        if node != inp {
                            hugr.connect(node, port, out, out_pos);
                        } else {
                            let &inp_pos = old_output_to_new_input.get(&old_pos.into()).unwrap();
                            hugr.connect(inp, inp_pos, out, out_pos);
                        }
                    }
                }

                Ok(())
            }

            /// Add copyable inputs to the DFG to modify its signature.
            ///
            /// Append new inputs to the DFG. These will not be connected to any op and
            /// must be copyable. This will recursively update the signatures of all
            /// ancestors of the entrypoint.
            ///
            /// ### Arguments
            ///
            /// * `new_inputs`: The new input types to append to the signature.
            ///
            /// Returns an `InvalidSignature` error if the new_input types are not
            /// copyable.
            ///
            /// ### Panics
            ///
            /// Panics if the DFG is not trivially nested, i.e. if there is an ancestor
            /// DFG of the entrypoint that has more than one inner DFG.
            pub fn extend_inputs<'a>(
                &mut self,
                new_inputs: impl IntoIterator<Item = &'a Type>,
            ) -> Result<(), InvalidSignature> {
                let Self(hugr, _) = self;
                let curr_sig = hugr
                    .get_optype(hugr.entrypoint())
                    .inner_function_type()
                    .expect("valid DFG graph")
                    .into_owned();

                let n_inputs = curr_sig.input_count();

                let new_inputs: Vec<_> = new_inputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| {
                        if t.copyable() {
                            Ok(t)
                        } else {
                            let p = IncomingPort::from(n_inputs + i);
                            Err(InvalidSignature::ExpectedCopyable(p.into()))
                        }
                    })
                    .try_collect()?;

                let new_sig = Signature::new(curr_sig.input.extend(new_inputs), curr_sig.output);

                // Update the signatures of the IO and their ancestors
                let mut node = hugr.entrypoint();
                let mut is_ancestor = false;
                while matches!(hugr.get_optype(node), OpType::FuncDefn(_) | OpType::DFG(_)) {
                    let [inner_inp, inner_out] = hugr.get_io(node).expect("valid DFG graph");
                    for node in [node, inner_inp, inner_out] {
                        update_signature(hugr, node, &new_sig);
                    }
                    if is_ancestor {
                        update_inner_dfg_links(hugr, node);
                    }
                    if let Some(parent) = hugr.get_parent(node) {
                        node = parent;
                        is_ancestor = true;
                    } else {
                        break;
                    }
                }

                Ok(())
            }
        }
    };
}

impl_dataflow_parent_methods!(DataflowParentID);
impl_dataflow_parent_methods!(DfgID);

/// Panics if the DFG within `node` is not a single inner DFG.
fn update_inner_dfg_links<H: HugrMut>(hugr: &mut H, node: H::Node) {
    // connect all edges of the inner DFG to the input and output nodes
    let inner_dfg = hugr
        .children(node)
        .skip(2)
        .exactly_one()
        .ok()
        .expect("no non-trivial inner DFG");

    let [inp, out] = hugr.get_io(node).expect("valid DFG graph");
    disconnect_all(hugr, inner_dfg);
    for (out_port, _) in hugr.out_value_types(inp).collect_vec() {
        hugr.connect(inp, out_port, inner_dfg, out_port.index());
    }
    for (in_port, _) in hugr.in_value_types(out).collect_vec() {
        hugr.connect(inner_dfg, in_port.index(), out, in_port);
    }
}

fn disconnect_all<H: HugrMut>(hugr: &mut H, node: H::Node) {
    let all_ports = hugr.all_node_ports(node).collect_vec();
    for port in all_ports {
        hugr.disconnect(node, port);
    }
}

fn update_signature<H: HugrMut>(hugr: &mut H, node: H::Node, new_sig: &Signature) {
    match hugr.optype_mut(node) {
        OpType::DFG(dfg) => {
            dfg.signature = new_sig.clone();
        }
        OpType::FuncDefn(fn_def_op) => *fn_def_op.signature_mut() = new_sig.clone().into(),
        OpType::Input(inp) => {
            inp.types = new_sig.input().clone();
        }
        OpType::Output(out) => out.types = new_sig.output().clone(),
        _ => panic!("only update signature of DFG, FuncDefn, Input, or Output"),
    };
    let new_op = hugr.get_optype(node);
    hugr.set_num_ports(node, new_op.input_count(), new_op.output_count());
}

fn check_valid_inputs<V>(
    old_ports: &[Vec<V>],
    old_sig: &[TypeBase<NoRV>],
    map_sig: &[usize],
) -> Result<(), InvalidSignature> {
    if let Some(old_pos) = map_sig
        .iter()
        .find_map(|&old_pos| (old_pos >= old_sig.len()).then_some(old_pos))
    {
        return Err(InvalidSignature::UnknownIO(old_pos, "input"));
    }

    let counts = map_sig.iter().copied().counts();
    if let Some(old_pos) = old_ports.iter().enumerate().find_map(|(old_pos, vec)| {
        ((!vec.is_empty() || old_sig.get(old_pos).is_some_and(|t| !t.copyable()))
            && !counts.contains_key(&old_pos))
        .then_some(old_pos)
    }) {
        return Err(InvalidSignature::MissingIO(old_pos, "input"));
    }

    if let Some(old_pos) = counts
        .iter()
        .find_map(|(&old_pos, &count)| (count > 1).then_some(old_pos))
    {
        return Err(InvalidSignature::DuplicateInput(old_pos));
    }

    Ok(())
}

fn check_valid_outputs(
    old_sig: &[TypeBase<NoRV>],
    map_sig: &[usize],
) -> Result<(), InvalidSignature> {
    if let Some(old_pos) = map_sig
        .iter()
        .find_map(|&old_pos| (old_pos >= old_sig.len()).then_some(old_pos))
    {
        return Err(InvalidSignature::UnknownIO(old_pos, "output"));
    }

    let counts = map_sig.iter().copied().counts();
    let linear_types = old_sig
        .iter()
        .enumerate()
        .filter_map(|(pos, t)| (!t.copyable()).then_some(pos));
    for old_pos in linear_types {
        let Some(&cnt) = counts.get(&old_pos) else {
            return Err(InvalidSignature::MissingIO(old_pos, "output"));
        };
        if cnt != 1 {
            return Err(InvalidSignature::LinearityViolation(old_pos, "output"));
        }
    }

    Ok(())
}

/// Errors that can occur when mapping the I/O of a DFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum InvalidSignature {
    /// Error when a required input/output is missing from the new signature
    #[error("{1} at position {0} is required but missing in new signature")]
    MissingIO(usize, &'static str),
    /// Error when trying to access an input/output that doesn't exist in the
    /// signature
    #[error("No {1} at position {0} in signature")]
    UnknownIO(usize, &'static str),
    /// Error when a linear input/output is used multiple times or not at all
    #[error("Linearity of {1} at position {0} is not preserved in new signature")]
    LinearityViolation(usize, &'static str),
    /// Error when an input is used multiple times in the new signature
    #[error("Input at position {0} is duplicated in new signature")]
    DuplicateInput(usize),
    /// Expected a copyable type at the given port
    #[error("Type at port {0:?} must be copyable")]
    ExpectedCopyable(Port),
}

#[cfg(test)]
mod test {
    use insta::assert_snapshot;

    use super::*;
    use crate::builder::{
        DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, endo_sig,
    };
    use crate::extension::prelude::{bool_t, qb_t};
    use crate::hugr::views::root_checked::RootChecked;
    use crate::ops::handle::NodeHandle;
    use crate::ops::{NamedOp, OpParent};
    use crate::std_extensions::arithmetic::float_types::float64_type;
    use crate::types::Signature;
    use crate::utils::test_quantum_extension::cx_gate;
    use crate::{Hugr, HugrView};

    fn new_empty_dfg(sig: Signature) -> Hugr {
        let dfg_builder = DFGBuilder::new(sig).unwrap();
        let wires = dfg_builder.input_wires();
        dfg_builder.finish_hugr_with_outputs(wires).unwrap()
    }

    #[test]
    fn test_map_io() {
        // Create a DFG with 2 inputs and 2 outputs
        let sig = Signature::new_endo(vec![qb_t(), qb_t()]);
        let mut hugr = new_empty_dfg(sig);

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DataflowParentID>::try_new(&mut hugr).unwrap();

        // Test mapping inputs: [0,1] -> [1,0]
        let input_map = vec![1, 0];
        let output_map = vec![0, 1];

        // Map the I/O
        dfg_view.map_function_type(&input_map, &output_map).unwrap();

        // Verify the new signature
        let dfg_hugr = dfg_view.hugr();
        let new_sig = dfg_hugr
            .get_optype(dfg_hugr.entrypoint())
            .dataflow_signature()
            .unwrap();
        assert_eq!(new_sig.input_count(), 2);
        assert_eq!(new_sig.output_count(), 2);

        // Test invalid mapping - missing input
        let invalid_input_map = vec![0, 0];
        let err = dfg_view.map_function_type(&invalid_input_map, &output_map);
        assert!(matches!(err, Err(InvalidSignature::MissingIO(1, "input"))));

        // Test invalid mapping - duplicate input
        let invalid_input_map = vec![0, 0, 1];
        assert!(matches!(
            dfg_view.map_function_type(&invalid_input_map, &output_map),
            Err(InvalidSignature::DuplicateInput(0))
        ));

        // Test invalid mapping - unknown output
        let invalid_output_map = vec![0, 2];
        assert!(matches!(
            dfg_view.map_function_type(&input_map, &invalid_output_map),
            Err(InvalidSignature::UnknownIO(2, "output"))
        ));
    }

    #[test]
    fn test_map_io_dfg_id() {
        // Create a DFG with 2 inputs and 2 outputs
        let sig = Signature::new_endo(vec![qb_t(), qb_t()]);
        let mut hugr = new_empty_dfg(sig);

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DfgID>::try_new(&mut hugr).unwrap();

        // Test mapping inputs: [0,1] -> [1,0]
        let input_map = vec![1, 0];
        let output_map = vec![0, 1];

        // Map the I/O
        dfg_view.map_function_type(&input_map, &output_map).unwrap();

        // Verify the new signature
        let dfg_hugr = dfg_view.hugr();
        let new_sig = dfg_hugr
            .get_optype(dfg_hugr.entrypoint())
            .dataflow_signature()
            .unwrap();
        assert_eq!(new_sig.input_count(), 2);
        assert_eq!(new_sig.output_count(), 2);

        // Test invalid mapping - missing input
        let invalid_input_map = vec![0, 0];
        let err = dfg_view.map_function_type(&invalid_input_map, &output_map);
        assert!(matches!(err, Err(InvalidSignature::MissingIO(1, "input"))));

        // Test invalid mapping - duplicate input
        let invalid_input_map = vec![0, 0, 1];
        assert!(matches!(
            dfg_view.map_function_type(&invalid_input_map, &output_map),
            Err(InvalidSignature::DuplicateInput(0))
        ));

        // Test invalid mapping - unknown output
        let invalid_output_map = vec![0, 2];
        assert!(matches!(
            dfg_view.map_function_type(&input_map, &invalid_output_map),
            Err(InvalidSignature::UnknownIO(2, "output"))
        ));
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[test]
    fn test_map_io_duplicate_output() {
        // Create a DFG with 1 input and 1 output
        let sig = Signature::new_endo(vec![bool_t()]);
        let mut hugr = new_empty_dfg(sig);

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DataflowParentID>::try_new(&mut hugr).unwrap();

        // Test mapping outputs: [0] -> [0,0] (duplicating the output)
        let input_map = vec![0];
        let output_map = vec![0, 0];

        // Map the I/O
        dfg_view.map_function_type(&input_map, &output_map).unwrap();

        let dfg_hugr = dfg_view.hugr();
        if let Err(err) = dfg_hugr.validate() {
            panic!("Invalid Hugr: {err}");
        }

        // Verify the new signature
        let new_sig = dfg_hugr
            .get_optype(dfg_hugr.entrypoint())
            .dataflow_signature()
            .unwrap();
        assert_eq!(new_sig.input_count(), 1);
        assert_eq!(new_sig.output_count(), 2);
        assert_snapshot!(dfg_hugr.mermaid_string());
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[test]
    fn test_map_io_cx_gate() {
        // Create a DFG with 2 inputs and 2 outputs for a CX gate
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
        let [wire0, wire1] = dfg_builder.input_wires_arr();
        let cx_handle = dfg_builder
            .add_dataflow_op(cx_gate(), vec![wire0, wire1])
            .unwrap();
        let cx_node = cx_handle.node();
        let [wire0, wire1] = cx_handle.outputs_arr();
        let mut hugr = dfg_builder
            .finish_hugr_with_outputs(vec![wire0, wire1])
            .unwrap();

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DataflowParentID>::try_new(&mut hugr).unwrap();

        // Test mapping inputs: [0,1] -> [1,0] (swapping inputs)
        let input_map = vec![1, 0];
        let output_map = vec![0, 1];

        // Map the I/O
        dfg_view.map_function_type(&input_map, &output_map).unwrap();

        let dfg_hugr = dfg_view.hugr();
        if let Err(err) = dfg_hugr.validate() {
            panic!("Invalid Hugr: {err}");
        }

        // Verify the new signature
        let new_sig = dfg_hugr
            .get_optype(dfg_hugr.entrypoint())
            .dataflow_signature()
            .unwrap();
        assert_eq!(new_sig.input_count(), 2);
        assert_eq!(new_sig.output_count(), 2);

        // Verify the connections are preserved but swapped
        let [new_inp, new_out] = dfg_view.get_io();
        assert_eq!(
            dfg_hugr.linked_inputs(new_inp, 0).collect_vec(),
            vec![(cx_node, 1.into())]
        );
        assert_eq!(
            dfg_hugr.linked_inputs(new_inp, 1).collect_vec(),
            vec![(cx_node, 0.into())]
        );
        assert_eq!(
            dfg_hugr.linked_outputs(new_out, 0).collect_vec(),
            vec![(cx_node, 0.into())]
        );
        assert_eq!(
            dfg_hugr.linked_outputs(new_out, 1).collect_vec(),
            vec![(cx_node, 1.into())]
        );

        assert_snapshot!(dfg_hugr.mermaid_string());
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[test]
    fn test_map_io_cycle_3qb() {
        // Create a DFG with 3 inputs and 3 outputs: CX[0, 1] and empty 2nd qubit
        let mut dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t(); 3])).unwrap();
        let [wire0, wire1, wire2] = dfg_builder.input_wires_arr();
        let cx_handle = dfg_builder
            .add_dataflow_op(cx_gate(), vec![wire0, wire1])
            .unwrap();
        let cx_node = cx_handle.node();
        let [wire0, wire1] = cx_handle.outputs_arr();
        let mut hugr = dfg_builder
            .finish_hugr_with_outputs(vec![wire0, wire1, wire2])
            .unwrap();

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DfgID>::try_new(&mut hugr).unwrap();

        // Test cycling outputs: [0,1,2] -> [1,2,0]
        let input_map = vec![1, 2, 0];
        let output_map = vec![0, 1, 2];

        // Map the I/O
        dfg_view.map_function_type(&input_map, &output_map).unwrap();
        let [dfg_inp, dfg_out] = dfg_view.get_io();

        let dfg_hugr = dfg_view.hugr();
        if let Err(err) = dfg_hugr.validate() {
            panic!("Invalid Hugr: {err}");
        }

        // Verify the new signature
        let new_sig = dfg_hugr
            .get_optype(dfg_hugr.entrypoint())
            .dataflow_signature()
            .unwrap();
        assert_eq!(new_sig.input_count(), 3);
        assert_eq!(new_sig.output_count(), 3);

        // Verify inp(0) -> cx(1), inp(1) -> out(2), inp(2) -> cx(0)
        for (i, exp_gate) in [cx_node, dfg_out, cx_node].into_iter().enumerate() {
            assert_eq!(
                dfg_hugr.linked_inputs(dfg_inp, i).collect_vec(),
                vec![(exp_gate, ((i + 1) % 3).into())]
            );
        }
        // Verify cx(0) -> out(0), cx(1) -> out(1), inp(1) -> out(2)
        for (i, exp_gate) in [cx_node, cx_node, dfg_inp].into_iter().enumerate() {
            let exp_outport = std::cmp::min(i, 1);
            assert_eq!(
                dfg_hugr.linked_outputs(dfg_out, i).collect_vec(),
                vec![(exp_gate, exp_outport.into())],
                "expected {}({exp_outport}) -> out({i})",
                dfg_hugr.get_optype(exp_gate).name()
            );
        }

        assert_snapshot!(dfg_hugr.mermaid_string());
    }

    #[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
    #[test]
    fn test_map_io_recursive() {
        use crate::builder::ModuleBuilder;
        use crate::extension::prelude::{bool_t, qb_t};
        use crate::types::Signature;

        // Create a module with two functions: "foo" and "bar"
        let mut module_builder = ModuleBuilder::new();

        // Define function "foo" with nested DFGs
        let dfg_roots = {
            let mut foo_builder = module_builder
                .define_function("foo", Signature::new_endo(vec![qb_t(), bool_t()]))
                .unwrap();

            let [qb, b] = foo_builder.input_wires_arr();

            // Create first nested DFG
            let mut dfg1_builder = foo_builder
                .dfg_builder_endo([(qb_t(), qb), (bool_t(), b)])
                .unwrap();
            let [dfg1_qb, dfg1_b] = dfg1_builder.input_wires_arr();

            // Create second nested DFG inside the first one
            let dfg2_builder = dfg1_builder
                .dfg_builder_endo([(qb_t(), dfg1_qb), (bool_t(), dfg1_b)])
                .unwrap();
            let [dfg2_qb, dfg2_b] = dfg2_builder.input_wires_arr();

            // Connect inputs to outputs in innermost DFG
            let dfg2_id = dfg2_builder.finish_with_outputs([dfg2_qb, dfg2_b]).unwrap();

            // Connect through first DFG
            let dfg1_id = dfg1_builder.finish_with_outputs(dfg2_id.outputs()).unwrap();

            // Finish function
            let foo_id = foo_builder.finish_with_outputs(dfg1_id.outputs()).unwrap();

            [foo_id.node(), dfg1_id.node(), dfg2_id.node()]
        };

        let mut hugr = module_builder.finish_hugr().unwrap();
        hugr.set_entrypoint(dfg_roots[2]);

        // Test successful signature update in "foo"
        let mut dfg_view = RootChecked::<&mut Hugr, DfgID>::try_new(&mut hugr).unwrap();

        // Swap the outputs: [0,1] -> [1,0]
        let input_map = vec![0, 1];
        let output_map = vec![1, 0];

        dfg_view.map_function_type(&input_map, &output_map).unwrap();

        // Verify the new signature at each level
        for node in dfg_roots {
            let sig = hugr.get_optype(node).inner_function_type().unwrap();
            assert_eq!(sig.input_types(), vec![qb_t(), bool_t()]);
            assert_eq!(sig.output_types(), vec![bool_t(), qb_t()]);
        }

        assert_snapshot!(hugr.mermaid_string());
    }

    #[test]
    fn test_extend_inputs() {
        // Create an empty DFG
        let dfg_builder = DFGBuilder::new(endo_sig(vec![qb_t()])).unwrap();
        let [wire] = dfg_builder.input_wires_arr();
        let mut hugr = dfg_builder.finish_hugr_with_outputs(vec![wire]).unwrap();

        // Wrap in RootChecked
        let mut dfg_view = RootChecked::<&mut Hugr, DataflowParentID>::try_new(&mut hugr).unwrap();

        // Extend the inputs
        let new_inputs = vec![bool_t(), float64_type()];
        dfg_view.extend_inputs(&new_inputs).unwrap();
        assert_eq!(
            dfg_view.hugr().inner_function_type().unwrap(),
            Signature::new(vec![qb_t(), bool_t(), float64_type()], vec![qb_t()])
        );

        let new_inputs_fail = vec![qb_t()];
        let err = dfg_view.extend_inputs(&new_inputs_fail);
        assert_eq!(
            err,
            Err(InvalidSignature::ExpectedCopyable(
                IncomingPort::from(3).into()
            ))
        );
    }
}
