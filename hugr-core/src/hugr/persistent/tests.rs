use std::collections::{BTreeMap, HashMap};

use itertools::Itertools;
use rstest::*;

use crate::{
    builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::bool_t,
    hugr::{
        patch::ApplyPatch,
        persistent::{PatchNode, PersistentHugr},
        views::SiblingSubgraph,
        Hugr, HugrView,
    },
    ops::handle::NodeHandle,
    std_extensions::logic::LogicOp,
    IncomingPort, Node, OutgoingPort, SimpleReplacement,
};

/// Creates a simple test Hugr with a DFG that contains a small boolean circuit
///
/// Graph structure:
/// ```
///    ┌─────────┐
/// ───┤ (0) NOT ├─┐    ┌─────────┐
///    └─────────┘ └────┤         │
///    ┌─────────┐      │ (2) AND ├───
/// ───┤ (1) NOT ├──────┤         │
///    └─────────┘      └─────────┘
/// ```
///
/// Returns (Hugr, [not0_node, not1_node, and_node])
#[fixture]
fn simple_hugr() -> (Hugr, Vec<Node>) {
    let mut dfg_builder =
        DFGBuilder::new(inout_sig(vec![bool_t(), bool_t()], vec![bool_t()])).unwrap();

    let [b0, b1] = dfg_builder.input_wires_arr();

    let not0 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b0]).unwrap();
    let [b0_not] = not0.outputs_arr();

    let not1 = dfg_builder.add_dataflow_op(LogicOp::Not, vec![b1]).unwrap();
    let [b1_not] = not1.outputs_arr();

    let and = dfg_builder
        .add_dataflow_op(LogicOp::And, vec![b0_not, b1_not])
        .unwrap();

    let hugr = dfg_builder.finish_hugr_with_outputs(and.outputs()).unwrap();

    (hugr, vec![not0.node(), not1.node(), and.node()])
}

/// Creates a replacement that replaces a node with a sequence of two NOT gates
fn create_double_not_replacement(hugr: &Hugr, node_to_replace: Node) -> SimpleReplacement {
    // Create a simple hugr with two NOT gates in sequence
    let mut dfg_builder = DFGBuilder::new(inout_sig(vec![bool_t()], vec![bool_t()])).unwrap();
    let [input_wire] = dfg_builder.input_wires_arr();

    // Add first NOT gate
    let not1 = dfg_builder
        .add_dataflow_op(LogicOp::Not, vec![input_wire])
        .unwrap();
    let [not1_out] = not1.outputs_arr();

    // Add second NOT gate
    let not2 = dfg_builder
        .add_dataflow_op(LogicOp::Not, vec![not1_out])
        .unwrap();
    let [not2_out] = not2.outputs_arr();

    let replacement_hugr = dfg_builder.finish_hugr_with_outputs([not2_out]).unwrap();

    // Find the input and output of the node to replace
    let host_output_target = hugr
        .single_linked_input(node_to_replace, OutgoingPort::from(0))
        .unwrap();

    // Create the mappings
    let mut nu_inp = HashMap::new();
    nu_inp.insert(
        (not1.node(), IncomingPort::from(0)),
        (node_to_replace, IncomingPort::from(0)),
    );

    let mut nu_out = HashMap::new();
    nu_out.insert(host_output_target, IncomingPort::from(0));

    // Create the subgraph with the single node
    let subgraph = SiblingSubgraph::try_from_nodes(vec![node_to_replace], hugr).unwrap();

    // Create the replacement
    SimpleReplacement::new(subgraph, replacement_hugr, nu_inp, nu_out)
}

/// Creates a replacement that replaces the unique AND gate in `hugr` and its
/// predecessor NOT gate on 1st input with an XOR gate
fn create_not_and_to_xor_replacement(hugr: &Hugr) -> SimpleReplacement {
    // Create second replacement that replaces the second NOT gate from the first
    // replacement
    // Find the AND gate in the hugr
    let and_gate = hugr
        .nodes()
        .find(|&n| hugr.get_optype(n) == &LogicOp::And.into())
        .unwrap();
    // The NOT gate before the AND on its first input
    let not_node = hugr.input_neighbours(and_gate).next().unwrap();

    // Create a replacement for the AND and the NOT0 with an XOR gate
    let mut dfg_builder =
        DFGBuilder::new(inout_sig(vec![bool_t(), bool_t()], vec![bool_t()])).unwrap();
    let [in1, in2] = dfg_builder.input_wires_arr();

    // Add an XOR gate
    let xor_op = dfg_builder
        .add_dataflow_op(LogicOp::Xor, vec![in1, in2])
        .unwrap();

    let replacement_hugr = dfg_builder
        .finish_hugr_with_outputs(xor_op.outputs())
        .unwrap();

    // Create mappings for the inputs
    let mut nu_inp = HashMap::new();

    // Map the first input of XOR to the input of the NOT gate
    nu_inp.insert(
        (xor_op.node(), IncomingPort::from(0)),
        (not_node, IncomingPort::from(0)),
    );

    // Map the second input of XOR to the second input of the AND gate
    nu_inp.insert(
        (xor_op.node(), IncomingPort::from(1)),
        (and_gate, IncomingPort::from(1)),
    );

    // Output mapping - AND gate's output to XOR's output
    let mut nu_out = HashMap::new();
    let and_output_port = hugr.single_linked_input(and_gate, 0).unwrap();
    nu_out.insert(and_output_port, IncomingPort::from(0));

    // Create subgraph with both the AND gate and NOT0 node
    let subgraph = SiblingSubgraph::try_from_nodes(vec![not_node, and_gate], &hugr).unwrap();

    SimpleReplacement::new(subgraph, replacement_hugr, nu_inp, nu_out)
}

#[rstest]
fn test_successive_replacements(simple_hugr: (Hugr, Vec<Node>)) {
    let (base_hugr, nodes) = simple_hugr;
    let base_hugr_clone = base_hugr.clone();

    println!("{}", base_hugr.mermaid_string());

    // Extract nodes
    let (not0_node, _not1_node, _and_node) = nodes.into_iter().collect_tuple().unwrap();

    // Create first replacement (replace NOT0 with two NOT gates)
    let replacement1 = create_double_not_replacement(&base_hugr, not0_node);

    // Apply the first replacement directly to a clone
    let mut direct_hugr = base_hugr_clone.clone();
    let outcome1 = replacement1.clone().apply(&mut direct_hugr).unwrap();
    println!("{}", direct_hugr.mermaid_string());

    // Create second replacement (replace NOT+AND with XOR)
    let replacement2 = create_not_and_to_xor_replacement(&direct_hugr);

    // Apply the second replacement directly
    let _outcome2 = replacement2.clone().apply(&mut direct_hugr).unwrap();

    // Create a persistent hugr
    let mut persistent_hugr = PersistentHugr::with_base(base_hugr);

    // Add first replacement
    let to_patch_node = |n: Node| PatchNode(persistent_hugr.base(), n);
    // translate replacement1 to patch nodes in `persistent_hugr`
    let replacement1 = replacement1.map_host_nodes(to_patch_node);
    let patch1 = persistent_hugr.add_replacement(replacement1);

    // Add second replacement
    let to_patch_node = {
        let inv_node_map = {
            let mut inv = BTreeMap::new();
            for (repl_node, hugr_node) in outcome1.node_map {
                inv.insert(hugr_node, repl_node);
            }
            inv
        };
        let base_patch = persistent_hugr.base();
        move |n| {
            if let Some(&n) = inv_node_map.get(&n) {
                // node was replaced by patch1
                PatchNode(patch1, n)
            } else {
                // node is in base hugr
                PatchNode(base_patch, n)
            }
        }
    };
    // translate replacement2 to patch nodes
    let replacement2 = replacement2.map_host_nodes(to_patch_node);
    persistent_hugr.add_replacement(replacement2);

    // Get the final hugr from the persistent context
    let persistent_final_hugr = persistent_hugr.to_hugr();

    // Check we have the expected number of patches (original + 2 replacements)
    assert_eq!(persistent_hugr.patch_ids().count(), 3);

    assert_eq!(direct_hugr.validate(), Ok(()));
    assert_eq!(persistent_final_hugr.validate(), Ok(()));
    assert_eq!(
        direct_hugr.mermaid_string(),
        persistent_final_hugr.mermaid_string()
    );
}

#[rstest]
fn test_conflicting_replacements(simple_hugr: (Hugr, Vec<Node>)) {
    let (base_hugr, nodes) = simple_hugr;

    // Extract nodes
    let (not0_node, _not1_node, _and_node) = nodes.into_iter().collect_tuple().unwrap();

    // Create first replacement (replace NOT0 with with two nots)
    let replacement1 = create_double_not_replacement(&base_hugr, not0_node);

    // Create a second replacement that will conflict with the first
    // by replacing NOT0 and AND with XOR
    let replacement2 = create_not_and_to_xor_replacement(&base_hugr);

    // Create a persistent hugr and add first replacement
    let mut persistent_hugr1 = PersistentHugr::with_base(base_hugr.clone());
    let to_patch_node = |n: Node| PatchNode(persistent_hugr1.base(), n);
    persistent_hugr1.add_replacement(replacement1.map_host_nodes(to_patch_node));

    // Create another persistent hugr and add second replacement
    let mut persistent_hugr2 =
        PersistentHugr::try_new([persistent_hugr1.base_patch().clone()]).unwrap();
    let to_patch_node = |n: Node| PatchNode(persistent_hugr2.base(), n);
    persistent_hugr2.add_replacement(replacement2.map_host_nodes(to_patch_node));

    // Both individual replacements should be valid
    assert_eq!(persistent_hugr1.to_hugr().validate(), Ok(()));
    assert_eq!(persistent_hugr2.to_hugr().validate(), Ok(()));

    // But trying to create a history with both replacements should fail
    let common_state_space = {
        let mut space = persistent_hugr1.clone().into_state_space();
        space.extend(persistent_hugr2.clone());
        space
    };
    assert_eq!(common_state_space.all_patch_ids().count(), 3);
    let result = common_state_space.try_extract_hugr(common_state_space.all_patch_ids());
    assert!(
        result.is_err(),
        "Creating history with conflicting patches should fail"
    );

    // Apply first replacement directly to a clone
    let mut direct_hugr1 = base_hugr.clone();
    replacement1.apply(&mut direct_hugr1).unwrap();

    assert_eq!(
        direct_hugr1.mermaid_string(),
        persistent_hugr1.to_hugr().mermaid_string()
    );

    // Apply second replacement directly to another clone
    let mut direct_hugr2 = base_hugr;
    replacement2.apply(&mut direct_hugr2).unwrap();

    assert_eq!(
        direct_hugr2.mermaid_string(),
        persistent_hugr2.to_hugr().mermaid_string()
    );
}

#[rstest]
fn test_disjoint_replacements(simple_hugr: (Hugr, Vec<Node>)) {
    let (base_hugr, nodes) = simple_hugr;

    // Extract nodes
    let (not0_node, not1_node, _and_node) = nodes.into_iter().collect_tuple().unwrap();

    // Create non-overlapping replacements for NOT0 and NOT1
    let replacement1 = create_double_not_replacement(&base_hugr, not0_node);
    let replacement2 = create_double_not_replacement(&base_hugr, not1_node);

    // Apply replacements directly, in sequence
    let mut direct_hugr = base_hugr.clone();
    replacement1.clone().apply(&mut direct_hugr).unwrap();
    replacement2.clone().apply(&mut direct_hugr).unwrap();

    // Create a persistent hugr and add both replacements
    let mut persistent_hugr = PersistentHugr::with_base(base_hugr);
    let base_id = persistent_hugr.base();
    let to_patch_node = |n: Node| PatchNode(base_id, n);
    persistent_hugr.add_replacement(replacement1.map_host_nodes(to_patch_node));
    persistent_hugr.add_replacement(replacement2.map_host_nodes(to_patch_node));

    // Get the final hugr
    let persistent_final_hugr = persistent_hugr.to_hugr();

    // Both hugrs should be valid
    assert_eq!(direct_hugr.validate(), Ok(()));
    assert_eq!(persistent_final_hugr.validate(), Ok(()));

    // We should have 3 patches (base + 2 replacements)
    assert_eq!(persistent_hugr.patch_ids().count(), 3);

    assert_eq!(
        direct_hugr.mermaid_string(),
        persistent_final_hugr.mermaid_string()
    );
}
