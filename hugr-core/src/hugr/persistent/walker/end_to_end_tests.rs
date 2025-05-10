//! A test of the walker as it would typically be used by a user in practice.
//!
//! Currently a lot of the API is still being actively developed, so many
//! changes (for the better) should be expected.

use std::collections::{BTreeSet, HashMap, VecDeque};

use itertools::Itertools;

use crate::hugr::persistent::{CommitStateSpace, PatchNode, PersistentReplacement};
use crate::hugr::views::SiblingSubgraph;
use crate::utils::test_quantum_extension::h_gate;
use crate::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    utils::test_quantum_extension::cx_gate,
    Hugr,
};
use crate::{HugrView, IncomingPort, OutgoingPort, PortIndex, SimpleReplacement};

/// The maximum comimt depth that we will consider in this example
const MAX_COMMITS: usize = 2;

use super::{PinnedWire, Walker};

fn dfg_hugr() -> Hugr {
    // Assume all these CX gates are CZ gates (i.e. they commute with eachother)
    // (we don't have easy access to a CZ gate in the test utils), i.e. the
    // following circuit:
    //
    // --o--o-----o--o-----
    //   |  |     |  |
    // --o--+--o--+--o--o--
    //      |  |  |     |
    // -----o--o--o-----o--
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t(), qb_t()])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cx1 = builder.add_dataflow_op(cx_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cx1.outputs_arr();
    let cx2 = builder.add_dataflow_op(cx_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cx2.outputs_arr();
    let cx3 = builder.add_dataflow_op(cx_gate(), vec![q1, q2]).unwrap();
    let [q1, q2] = cx3.outputs_arr();
    let cx4 = builder.add_dataflow_op(cx_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cx4.outputs_arr();
    let cx5 = builder.add_dataflow_op(cx_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cx5.outputs_arr();
    let cx6 = builder.add_dataflow_op(cx_gate(), vec![q1, q2]).unwrap();
    let [q1, q2] = cx6.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

// TODO: currently empty replacements are buggy, so we have temporarily added
// a single Hadamard gate on each qubit.
fn empty_2qb_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
    let [q0, q1] = builder.input_wires_arr();
    let h0 = builder.add_dataflow_op(h_gate(), vec![q0]).unwrap();
    let [q0] = h0.outputs_arr();
    let h1 = builder.add_dataflow_op(h_gate(), vec![q1]).unwrap();
    let [q1] = h1.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1]).unwrap()
}

fn two_cz_3qb_hugr() -> Hugr {
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t(), qb_t()])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cx1 = builder.add_dataflow_op(cx_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cx1.outputs_arr();
    let cx2 = builder.add_dataflow_op(cx_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cx2.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

fn build_state_space() -> CommitStateSpace {
    let base_hugr = dfg_hugr();
    let mut state_space = CommitStateSpace::with_base(base_hugr);

    let mut wire_queue = VecDeque::new();
    let mut added_patches = BTreeSet::new();

    // Traverse all commits in state space, enqueueing all outgoing wires of
    // CX nodes
    let enqueue_all = |queue: &mut VecDeque<_>, state_space: &CommitStateSpace| {
        for id in state_space.all_commit_ids() {
            let hugr = state_space.commit_hugr(id);
            let cx_nodes = hugr
                .nodes()
                .filter(|&n| hugr.get_optype(n) == &cx_gate().into());
            for node in cx_nodes {
                let patch_node = PatchNode(id, node);
                let mut walker: Walker<'static> = Walker::new(state_space.clone());
                walker
                    .try_pin_node(patch_node)
                    .expect("pinning a single node should never fail");
                if walker.selected_commits.all_commit_ids().count() > MAX_COMMITS {
                    continue;
                }
                for outport in state_space.base_hugr().node_outputs(node) {
                    if !state_space
                        .base_hugr()
                        .get_optype(node)
                        .port_kind(outport)
                        .unwrap()
                        .is_value()
                    {
                        continue;
                    }
                    let wire = walker.get_wire(patch_node, outport);
                    queue.push_back((wire, walker.clone()));
                }
            }
        }
    };

    enqueue_all(&mut wire_queue, &state_space);

    while let Some((wire, walker)) = wire_queue.pop_front() {
        if !wire.is_complete(None) {
            // expand the wire in all possible ways
            let (pinned_node, pinned_port) = wire
                .all_ports()
                .next()
                .expect("at least one port was already pinned");
            assert!(
                !walker
                    .selected_commits
                    .deleted_nodes(pinned_node.0)
                    .contains(&pinned_node.1),
                "pinned node is deleted"
            );
            for walker in walker.expand(&wire, None) {
                assert!(
                    !walker
                        .selected_commits
                        .deleted_nodes(pinned_node.0)
                        .contains(&pinned_node.1),
                    "pinned node is deleted"
                );
                wire_queue.push_back((walker.get_wire(pinned_node, pinned_port), walker));
            }
        } else {
            // we have a complete wire, so we can commute the CZ gates (or
            // cancel them out)
            let Ok(repl) = create_replacement(wire, &walker) else {
                continue;
            };

            let patch_nodes: BTreeSet<_> = repl.subgraph().nodes().iter().copied().collect();

            // check that the patch applies to more than one commit (or the base),
            // otherwise we have infinite commutations back and forth
            let patch_owners: BTreeSet<_> = patch_nodes.iter().map(|n| n.0).collect();
            if patch_owners.len() <= 1 && !patch_owners.contains(&state_space.base()) {
                continue;
            }
            // check further that the same patch was not already added to `state_space`
            // (we currently do not have automatic deduplication)
            if !added_patches.insert(patch_nodes) {
                continue;
            }

            state_space
                .try_add_replacement(repl)
                .expect("repl acts on non-empty subgraph");

            // enqueue new wires added by the replacement
            // (this will also add a lot of already visited wires, but they will
            // be deduplicated)
            enqueue_all(&mut wire_queue, &state_space);
        }
    }

    state_space
}

fn create_replacement(wire: PinnedWire, walker: &Walker) -> Result<PersistentReplacement, ()> {
    let hugr = walker.clone().into_hugr();
    let (out_node, _) = wire
        .outgoing_port()
        .expect("outgoing port was already pinned (and is unique)");

    let (in_node, _) = wire
        .incoming_ports()
        .exactly_one()
        .ok()
        .expect("all our wires have exactly one incoming port");

    if hugr.commit_hugr(out_node.0).get_optype(out_node.1) != &cx_gate().into()
        || hugr.commit_hugr(in_node.0).get_optype(in_node.1) != &cx_gate().into()
    {
        // one of the nodes we have matched is (presumably) an input or output gate
        // => skip
        return Err(());
    }

    // figure out whether the two CZ gates act on the same qubits (iff the
    // the only outgoing neighbour of the first CZ is the second CZ gate)
    let all_in_nodes = (0..2)
        .map(OutgoingPort::from)
        .map(|p| {
            let (in_node, _) = hugr
                .get_all_incoming_ports(out_node, p)
                .exactly_one()
                .ok()
                .expect("all our wires have exactly one incoming port");
            in_node
        })
        .collect_vec();
    let act_on_same_qubits = all_in_nodes.iter().all(|n| n == &in_node);

    // TODO: simplify this once PersistentHugr implements HugrView
    // The current strategy is to apply all commits to obtain a concrete HUGR, with
    // `apply_all` on which a replacement `repl` can be constructed. The `repl`
    // is then mapped back to the original node types such that it applies on
    // the persistent hugr.
    if act_on_same_qubits {
        // cancel out the two CZ gates
        let (hugr, node_map) = hugr.apply_all();
        let subgraph_nodes = [out_node, in_node].map(|n| node_map[&n]);
        let subgraph = SiblingSubgraph::try_from_nodes(subgraph_nodes, &hugr).map_err(|_| ())?;
        let repl_hugr = empty_2qb_hugr();
        let [repl_hugr_inp, repl_hugr_out] = repl_hugr.get_io(repl_hugr.entrypoint()).unwrap();
        // The input boundary
        let nu_inp = repl_hugr
            .all_linked_inputs(repl_hugr_inp)
            .zip(
                hugr.node_inputs(subgraph_nodes[0])
                    .map(|p| (subgraph_nodes[0], p)),
            )
            .collect();
        // The output boundary
        let nu_out: HashMap<_, _> = hugr
            .node_outputs(subgraph_nodes[1])
            .map(|p| (subgraph_nodes[1], p))
            .zip(repl_hugr.node_inputs(repl_hugr_out))
            .collect();
        let repl = SimpleReplacement::new(subgraph, repl_hugr, nu_inp, nu_out);
        let node_map_inv: HashMap<_, _> = subgraph_nodes
            .iter()
            .copied()
            .zip([out_node, in_node])
            .collect();
        let repl = repl.map_host_nodes(|n| node_map_inv[&n]);
        Ok(repl)
    } else {
        // commute the two CZ gates

        // we need to establish which qubit is shared between the two CZ gates
        let shared_qb_on_out_node = all_in_nodes.iter().position(|&n| n == in_node).unwrap();
        let other_qb = 1 - shared_qb_on_out_node;
        let (_, port) = (0..2)
            .map(OutgoingPort::from)
            .map(|p| {
                hugr.get_all_incoming_ports(out_node, p)
                    .exactly_one()
                    .ok()
                    .expect("all our wires have exactly one incoming port")
            })
            .filter(|(n, _)| *n == in_node)
            .exactly_one()
            .expect("there must be one unknown neighbour");
        let shared_qb_on_in_node = port.index();
        let third_qb = 1 - shared_qb_on_in_node;

        let (hugr, node_map) = hugr.apply_all();

        let subgraph_nodes = [out_node, in_node].map(|n| node_map[&n]);
        let subgraph = SiblingSubgraph::try_from_nodes(subgraph_nodes, &hugr).map_err(|_| ())?;
        let repl_hugr = two_cz_3qb_hugr();
        let [repl_hugr_inp, repl_hugr_out] = repl_hugr.get_io(repl_hugr.entrypoint()).unwrap();

        let subgraph_input = vec![
            (subgraph_nodes[0], IncomingPort::from(shared_qb_on_out_node)),
            (subgraph_nodes[0], IncomingPort::from(other_qb)),
            (subgraph_nodes[1], IncomingPort::from(third_qb)),
        ];
        let subgraph_output = vec![
            (subgraph_nodes[1], OutgoingPort::from(shared_qb_on_in_node)),
            (subgraph_nodes[0], OutgoingPort::from(other_qb)),
            (subgraph_nodes[1], OutgoingPort::from(third_qb)),
        ];

        // The input boundary
        let nu_inp = repl_hugr
            .all_linked_inputs(repl_hugr_inp)
            .zip(subgraph_input)
            .collect();
        // The output boundary
        let nu_out: HashMap<_, _> = subgraph_output
            .into_iter()
            .zip(repl_hugr.node_inputs(repl_hugr_out))
            .collect();
        let repl = SimpleReplacement::new(subgraph, repl_hugr, nu_inp, nu_out);
        let node_map_inv: HashMap<_, _> = subgraph_nodes
            .iter()
            .copied()
            .zip([out_node, in_node])
            .collect();
        let repl = repl.map_host_nodes(|n| node_map_inv[&n]);
        Ok(repl)
    }
}

#[test]
fn run_end_to_end_test() {
    let state_space = build_state_space();
    println!("n commits = {:?}", state_space.all_commit_ids().count());

    for commit_id in state_space.all_commit_ids() {
        println!("========== Commit {commit_id:?} ============");
        println!(
            "parents = {:?}",
            state_space.parents(commit_id).collect_vec()
        );
        println!(
            "nodes deleted = {:?}",
            state_space
                .get_commit(commit_id)
                .deleted_nodes()
                .collect_vec()
        );
        println!("contents:");
        println!("{}\n", state_space.commit_hugr(commit_id).mermaid_string());
    }

    // assert_eq!(state_space.all_commit_ids().count(), 13);

    let empty_commits = state_space
        .all_commit_ids()
        // .filter(|&id| state_space.commit_hugr(id).num_nodes() == 3)
        .filter(|&id| {
            state_space
                .commit_hugr(id)
                .nodes()
                .filter(|&n| state_space.commit_hugr(id).get_optype(n) == &h_gate().into())
                .count()
                == 2
        })
        .collect_vec();

    // there should be a combination of three empty commits that are compatible
    // and such that the resulting HUGR is empty
    let mut empty_hugr = None;
    // for cs in empty_commits.iter().combinations(3) {
    for cs in empty_commits.iter().combinations(2) {
        let cs = cs.into_iter().copied().collect_vec();
        if let Ok(hugr) = state_space.try_extract_hugr(cs) {
            empty_hugr = Some(hugr);
        }
    }
    let empty_hugr = empty_hugr.unwrap().to_hugr();

    // assert_eq!(empty_hugr.num_nodes(), 3);

    let n_cx = empty_hugr
        .nodes()
        .filter(|&n| empty_hugr.get_optype(n) == &cx_gate().into())
        .count();
    let n_h = empty_hugr
        .nodes()
        .filter(|&n| empty_hugr.get_optype(n) == &h_gate().into())
        .count();
    assert_eq!(n_cx, 2);
    assert_eq!(n_h, 4);
}
