//! A test of the walker as it would typically be used by a user in practice.

use std::collections::{BTreeSet, HashMap, VecDeque};

use itertools::Itertools;

use hugr_core::{
    Hugr, HugrView, IncomingPort, OutgoingPort, PortIndex, SimpleReplacement,
    builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig},
    extension::prelude::qb_t,
    hugr::{
        persistent::{CommitStateSpace, PersistentReplacement, PinnedWire, Walker},
        views::SiblingSubgraph,
    },
};

/// The maximum comimt depth that we will consider in this example
const MAX_COMMITS: usize = 2;

// We define a HUGR extension within this file, with CZ and H gates. Normally, you
// would use an existing extension (e.g. as provided by tket2).
use walker_example_extension::{cz_gate, h_gate};
mod walker_example_extension {
    use std::sync::Arc;

    use hugr::Extension;
    use hugr::extension::ExtensionId;
    use hugr::ops::{ExtensionOp, OpName};
    use hugr::types::{FuncValueType, PolyFuncTypeRV};

    use lazy_static::lazy_static;
    use semver::Version;

    use super::*;

    fn one_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(qb_t()).into()
    }

    fn two_qb_func() -> PolyFuncTypeRV {
        FuncValueType::new_endo(vec![qb_t(), qb_t()]).into()
    }

    /// The extension identifier.
    const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("test.walker");
    fn extension() -> Arc<Extension> {
        Extension::new_arc(
            EXTENSION_ID,
            Version::new(0, 0, 0),
            |extension, extension_ref| {
                extension
                    .add_op(
                        OpName::new_inline("H"),
                        "Hadamard".into(),
                        one_qb_func(),
                        extension_ref,
                    )
                    .unwrap();

                extension
                    .add_op(
                        OpName::new_inline("CZ"),
                        "CZ".into(),
                        two_qb_func(),
                        extension_ref,
                    )
                    .unwrap();
            },
        )
    }

    lazy_static! {
        /// Quantum extension definition.
        static ref EXTENSION: Arc<Extension> = extension();
    }

    pub fn h_gate() -> ExtensionOp {
        EXTENSION.instantiate_extension_op("H", []).unwrap()
    }

    pub fn cz_gate() -> ExtensionOp {
        EXTENSION.instantiate_extension_op("CZ", []).unwrap()
    }
}

fn dfg_hugr() -> Hugr {
    // All gates are CZ gates (i.e. they commute with eachother):
    //
    // --o--o-----o--o-----
    //   |  |     |  |
    // --o--+--o--+--o--o--
    //      |  |  |     |
    // -----o--o--o-----o--
    let mut builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t(), qb_t()])).unwrap();
    let [q0, q1, q2] = builder.input_wires_arr();
    let cz1 = builder.add_dataflow_op(cz_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cz1.outputs_arr();
    let cz2 = builder.add_dataflow_op(cz_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cz2.outputs_arr();
    let cz3 = builder.add_dataflow_op(cz_gate(), vec![q1, q2]).unwrap();
    let [q1, q2] = cz3.outputs_arr();
    let cz4 = builder.add_dataflow_op(cz_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cz4.outputs_arr();
    let cz5 = builder.add_dataflow_op(cz_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cz5.outputs_arr();
    let cz6 = builder.add_dataflow_op(cz_gate(), vec![q1, q2]).unwrap();
    let [q1, q2] = cz6.outputs_arr();
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
    let cz1 = builder.add_dataflow_op(cz_gate(), vec![q0, q2]).unwrap();
    let [q0, q2] = cz1.outputs_arr();
    let cz2 = builder.add_dataflow_op(cz_gate(), vec![q0, q1]).unwrap();
    let [q0, q1] = cz2.outputs_arr();
    builder.finish_hugr_with_outputs(vec![q0, q1, q2]).unwrap()
}

fn build_state_space() -> CommitStateSpace {
    let base_hugr = dfg_hugr();
    let mut state_space = CommitStateSpace::with_base(base_hugr);

    let mut wire_queue = VecDeque::new();
    let mut added_patches = BTreeSet::new();

    // Traverse all commits in state space, enqueueing all outgoing wires of
    // CZ nodes
    let enqueue_all = |queue: &mut VecDeque<_>, state_space: &CommitStateSpace| {
        for id in state_space.all_commit_ids() {
            let cz_nodes = state_space
                .inserted_nodes(id)
                .filter(|&n| state_space.get_optype(n) == &cz_gate().into());
            for node in cz_nodes {
                let mut walker: Walker<'static> = Walker::new(state_space.clone());
                walker
                    .try_pin_node(node)
                    .expect("pinning a single node should never fail");
                if walker.as_hugr().all_commit_ids().count() > MAX_COMMITS {
                    continue;
                }
                for outport in state_space.node_outputs(node) {
                    if !state_space
                        .get_optype(node)
                        .port_kind(outport)
                        .unwrap()
                        .is_value()
                    {
                        continue;
                    }
                    let wire = walker.get_wire(node, outport);
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
                    .as_hugr()
                    .deleted_nodes(pinned_node.0)
                    .contains(&pinned_node.1),
                "pinned node is deleted"
            );
            for walker in walker.expand(&wire, None) {
                assert!(
                    !walker
                        .as_hugr()
                        .deleted_nodes(pinned_node.0)
                        .contains(&pinned_node.1),
                    "pinned node is deleted"
                );
                wire_queue.push_back((walker.get_wire(pinned_node, pinned_port), walker));
            }
        } else {
            // we have a complete wire, so we can commute the CZ gates (or
            // cancel them out)

            let patch_nodes: BTreeSet<_> = wire.all_ports().map(|(n, _)| n).collect();
            // check that the patch applies to more than one commit (or the base),
            // otherwise we have infinite commutations back and forth
            let patch_owners: BTreeSet<_> = patch_nodes.iter().map(|n| n.0).collect();
            if patch_owners.len() <= 1 && !patch_owners.contains(&state_space.base()) {
                continue;
            }
            // check further that the same patch was not already added to `state_space`
            // (we currently do not have automatic deduplication)
            if !added_patches.insert(patch_nodes.clone()) {
                continue;
            }

            let Ok(repl) = create_replacement(wire, &walker) else {
                continue;
            };

            assert_eq!(
                repl.subgraph()
                    .nodes()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                patch_nodes
            );

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

    if hugr.get_optype(out_node) != &cz_gate().into()
        || hugr.get_optype(in_node) != &cz_gate().into()
    {
        // one of the nodes we have matched is (presumably) an input or output gate
        // => skip
        return Err(());
    }

    // figure out whether the two CZ gates act on the same qubits (iff the
    // the only outgoing neighbour of the first CZ is the second CZ gate)
    let all_edges = hugr.node_connections(out_node, in_node).collect_vec();
    let n_shared_qubits = all_edges.len();

    // The subgraph that we will replace
    let subgraph_nodes = [out_node, in_node];
    let subgraph = SiblingSubgraph::try_from_nodes(subgraph_nodes, &hugr).map_err(|_| ())?;

    let (repl_hugr, nu_inp, nu_out) = match n_shared_qubits {
        2 => {
            // out_node and in_node act on the same qubits
            // => cancel out the two CZ gates

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

            (repl_hugr, nu_inp, nu_out)
        }
        1 => {
            // out_node and in_node share just one qubit
            // => commute the two CZ gates past each other
            let [out_port, in_port] = all_edges.into_iter().exactly_one().unwrap();

            // we need to establish which qubit is shared between the two CZ gates
            let shared_qb_on_out_node = out_port.index();
            let shared_qb_on_in_node = in_port.index();

            let second_qb = 1 - shared_qb_on_out_node;
            let third_qb = 1 - shared_qb_on_in_node;

            let repl_hugr = two_cz_3qb_hugr();
            let [repl_hugr_inp, repl_hugr_out] = repl_hugr.get_io(repl_hugr.entrypoint()).unwrap();

            let subgraph_input = vec![
                (subgraph_nodes[0], IncomingPort::from(shared_qb_on_out_node)),
                (subgraph_nodes[0], IncomingPort::from(second_qb)),
                (subgraph_nodes[1], IncomingPort::from(third_qb)),
            ];
            let subgraph_output = vec![
                (subgraph_nodes[1], OutgoingPort::from(shared_qb_on_in_node)),
                (subgraph_nodes[0], OutgoingPort::from(second_qb)),
                (subgraph_nodes[1], OutgoingPort::from(third_qb)),
            ];

            // The input boundary
            let nu_inp = repl_hugr
                .all_linked_inputs(repl_hugr_inp)
                .zip(subgraph_input)
                .collect();
            // The output boundary
            let nu_out = subgraph_output
                .into_iter()
                .zip(repl_hugr.node_inputs(repl_hugr_out))
                .collect();

            (repl_hugr, nu_inp, nu_out)
        }
        _ => unreachable!(),
    };

    let repl = SimpleReplacement::new(subgraph, repl_hugr, nu_inp, nu_out);
    Ok(repl)
}

#[test]
fn walker_example() {
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
        println!("nodes added:");
        println!(
            "{:?}\n",
            state_space.inserted_nodes(commit_id).collect_vec()
        );
    }

    // assert_eq!(state_space.all_commit_ids().count(), 13);

    let empty_commits = state_space
        .all_commit_ids()
        // .filter(|&id| state_space.commit_hugr(id).num_nodes() == 3)
        .filter(|&id| {
            state_space
                .inserted_nodes(id)
                .filter(|&n| state_space.get_optype(n) == &h_gate().into())
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

    let n_cz = empty_hugr
        .nodes()
        .filter(|&n| empty_hugr.get_optype(n) == &cz_gate().into())
        .count();
    let n_h = empty_hugr
        .nodes()
        .filter(|&n| empty_hugr.get_optype(n) == &h_gate().into())
        .count();
    assert_eq!(n_cz, 2);
    assert_eq!(n_h, 4);
}
