//! A test of the walker as it would typically be used by a user in practice.

use std::collections::{BTreeSet, VecDeque};

use itertools::{Either, Itertools};

use hugr_core::{
    Hugr, HugrView, IncomingPort, OutgoingPort, Port, PortIndex,
    builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig},
    extension::prelude::qb_t,
    ops::OpType,
    types::EdgeKind,
};

use hugr_persistent::{
    Commit, CommitStateSpace, PersistentHugr, PersistentWire, PinnedSubgraph, Walker,
};

/// The maximum commit depth that we will consider in this example
const MAX_COMMITS: usize = 4;

// We define a HUGR extension within this file, with CZ and H gates. Normally,
// you would use an existing extension (e.g. as provided by tket2).
use walker_example_extension::cz_gate;
mod walker_example_extension {
    use std::sync::{Arc, LazyLock};

    use hugr_core::Extension;
    use hugr_core::extension::ExtensionId;
    use hugr_core::ops::{ExtensionOp, OpName};
    use hugr_core::types::{FuncValueType, PolyFuncTypeRV};

    use semver::Version;

    use super::*;

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
                        OpName::new_inline("CZ"),
                        "CZ".into(),
                        two_qb_func(),
                        extension_ref,
                    )
                    .unwrap();
            },
        )
    }

    /// Quantum extension definition.
    static EXTENSION: LazyLock<Arc<Extension>> = LazyLock::new(extension);

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

fn empty_2qb_hugr(flip_args: bool) -> Hugr {
    let builder = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();
    let [mut q0, mut q1] = builder.input_wires_arr();
    if flip_args {
        (q0, q1) = (q1, q0);
    }
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

/// Traverse all commits in state space, enqueueing all outgoing wires of
/// CZ nodes
fn enqueue_all<'a>(
    queue: &mut VecDeque<(PersistentWire, Walker<'a>)>,
    all_commits: &[Commit],
    state_space: &'a CommitStateSpace,
) {
    for commit in all_commits {
        let cz_nodes = commit
            .inserted_nodes()
            .filter(|&n| commit.get_optype(n) == &cz_gate().into());
        for node in cz_nodes {
            let walker = Walker::from_pinned_node(commit.to_patch_node(node), state_space);
            if walker.as_hugr_view().all_commit_ids().count() > MAX_COMMITS {
                continue;
            }
            for outport in commit.node_outputs(node) {
                if !matches!(
                    commit.get_optype(node).port_kind(outport),
                    Some(EdgeKind::Value(_))
                ) {
                    continue;
                }
                let wire = walker.get_wire(commit.to_patch_node(node), outport);
                queue.push_back((wire, walker.clone()));
            }
        }
    }
}

fn explore_state_space<'a>(
    base_commit: Commit<'a>,
    state_space: &'a CommitStateSpace,
) -> Vec<Commit<'a>> {
    let mut all_commits = vec![base_commit.clone()];

    let mut wire_queue = VecDeque::new();
    let mut added_patches = BTreeSet::new();

    enqueue_all(&mut wire_queue, &all_commits, state_space);

    while let Some((wire, walker)) = wire_queue.pop_front() {
        if !walker.is_complete(&wire, None) {
            // expand the wire in all possible ways
            let (pinned_node, pinned_port) = walker
                .wire_pinned_ports(&wire, None)
                .next()
                .expect("at least one port was already pinned");
            assert!(
                walker.as_hugr_view().contains_node(pinned_node),
                "pinned node is deleted"
            );
            for subwalker in walker.expand(&wire, None) {
                assert!(
                    subwalker.as_hugr_view().contains_node(pinned_node),
                    "pinned node {pinned_node:?} is deleted",
                );
                wire_queue.push_back((subwalker.get_wire(pinned_node, pinned_port), subwalker));
            }
        } else {
            // we have a complete wire, so we can commute the CZ gates (or
            // cancel them out)

            let patch_nodes: BTreeSet<_> = walker
                .wire_pinned_ports(&wire, None)
                .map(|(n, _)| n)
                .collect();
            // check that the patch applies to more than one commit (or the base),
            // otherwise we have infinite commutations back and forth
            let patch_owners: BTreeSet<_> = patch_nodes.iter().map(|n| n.0).collect();
            if patch_owners.len() <= 1 && !patch_owners.contains(&base_commit.id()) {
                continue;
            }
            // check further that the same patch was not already added to `state_space`
            // (we currently do not have automatic deduplication)
            if !added_patches.insert(patch_nodes.clone()) {
                continue;
            }

            let Some(new_commit) = create_commit(wire, &walker) else {
                continue;
            };

            assert_eq!(
                new_commit.deleted_parent_nodes().collect::<BTreeSet<_>>(),
                patch_nodes
            );

            all_commits.push(new_commit);

            // enqueue new wires added by the replacement
            // (this will also add a lot of already visited wires, but they will
            // be deduplicated)
            enqueue_all(&mut wire_queue, &all_commits, state_space);
        }
    }

    all_commits
}

fn create_commit<'a>(wire: PersistentWire, walker: &Walker<'a>) -> Option<Commit<'a>> {
    let hugr = walker.clone().into_persistent_hugr();
    let (out_node, _) = wire
        .single_outgoing_port(&hugr)
        .expect("outgoing port was already pinned (and is unique)");

    let (in_node, _) = wire
        .all_incoming_ports(&hugr)
        .exactly_one()
        .ok()
        .expect("all our wires have exactly one incoming port");

    if hugr.get_optype(out_node) != &cz_gate().into()
        || hugr.get_optype(in_node) != &cz_gate().into()
    {
        // one of the nodes we have matched is (presumably) an input or output gate
        // => skip
        return None;
    }

    // figure out whether the two CZ gates act on the same qubits (iff the
    // the only outgoing neighbour of the first CZ is the second CZ gate)
    let all_edges = hugr.node_connections(out_node, in_node).collect_vec();
    let n_shared_qubits = all_edges.len();

    match n_shared_qubits {
        2 => {
            // out_node and in_node act on the same qubits
            // => replace the two CZ gates with the empty 2qb HUGR

            // If the two CZ gates have flipped port ordering, we need to insert
            // a swap gate
            let add_swap = all_edges[0][0].index() != all_edges[0][1].index();

            // Get the wires between the two CZ gates
            let wires = all_edges
                .into_iter()
                .map(|[out_port, _]| walker.get_wire(out_node, out_port));

            // Create the commit
            walker.try_create_commit(
                PinnedSubgraph::try_from_wires(wires, walker).unwrap(),
                empty_2qb_hugr(add_swap),
                |_, port| {
                    // the incoming/outgoing ports of the subgraph map trivially to the empty 2qb
                    // HUGR
                    let dir = port.direction();
                    Port::new(dir.reverse(), port.index())
                },
            )
        }
        1 => {
            // out_node and in_node share just one qubit
            // => commute the two CZ gates past each other
            let repl_hugr = two_cz_3qb_hugr();

            // Need to figure out the permutation of the qubits
            // => establish which qubit is shared between the two CZ gates
            let [out_port, in_port] = all_edges.into_iter().exactly_one().unwrap();
            let shared_qb_out = out_port.index();
            let shared_qb_in = in_port.index();

            walker.try_create_commit(
                PinnedSubgraph::try_from_wires([wire], walker).unwrap(),
                repl_hugr,
                |node, port| {
                    // map the incoming/outgoing ports of the subgraph to the replacement as
                    // follows:
                    //  - the first qubit is the one that is shared between the two CZ gates
                    //  - the second qubit only touches the first CZ (out_node)
                    //  - the third qubit only touches the second CZ (in_node)
                    match port.as_directed() {
                        Either::Left(incoming) => {
                            let in_boundary: [(_, IncomingPort); 3] = [
                                (out_node, shared_qb_out.into()),
                                (out_node, (1 - shared_qb_out).into()),
                                (in_node, (1 - shared_qb_in).into()),
                            ];
                            let out_index = in_boundary
                                .iter()
                                .position(|&(n, p)| n == node && p == incoming)
                                .expect("invalid input port");
                            OutgoingPort::from(out_index).into()
                        }
                        Either::Right(outgoing) => {
                            let out_boundary: [(_, OutgoingPort); 3] = [
                                (in_node, shared_qb_in.into()),
                                (out_node, (1 - shared_qb_out).into()),
                                (in_node, (1 - shared_qb_in).into()),
                            ];
                            let in_index = out_boundary
                                .iter()
                                .position(|&(n, p)| n == node && p == outgoing)
                                .expect("invalid output port");
                            IncomingPort::from(in_index).into()
                        }
                    }
                },
            )
        }
        _ => unreachable!(),
    }
    .ok()
}

#[ignore = "takes 10s (todo: optimise)"]
#[test]
fn walker_example() {
    let base_hugr = dfg_hugr();
    let state_space = CommitStateSpace::new();
    let base_commit = state_space.try_set_base(base_hugr).unwrap();

    let all_commits = explore_state_space(base_commit, &state_space);
    println!("n commits = {:?}", all_commits.len());

    for commit in all_commits.iter() {
        println!("========== Commit {:?} ============", commit.id());
        println!(
            "parents = {:?}",
            commit.parents().map(|p| p.id()).collect_vec()
        );
        println!(
            "nodes deleted in parents = {:?}",
            commit.deleted_parent_nodes().collect_vec()
        );
        println!("nodes added:");
        println!("{:?}\n", commit.inserted_nodes().collect_vec());
    }

    let empty_commits = all_commits
        .iter()
        .filter(|cm| cm.inserted_nodes().count() == 0)
        .collect_vec();

    // there should be a combination of three empty commits that are compatible
    // and such that the resulting HUGR is empty
    let mut empty_hugr = None;
    for cs in empty_commits.iter().combinations(3) {
        let cs = cs.into_iter().copied().cloned();
        if let Ok(hugr) = PersistentHugr::try_new(cs) {
            empty_hugr = Some(hugr);
        }
    }

    let empty_hugr = empty_hugr.unwrap().to_hugr();

    // The empty hugr should have 7 nodes:
    // module root, funcdef, 2 func IO, DFG root, 2 DFG IO
    assert_eq!(empty_hugr.num_nodes(), 7);
    assert_eq!(
        empty_hugr
            .nodes()
            .filter(|&n| {
                !matches!(
                    empty_hugr.get_optype(n),
                    OpType::Input(_)
                        | OpType::Output(_)
                        | OpType::FuncDefn(_)
                        | OpType::Module(_)
                        | OpType::DFG(_)
                )
            })
            .count(),
        0
    );
}
