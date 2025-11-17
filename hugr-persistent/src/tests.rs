use std::collections::{BTreeMap, HashMap};

use derive_more::derive::{From, Into};
use hugr_core::{
    IncomingPort, Node, OutgoingPort, SimpleReplacement,
    builder::{DFGBuilder, Dataflow, DataflowHugr, endo_sig, inout_sig},
    extension::prelude::bool_t,
    hugr::{Hugr, HugrView, patch::Patch, views::SiblingSubgraph},
    ops::handle::NodeHandle,
    std_extensions::logic::LogicOp,
};
use rstest::*;

use crate::{
    Commit, CommitStateSpace, PatchNode, PersistentHugr, PersistentReplacement,
    state_space::CommitId,
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
fn simple_hugr() -> (Hugr, [Node; 3]) {
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

    (hugr, [not0.node(), not1.node(), and.node()])
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

    // Create the mappings
    let mut nu_inp = HashMap::new();
    nu_inp.insert(
        (not1.node(), IncomingPort::from(0)),
        (node_to_replace, IncomingPort::from(0)),
    );

    let mut nu_out = HashMap::new();
    nu_out.insert(
        (node_to_replace, OutgoingPort::from(0)),
        IncomingPort::from(0),
    );

    // Create the subgraph with the single node
    let subgraph = SiblingSubgraph::try_from_nodes(vec![node_to_replace], hugr).unwrap();

    // Create the replacement
    SimpleReplacement::try_new(subgraph, hugr, replacement_hugr).unwrap()
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
    nu_out.insert((and_gate, OutgoingPort::from(0)), IncomingPort::from(0));

    // Create subgraph with both the AND gate and NOT0 node
    let subgraph = SiblingSubgraph::try_from_nodes(vec![not_node, and_gate], &hugr).unwrap();

    SimpleReplacement::try_new(subgraph, hugr, replacement_hugr).unwrap()
}

/// A state space for testing purposes, along with commits within it.
pub struct TestStateSpace {
    #[allow(dead_code)] // this needs to stay in scope for the commits to be alive
    state_space: CommitStateSpace,
    // these commits typically borrow from the state space -- we replace the
    // lifetime with a dummy static lifetime, fixed in the getter methods
    commits: Vec<Commit<'static>>,
}

impl TestStateSpace {
    fn new<'a>(state_space: CommitStateSpace, commits: Vec<Commit<'a>>) -> Self {
        assert!(commits.iter().all(|c| c.state_space() == state_space));
        let commits = commits
            .into_iter()
            // SAFETY: the commits will be alive as long as the state space is
            // alive
            .map(|c| unsafe { c.upgrade_lifetime() })
            .collect();
        Self {
            state_space,
            commits,
        }
    }

    #[allow(dead_code)]
    pub fn state_space(&self) -> &CommitStateSpace {
        &self.state_space
    }

    pub fn commits<const N: usize>(&self) -> &[Commit<'_>; N] {
        TryFrom::try_from(self.commits.as_slice()).unwrap()
    }
}

/// Creates 4 commits in a shared state space on top of the base hugr
/// `simple_hugr`:
///
/// ```
///    ┌─────────┐
/// ───┤ (0) NOT ├─┐    ┌─────────┐
///    └─────────┘ └────┤         │
///    ┌─────────┐      │ (2) AND ├───
/// ───┤ (1) NOT ├──────┤         │
///    └─────────┘      └─────────┘
/// ```
///
/// Note that for the sake of these tests, we do not care about the semantics of
/// the rewrites. Unlike real-world use cases, the LHS and RHS here are not
/// functionally equivalent.
/// The state space will contain the following commits:
///
/// ```
/// [commit1]
///    ┌─────────┐                ┌─────────┐     ┌─────────┐
/// ───┤ (0) NOT ├───    -->   ───┤ (3) NOT ├─────┤ (4) NOT ├──────
///    └─────────┘                └─────────┘     └─────────┘
///
/// [commit2]
///    ┌─────────┐
/// ───┤ (4) NOT ├─┐    ┌─────────┐                  ┌─────────┐
///    └─────────┘ └────┤         │             ─────┤         │
///                     │ (2) AND ├───    -->        │ (5) XOR ├───
/// ────────────────────┤         │             ─────┤         │
///                     └─────────┘                  └─────────┘
///
/// [commit3]
///    ┌─────────┐
/// ───┤ (0) NOT ├─┐    ┌─────────┐                  ┌─────────┐
///    └─────────┘ └────┤         │             ─────┤         │
///                     │ (2) AND ├───    -->        │ (6) XOR ├───
/// ────────────────────┤         │             ─────┤         │
///                     └─────────┘                  └─────────┘
///
/// [commit4]
///    ┌─────────┐                ┌─────────┐     ┌─────────┐
/// ───┤ (1) NOT ├───    -->   ───┤ (7) NOT ├─────┤ (8) NOT ├──────
///    └─────────┘                └─────────┘     └─────────┘
/// ```
///
/// Viewed as a history of commits, the commits' hierarchy is as follows
///
/// ```
///                                 base
///                               /   |   \
///                              /    |    \
///                             /     |     \
///                            /      |      \
///                        commit1 commit3 commit4
///                           |
///                           |
///                           |
///                        commit2
/// ```
/// where
/// - `commit1` and `commit2` are incompatible with `commit3`
/// - `commit1` and `commit2` are disjoint with `commit4` (i.e. compatible),
/// - `commit2` depends on `commit1`
#[fixture]
pub(crate) fn test_state_space() -> TestStateSpace {
    let (base_hugr, [not0_node, not1_node, _and_node]) = simple_hugr();

    let state_space = CommitStateSpace::new();
    let base = state_space.try_set_base(base_hugr).unwrap();

    // Create first replacement (replace NOT0 with two NOT gates)
    let replacement1 = create_double_not_replacement(base.commit_hugr(), not0_node);

    // Add first commit to state space, replacing NOT0 with two NOT gates
    let commit1 = {
        let new_host = PersistentHugr::try_new([base.clone()]).unwrap();
        // translate replacement1 to patch nodes in the base commit of the state space
        let replacement1 = replacement1
            .map_host_nodes(|n| base.to_patch_node(n), &new_host)
            .unwrap();
        Commit::try_from_replacement(replacement1, &state_space).unwrap()
    };

    // Add second commit to state space, that applies on top of `commit1` and
    // replaces the second NOT gate and the (original) AND gate with an XOR gate
    let commit2 = {
        // Create second replacement (replace NOT+AND with XOR) that applies on
        // the result of the first
        let mut direct_hugr = base.commit_hugr().clone();
        let node_map = replacement1
            .clone()
            .apply(&mut direct_hugr)
            .unwrap()
            .node_map;
        let replacement2 = create_not_and_to_xor_replacement(&direct_hugr);

        // The hard part: figure out the node map between nodes in `direct_hugr`
        // and nodes in the state space
        let inv_node_map = {
            let mut inv = BTreeMap::new();
            for (repl_node, hugr_node) in node_map {
                inv.insert(hugr_node, repl_node);
            }
            inv
        };
        let to_patch_node = {
            |n| {
                if let Some(&n) = inv_node_map.get(&n) {
                    // node was replaced by commit1
                    commit1.to_patch_node(n)
                } else {
                    // node is in base hugr
                    base.to_patch_node(n)
                }
            }
        };

        // translate replacement2 to patch nodes
        let new_host = PersistentHugr::try_new([commit1.clone()]).unwrap();
        let replacement2 = replacement2
            .map_host_nodes(to_patch_node, &new_host)
            .unwrap();
        Commit::try_from_replacement(replacement2, &state_space).unwrap()
    };

    // Create a third commit that will conflict with `commit1`, replacing NOT0
    // and AND with XOR
    let commit3 = {
        let replacement3 = create_not_and_to_xor_replacement(base.commit_hugr());
        let new_host = PersistentHugr::try_new([commit1.clone()]).unwrap();
        let replacement3 = replacement3
            .map_host_nodes(|n| base.to_patch_node(n), &new_host)
            .unwrap();
        Commit::try_from_replacement(replacement3, &state_space).unwrap()
    };

    // Create a fourth commit that is disjoint from `commit1`, replacing NOT1
    // with two NOT gates
    let commit4 = {
        let replacement4 = create_double_not_replacement(base.commit_hugr(), not1_node);
        let new_host = PersistentHugr::try_new([commit1.clone()]).unwrap();
        let replacement4 = replacement4
            .map_host_nodes(|n| base.to_patch_node(n), &new_host)
            .unwrap();
        Commit::try_from_replacement(replacement4, &state_space).unwrap()
    };

    // [commit1, commit2, commit3, commit4]
    TestStateSpace::new(
        state_space.clone(),
        vec![commit1, commit2, commit3, commit4],
    )
}

#[fixture]
pub(super) fn persistent_hugr_empty_child() -> (PersistentHugr, [CommitId; 2], [PatchNode; 3]) {
    let (triple_not_hugr, not_nodes) = {
        let mut dfg_builder = DFGBuilder::new(endo_sig(bool_t())).unwrap();
        let [mut w] = dfg_builder.input_wires_arr();
        let not_nodes = [(); 3].map(|()| {
            let handle = dfg_builder.add_dataflow_op(LogicOp::Not, vec![w]).unwrap();
            [w] = handle.outputs_arr();
            handle.node()
        });
        (
            dfg_builder.finish_hugr_with_outputs([w]).unwrap(),
            not_nodes,
        )
    };
    let mut hugr = PersistentHugr::with_base(triple_not_hugr);
    let empty_hugr = {
        let dfg_builder = DFGBuilder::new(endo_sig(bool_t())).unwrap();
        let inputs = dfg_builder.input_wires();
        dfg_builder.finish_hugr_with_outputs(inputs).unwrap()
    };
    let subg_nodes = [PatchNode(hugr.base(), not_nodes[1])];
    let repl = PersistentReplacement::try_new(
        SiblingSubgraph::try_from_nodes(subg_nodes, &hugr).unwrap(),
        &hugr,
        empty_hugr,
    )
    .unwrap();

    let empty_commit = hugr.try_add_replacement(repl).unwrap();
    let base_commit = hugr.base();
    (
        hugr,
        [base_commit, empty_commit],
        not_nodes.map(|n| PatchNode(base_commit, n)),
    )
}

#[rstest]
fn test_successive_replacements(test_state_space: TestStateSpace) {
    let [commit1, commit2, _commit3, _commit4] = test_state_space.commits();
    let (mut hugr, [not0_node, _not1_node, _and_node]) = simple_hugr();

    // Apply first replacement (replace NOT0 with two NOT gates)
    let replacement1 = create_double_not_replacement(&hugr, not0_node);
    replacement1.clone().apply(&mut hugr).unwrap();

    // Apply second replacement (replace NOT+AND with XOR)
    let replacement2 = create_not_and_to_xor_replacement(&hugr);
    replacement2.clone().apply(&mut hugr).unwrap();

    // Create a persistent hugr
    let persistent_hugr = PersistentHugr::try_new([commit1.clone(), commit2.clone()])
        .expect("commit1 and commit2 are compatible");

    // Get the final hugr from the persistent context
    let persistent_final_hugr = persistent_hugr.to_hugr();

    // Check we have the expected number of patches (original + 2 replacements)
    assert_eq!(persistent_hugr.all_commit_ids().count(), 3);

    assert_eq!(hugr.validate(), Ok(()));
    assert_eq!(persistent_final_hugr.validate(), Ok(()));
    // TODO: use node-invariant equivalence check, e.g. hash-based comparison
    assert_eq!(
        hugr.mermaid_string(),
        persistent_final_hugr.mermaid_string()
    );
}

#[rstest]
fn test_conflicting_replacements(test_state_space: TestStateSpace) {
    let [commit1, _commit2, commit3, _commit4] = test_state_space.commits();
    let state_space = commit1.state_space();
    let (hugr, [not0_node, _not1_node, _and_node]) = simple_hugr();

    // Apply first replacement directly to a clone
    let hugr1 = {
        let mut hugr = hugr.clone();
        let replacement1 = create_double_not_replacement(&hugr, not0_node);
        replacement1.apply(&mut hugr).unwrap();
        hugr
    };

    // Apply second replacement directly to another clone
    let hugr2 = {
        let mut hugr = hugr.clone();
        let replacement2 = create_not_and_to_xor_replacement(&hugr);
        replacement2.apply(&mut hugr).unwrap();
        hugr
    };

    // Create a persistent hugr and add first replacement
    let persistent_hugr1 = PersistentHugr::try_new([commit1.clone()]).unwrap();

    // Create another persistent hugr and add second replacement
    let persistent_hugr2 = PersistentHugr::try_new([commit3.clone()]).unwrap();

    // Both individual replacements should be valid
    assert_eq!(persistent_hugr1.to_hugr().validate(), Ok(()));
    assert_eq!(persistent_hugr2.to_hugr().validate(), Ok(()));

    // But trying to create a history with both replacements should fail
    let result = state_space.try_create(
        persistent_hugr1
            .all_commit_ids()
            .chain(persistent_hugr2.all_commit_ids()),
    );
    assert!(
        result.is_err(),
        "Creating history with conflicting patches should fail"
    );

    assert_eq!(
        hugr1.mermaid_string(),
        persistent_hugr1.to_hugr().mermaid_string()
    );

    assert_eq!(
        hugr2.mermaid_string(),
        persistent_hugr2.to_hugr().mermaid_string()
    );
}

#[rstest]
fn test_disjoint_replacements(test_state_space: TestStateSpace) {
    let [commit1, _commit2, _commit3, commit4] = test_state_space.commits();
    let (mut hugr, [not0_node, not1_node, _and_node]) = simple_hugr();

    // Create and apply non-overlapping replacements for NOT0 and NOT1
    let replacement1 = create_double_not_replacement(&hugr, not0_node);
    let replacement2 = create_double_not_replacement(&hugr, not1_node);
    replacement1.clone().apply(&mut hugr).unwrap();
    replacement2.clone().apply(&mut hugr).unwrap();

    // Create a persistent hugr and add both replacements
    let persistent_hugr = PersistentHugr::try_new([commit1.clone(), commit4.clone()]).unwrap();

    // Get the final hugr
    let persistent_final_hugr = persistent_hugr.to_hugr();

    // Both hugrs should be valid
    assert_eq!(hugr.validate(), Ok(()));
    assert_eq!(persistent_final_hugr.validate(), Ok(()));

    // We should have 3 patches (base + 2 replacements)
    assert_eq!(persistent_hugr.all_commit_ids().count(), 3);

    // TODO: use node-invariant equivalence check, e.g. hash-based comparison
    assert_eq!(
        hugr.mermaid_string(),
        persistent_final_hugr.mermaid_string()
    );
}

#[rstest]
fn test_try_add_replacement(test_state_space: TestStateSpace) {
    let [commit1, commit2, commit3, commit4] = test_state_space.commits();

    // Create a persistent hugr and add first replacement
    let persistent_hugr = PersistentHugr::try_new([commit1.clone(), commit2.clone()]).unwrap();

    {
        let mut persistent_hugr = persistent_hugr.clone();
        let repl4 = commit4.replacement().unwrap();
        let result = persistent_hugr.try_add_replacement(repl4.clone());
        assert!(
            result.is_ok(),
            "[commit1, commit2] + [commit4] are compatible. Got {result:?}"
        );
        let hugr = persistent_hugr.to_hugr();
        let exp_hugr = PersistentHugr::try_new([commit1.clone(), commit2.clone(), commit4.clone()])
            .unwrap()
            .to_hugr();
        assert_eq!(hugr.mermaid_string(), exp_hugr.mermaid_string());
    }

    {
        let mut persistent_hugr = persistent_hugr.clone();
        let repl3 = commit3.replacement().unwrap();
        let result = persistent_hugr.try_add_replacement(repl3.clone());
        assert!(
            result.is_err(),
            "[commit1, commit2] + [commit3] are incompatible. Got {result:?}"
        );
    }
}

// same test as above, but using try_add_commit instead of try_add_replacement
#[rstest]
fn test_try_add_commit(test_state_space: TestStateSpace) {
    let [commit1, commit2, commit3, commit4] = test_state_space.commits();
    let state_space = commit1.state_space();

    // Create a persistent hugr and add first replacement
    let persistent_hugr = PersistentHugr::try_new([commit1.clone(), commit2.clone()]).unwrap();

    {
        let mut persistent_hugr = persistent_hugr.clone();
        let repl4 = commit4.replacement().unwrap().clone();
        let new_commit = Commit::try_from_replacement(repl4, &state_space).unwrap();
        let new_commit4_id = persistent_hugr
            .try_add_commit(new_commit)
            .expect("commit4 is compatible");
        let new_commit4 = persistent_hugr.get_commit(new_commit4_id);

        assert_eq!(new_commit4.inserted_nodes().count(), 2);
    }
    {
        let mut persistent_hugr = persistent_hugr.clone();
        let repl3 = commit3.replacement().unwrap().clone();
        let new_commit = Commit::try_from_replacement(repl3, &state_space).unwrap();
        persistent_hugr
            .try_add_commit(new_commit)
            .expect_err("commit3 is incompatible with [commit1, commit2]");
    }
}

/// A Hugr that serialises with no extensions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, From, Into)]
pub(crate) struct WrappedHugr {
    #[serde(with = "serial")]
    pub hugr: Hugr,
}

mod serial {
    use hugr_core::envelope::EnvelopeConfig;
    use hugr_core::std_extensions::STD_REG;
    use serde::Deserialize;

    use super::*;

    pub(crate) fn serialize<S>(hugr: &Hugr, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut str = hugr
            .store_str(EnvelopeConfig::text())
            .map_err(serde::ser::Error::custom)?;
        // TODO: replace this with a proper hugr hash (see https://github.com/CQCL/hugr/issues/2091)
        remove_encoder_version(&mut str);
        serializer.serialize_str(&str)
    }

    fn remove_encoder_version(str: &mut String) {
        // Remove encoder version information for consistent test output
        let encoder_pattern = r#""encoder":"hugr-rs v"#;
        if let Some(start) = str.find(encoder_pattern) {
            if let Some(end) = str[start..].find(r#"","#) {
                let end = start + end + 2; // +2 for the `",` part
                str.replace_range(start..end, "");
            }
        }
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Hugr, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let str = String::deserialize(deserializer)?;
        Hugr::load_str(str, Some(&STD_REG)).map_err(serde::de::Error::custom)
    }
}
