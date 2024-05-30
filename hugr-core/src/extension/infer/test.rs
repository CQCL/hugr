use std::error::Error;

use super::*;
use crate::builder::{
    Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, ModuleBuilder,
};
use crate::extension::prelude::PRELUDE_REGISTRY;
use crate::extension::prelude::QB_T;
use crate::extension::ExtensionId;
use crate::hugr::{Hugr, HugrMut, NodeType};
use crate::macros::const_extension_ids;
use crate::ops::custom::OpaqueOp;
use crate::ops::{self, dataflow::IOTrait};
use crate::ops::{CustomOp, Lift, OpType};
#[cfg(feature = "extension_inference")]
use crate::{
    builder::test::closed_dfg_root_hugr,
    hugr::validate::ValidationError,
    ops::{dataflow::DataflowParent, handle::NodeHandle},
};

use crate::type_row;
use crate::types::{FunctionType, Type, TypeRow};

use cool_asserts::assert_matches;
use itertools::Itertools;
use portgraph::NodeIndex;

const NAT: Type = crate::extension::prelude::USIZE_T;

const_extension_ids! {
    const A: ExtensionId = "A";
    const B: ExtensionId = "B";
    const C: ExtensionId = "C";
    const UNKNOWN_EXTENSION: ExtensionId = "Unknown";
}

#[test]
// Build up a graph with some holes in its extension requirements, and infer
// them.
fn from_graph() -> Result<(), Box<dyn Error>> {
    let rs = ExtensionSet::from_iter([A, B, C]);
    let main_sig = FunctionType::new(type_row![NAT, NAT], type_row![NAT]).with_extension_delta(rs);

    let op = ops::DFG {
        signature: main_sig,
    };

    let root_node = NodeType::new_open(op);
    let mut hugr = Hugr::new(root_node);

    let input = ops::Input::new(type_row![NAT, NAT]);
    let output = ops::Output::new(type_row![NAT]);

    let input = hugr.add_node_with_parent(hugr.root(), input);
    let output = hugr.add_node_with_parent(hugr.root(), output);

    assert_matches!(hugr.get_io(hugr.root()), Some(_));

    let add_a_sig = FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(A);

    let add_b_sig = FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(B);

    let add_ab_sig = FunctionType::new(type_row![NAT], type_row![NAT])
        .with_extension_delta(ExtensionSet::from_iter([A, B]));

    let mult_c_sig = FunctionType::new(type_row![NAT, NAT], type_row![NAT]).with_extension_delta(C);

    let add_a = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: add_a_sig,
        },
    );
    let add_b = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: add_b_sig,
        },
    );
    let add_ab = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: add_ab_sig,
        },
    );
    let mult_c = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: mult_c_sig,
        },
    );

    hugr.connect(input, 0, add_a, 0);
    hugr.connect(add_a, 0, add_b, 0);
    hugr.connect(add_b, 0, mult_c, 0);

    hugr.connect(input, 1, add_ab, 0);
    hugr.connect(add_ab, 0, mult_c, 1);

    hugr.connect(mult_c, 0, output, 0);

    let solution = infer_extensions(&hugr)?;
    let empty = ExtensionSet::new();
    let ab = ExtensionSet::from_iter([A, B]);
    assert_eq!(*solution.get(&(hugr.root())).unwrap(), empty);
    assert_eq!(*solution.get(&(mult_c)).unwrap(), ab);
    assert_eq!(*solution.get(&(add_ab)).unwrap(), empty);
    assert_eq!(*solution.get(&add_b).unwrap(), ExtensionSet::singleton(&A));
    Ok(())
}

#[test]
// Basic test that the `Plus` constraint works
fn plus() -> Result<(), InferExtensionError> {
    let hugr = Hugr::default();
    let mut ctx = UnificationContext::new(&hugr);

    let metas: Vec<Meta> = (2..8)
        .map(|i| {
            let meta = ctx.fresh_meta();
            ctx.extensions
                .insert((NodeIndex::new(i).into(), Direction::Incoming), meta);
            meta
        })
        .collect();

    ctx.solved.insert(metas[2], A.into());
    ctx.add_constraint(metas[1], Constraint::Equal(metas[2]));
    ctx.add_constraint(metas[0], Constraint::Plus(B.into(), metas[2]));
    ctx.add_constraint(metas[4], Constraint::Plus(C.into(), metas[0]));
    ctx.add_constraint(metas[3], Constraint::Equal(metas[4]));
    ctx.add_constraint(metas[5], Constraint::Equal(metas[0]));
    ctx.main_loop()?;

    let a = ExtensionSet::singleton(&A);
    let mut ab = a.clone();
    ab.insert(&B);
    let mut abc = ab.clone();
    abc.insert(&C);

    assert_eq!(ctx.get_solution(&metas[0]).unwrap(), &ab);
    assert_eq!(ctx.get_solution(&metas[1]).unwrap(), &a);
    assert_eq!(ctx.get_solution(&metas[2]).unwrap(), &a);
    assert_eq!(ctx.get_solution(&metas[3]).unwrap(), &abc);
    assert_eq!(ctx.get_solution(&metas[4]).unwrap(), &abc);
    assert_eq!(ctx.get_solution(&metas[5]).unwrap(), &ab);

    Ok(())
}

#[cfg(feature = "extension_inference")]
#[test]
// This generates a solution that causes validation to fail
// because of a missing lift node
fn missing_lift_node() {
    let mut hugr = Hugr::new(NodeType::new_pure(ops::DFG {
        signature: FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(A),
    }));

    let input = hugr.add_node_with_parent(
        hugr.root(),
        NodeType::new_pure(ops::Input {
            types: type_row![NAT],
        }),
    );

    let output = hugr.add_node_with_parent(
        hugr.root(),
        NodeType::new_pure(ops::Output {
            types: type_row![NAT],
        }),
    );

    hugr.connect(input, 0, output, 0);

    // Fail to catch the actual error because it's a difference between I/O
    // nodes and their parents and `report_mismatch` isn't yet smart enough
    // to handle that.
    assert_matches!(
        hugr.update_validate(&PRELUDE_REGISTRY),
        Err(ValidationError::CantInfer(_))
    );
}

#[test]
// Tests that we can succeed even when all variables don't have concrete
// extension sets, and we have an open variable at the start of the graph.
fn open_variables() -> Result<(), InferExtensionError> {
    let mut ctx = UnificationContext::new(&Hugr::default());
    let a = ctx.fresh_meta();
    let b = ctx.fresh_meta();
    let ab = ctx.fresh_meta();
    // Some nonsense so that the constraints register as "live"
    ctx.extensions
        .insert((NodeIndex::new(2).into(), Direction::Outgoing), a);
    ctx.extensions
        .insert((NodeIndex::new(3).into(), Direction::Outgoing), b);
    ctx.extensions
        .insert((NodeIndex::new(4).into(), Direction::Incoming), ab);
    ctx.variables.insert(a);
    ctx.variables.insert(b);
    ctx.add_constraint(ab, Constraint::Plus(A.into(), b));
    ctx.add_constraint(ab, Constraint::Plus(B.into(), a));
    let solution = ctx.main_loop()?;
    // We'll only find concrete solutions for the Incoming extension reqs of
    // the main node created by `Hugr::default`
    assert_eq!(solution.len(), 1);
    Ok(())
}

#[cfg(feature = "extension_inference")]
#[test]
// Infer the extensions on a child node with no inputs
fn dangling_src() -> Result<(), Box<dyn Error>> {
    let rs = ExtensionSet::singleton(&"R".try_into().unwrap());

    let mut hugr = closed_dfg_root_hugr(
        FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(rs.clone()),
    );

    let [input, output] = hugr.get_io(hugr.root()).unwrap();
    let add_r_sig =
        FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(rs.clone());

    let add_r = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: add_r_sig,
        },
    );

    // Dangling thingy
    let src_sig = FunctionType::new(type_row![], type_row![NAT]);

    let src = hugr.add_node_with_parent(hugr.root(), ops::DFG { signature: src_sig });

    let mult_sig = FunctionType::new(type_row![NAT, NAT], type_row![NAT]);
    // Mult has open extension requirements, which we should solve to be "R"
    let mult = hugr.add_node_with_parent(
        hugr.root(),
        ops::DFG {
            signature: mult_sig,
        },
    );

    hugr.connect(input, 0, add_r, 0);
    hugr.connect(add_r, 0, mult, 0);
    hugr.connect(src, 0, mult, 1);
    hugr.connect(mult, 0, output, 0);

    hugr.infer_extensions()?;
    assert_eq!(hugr.get_nodetype(src.node()).io_extensions().unwrap().1, rs);
    assert_eq!(
        hugr.get_nodetype(mult.node()).io_extensions().unwrap(),
        (rs.clone(), rs)
    );
    Ok(())
}

#[test]
fn resolve_test() -> Result<(), InferExtensionError> {
    let mut ctx = UnificationContext::new(&Hugr::default());
    let m0 = ctx.fresh_meta();
    let m1 = ctx.fresh_meta();
    let m2 = ctx.fresh_meta();
    ctx.add_constraint(m0, Constraint::Equal(m1));
    ctx.main_loop()?;
    let mid0 = ctx.resolve(m0);
    assert_eq!(ctx.resolve(m0), ctx.resolve(m1));
    ctx.add_constraint(mid0, Constraint::Equal(m2));
    ctx.main_loop()?;
    assert_eq!(ctx.resolve(m0), ctx.resolve(m2));
    assert_eq!(ctx.resolve(m1), ctx.resolve(m2));
    assert!(ctx.resolve(m0) != mid0);
    Ok(())
}

fn create_with_io(
    hugr: &mut Hugr,
    parent: Node,
    op: impl Into<OpType>,
    op_sig: FunctionType,
) -> Result<[Node; 3], Box<dyn Error>> {
    let op: OpType = op.into();

    let node = hugr.add_node_with_parent(parent, op);
    let input = hugr.add_node_with_parent(
        node,
        ops::Input {
            types: op_sig.input,
        },
    );
    let output = hugr.add_node_with_parent(
        node,
        ops::Output {
            types: op_sig.output,
        },
    );
    Ok([node, input, output])
}

#[cfg(feature = "extension_inference")]
#[test]
fn test_conditional_inference() -> Result<(), Box<dyn Error>> {
    fn build_case(
        hugr: &mut Hugr,
        conditional_node: Node,
        op: ops::Case,
        first_ext: ExtensionId,
        second_ext: ExtensionId,
    ) -> Result<Node, Box<dyn Error>> {
        let [case, case_in, case_out] =
            create_with_io(hugr, conditional_node, op.clone(), op.inner_signature())?;

        let lift1 = hugr.add_node_with_parent(
            case,
            Lift {
                type_row: type_row![NAT],
                new_extension: first_ext,
            },
        );

        let lift2 = hugr.add_node_with_parent(
            case,
            Lift {
                type_row: type_row![NAT],
                new_extension: second_ext,
            },
        );

        hugr.connect(case_in, 0, lift1, 0);
        hugr.connect(lift1, 0, lift2, 0);
        hugr.connect(lift2, 0, case_out, 0);

        Ok(case)
    }

    let sum_rows = vec![type_row![]; 2];
    let rs = ExtensionSet::from_iter([A, B]);

    let inputs = type_row![NAT];
    let outputs = type_row![NAT];

    let op = ops::Conditional {
        sum_rows,
        other_inputs: inputs.clone(),
        outputs: outputs.clone(),
        extension_delta: rs.clone(),
    };

    let mut hugr = Hugr::new(NodeType::new_pure(op));
    let conditional_node = hugr.root();

    let case_op = ops::Case {
        signature: FunctionType::new(inputs, outputs).with_extension_delta(rs),
    };
    let case0_node = build_case(&mut hugr, conditional_node, case_op.clone(), A, B)?;

    let case1_node = build_case(&mut hugr, conditional_node, case_op, B, A)?;

    hugr.infer_extensions()?;

    for node in [case0_node, case1_node, conditional_node] {
        assert_eq!(
            hugr.get_nodetype(node).io_extensions().unwrap().0,
            ExtensionSet::new()
        );
        assert_eq!(
            hugr.get_nodetype(node).io_extensions().unwrap().0,
            ExtensionSet::new()
        );
    }
    Ok(())
}

#[test]
fn extension_adding_sequence() -> Result<(), Box<dyn Error>> {
    let df_sig = FunctionType::new(type_row![NAT], type_row![NAT]);

    let mut hugr = Hugr::new(NodeType::new_open(ops::DFG {
        signature: df_sig
            .clone()
            .with_extension_delta(ExtensionSet::from_iter([A, B])),
    }));

    let root = hugr.root();
    let input = hugr.add_node_with_parent(
        root,
        ops::Input {
            types: type_row![NAT],
        },
    );
    let output = hugr.add_node_with_parent(
        root,
        ops::Output {
            types: type_row![NAT],
        },
    );

    // Make identical dataflow nodes which add extension requirement "A" or "B"
    let df_nodes: Vec<Node> = vec![A, A, B, B, A, B]
        .into_iter()
        .map(|ext| {
            let dfg_sig = df_sig.clone().with_extension_delta(ext.clone());
            let [node, input, output] = create_with_io(
                &mut hugr,
                root,
                ops::DFG {
                    signature: dfg_sig.clone(),
                },
                dfg_sig,
            )
            .unwrap();

            let lift = hugr.add_node_with_parent(
                node,
                Lift {
                    type_row: type_row![NAT],
                    new_extension: ext,
                },
            );

            hugr.connect(input, 0, lift, 0);
            hugr.connect(lift, 0, output, 0);

            node
        })
        .collect();

    // Connect nodes in order (0 -> 1 -> 2 ...)
    let nodes = [vec![input], df_nodes, vec![output]].concat();
    for (src, tgt) in nodes.into_iter().tuple_windows() {
        hugr.connect(src, 0, tgt, 0);
    }
    hugr.update_validate(&PRELUDE_REGISTRY)?;
    Ok(())
}

fn make_opaque(extension: impl Into<ExtensionId>, signature: FunctionType) -> CustomOp {
    ops::custom::OpaqueOp::new(extension.into(), "", "".into(), vec![], signature).into()
}

fn make_block(
    hugr: &mut Hugr,
    bb_parent: Node,
    inputs: TypeRow,
    sum_rows: impl IntoIterator<Item = TypeRow>,
    extension_delta: ExtensionSet,
) -> Result<Node, Box<dyn Error>> {
    let sum_rows: Vec<_> = sum_rows.into_iter().collect();
    let sum_type = Type::new_sum(sum_rows.clone());
    let dfb_sig = FunctionType::new(inputs.clone(), vec![sum_type])
        .with_extension_delta(extension_delta.clone());
    let dfb = ops::DataflowBlock {
        inputs,
        other_outputs: type_row![],
        sum_rows,
        extension_delta,
    };
    let op = make_opaque(UNKNOWN_EXTENSION, dfb_sig.clone());

    let [bb, bb_in, bb_out] = create_with_io(hugr, bb_parent, dfb, dfb_sig)?;

    let dfg = hugr.add_node_with_parent(bb, op);

    hugr.connect(bb_in, 0, dfg, 0);
    hugr.connect(dfg, 0, bb_out, 0);

    Ok(bb)
}

fn oneway(ty: Type) -> Vec<Type> {
    vec![Type::new_sum([vec![ty].into()])]
}

fn twoway(ty: Type) -> Vec<Type> {
    vec![Type::new_sum([vec![ty.clone()].into(), vec![ty].into()])]
}

fn create_entry_exit(
    hugr: &mut Hugr,
    root: Node,
    inputs: TypeRow,
    entry_variants: Vec<TypeRow>,
    entry_extensions: ExtensionSet,
    exit_types: impl Into<TypeRow>,
) -> Result<([Node; 3], Node), Box<dyn Error>> {
    let entry_sum = Type::new_sum(entry_variants.clone());
    let dfb = ops::DataflowBlock {
        inputs: inputs.clone(),
        other_outputs: type_row![],
        sum_rows: entry_variants,
        extension_delta: entry_extensions,
    };

    let exit = hugr.add_node_with_parent(
        root,
        ops::ExitBlock {
            cfg_outputs: exit_types.into(),
        },
    );

    let entry = hugr.add_node_before(exit, dfb);
    let entry_in = hugr.add_node_with_parent(entry, ops::Input { types: inputs });
    let entry_out = hugr.add_node_with_parent(
        entry,
        ops::Output {
            types: vec![entry_sum].into(),
        },
    );

    Ok(([entry, entry_in, entry_out], exit))
}

/// A CFG rooted hugr adding resources at each basic block.
/// Looks like this:
///
///          +-------------+
///          |    Entry    |
///          |  (Adds [A]) |
///          +-/---------\-+
///           /           \
///  +-------/-----+     +-\-------------+
///  |     BB0     |     |      BB1      |
///  | (Adds [BC]) |     |   (Adds [B])  |
///  +----\--------+     +---/------\----+
///        \                /        \
///         \              /          \
///          \       +----/-------+  +-\---------+
///           \      |   BB10     |  |  BB11     |
///            \     | (Adds [C]) |  | (Adds [C])|
///             \    +----+-------+  +/----------+
///              \        |          /
///         +-----\-------+---------/-+
///         |           Exit          |
///         +-------------------------+
#[test]
fn infer_cfg_test() -> Result<(), Box<dyn Error>> {
    let abc = ExtensionSet::from_iter([A, B, C]);
    let bc = ExtensionSet::from_iter([B, C]);

    let mut hugr = Hugr::new(NodeType::new_open(ops::CFG {
        signature: FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(abc),
    }));

    let root = hugr.root();

    let ([entry, entry_in, entry_out], exit) = create_entry_exit(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT], type_row![NAT]],
        A.into(),
        type_row![NAT],
    )?;

    let mkpred = hugr.add_node_with_parent(
        entry,
        make_opaque(
            A,
            FunctionType::new(vec![NAT], twoway(NAT)).with_extension_delta(A),
        ),
    );

    // Internal wiring for DFGs
    hugr.connect(entry_in, 0, mkpred, 0);
    hugr.connect(mkpred, 0, entry_out, 0);

    let bb0 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        bc.clone(),
    )?;

    let bb1 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT], type_row![NAT]],
        B.into(),
    )?;

    let bb10 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        C.into(),
    )?;

    let bb11 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        C.into(),
    )?;

    // CFG Wiring
    hugr.connect(entry, 0, bb0, 0);
    hugr.connect(entry, 0, bb1, 0);
    hugr.connect(bb1, 0, bb10, 0);
    hugr.connect(bb1, 0, bb11, 0);

    hugr.connect(bb0, 0, exit, 0);
    hugr.connect(bb10, 0, exit, 0);
    hugr.connect(bb11, 0, exit, 0);

    hugr.infer_extensions()?;

    Ok(())
}

/// A test case for a CFG with a node (BB2) which has multiple predecessors,
/// Like so:
///
///              +-----------------+
///              |      Entry      |
///              +------/--\-------+
///                    /    \
///                   /      \
///                  /        \
///       +---------/--+  +----\-------+
///       |     BB0    |  |    BB1     |
///       +--------\---+  +----/-------+
///                 \         /
///                  \       /
///                   \     /
///             +------\---/--------+
///             |        BB2        |
///             +---------+---------+
///                       |
///             +---------+----------+
///             |        Exit        |
///             +--------------------+
#[test]
fn multi_entry() -> Result<(), Box<dyn Error>> {
    let mut hugr = Hugr::new(NodeType::new_open(ops::CFG {
        signature: FunctionType::new(type_row![NAT], type_row![NAT]), // maybe add extensions?
    }));
    let cfg = hugr.root();
    let ([entry, entry_in, entry_out], exit) = create_entry_exit(
        &mut hugr,
        cfg,
        type_row![NAT],
        vec![type_row![NAT], type_row![NAT]],
        ExtensionSet::new(),
        type_row![NAT],
    )?;

    let entry_mid = hugr.add_node_with_parent(
        entry,
        make_opaque(UNKNOWN_EXTENSION, FunctionType::new(vec![NAT], twoway(NAT))),
    );

    hugr.connect(entry_in, 0, entry_mid, 0);
    hugr.connect(entry_mid, 0, entry_out, 0);

    let bb0 = make_block(
        &mut hugr,
        cfg,
        type_row![NAT],
        vec![type_row![NAT]],
        ExtensionSet::new(),
    )?;

    let bb1 = make_block(
        &mut hugr,
        cfg,
        type_row![NAT],
        vec![type_row![NAT]],
        ExtensionSet::new(),
    )?;

    let bb2 = make_block(
        &mut hugr,
        cfg,
        type_row![NAT],
        vec![type_row![NAT]],
        ExtensionSet::new(),
    )?;

    hugr.connect(entry, 0, bb0, 0);
    hugr.connect(entry, 0, bb1, 0);
    hugr.connect(bb0, 0, bb2, 0);
    hugr.connect(bb1, 0, bb2, 0);
    hugr.connect(bb2, 0, exit, 0);

    hugr.update_validate(&PRELUDE_REGISTRY)?;

    Ok(())
}

/// Create a CFG of the form below, with the extension deltas for `Entry`,
/// `BB1`, and `BB2` specified by arguments to the function.
///
///       +-----------+
///  +--->|   Entry   |
///  |    +-----+-----+
///  |          |
///  |          V
///  |    +------------+
///  |    |    BB1     +---+
///  |    +-----+------+   |
///  |          |          |
///  |          V          |
///  |    +------------+   |
///  +----+    BB2     |   |
///       +------------+   |
///                        |
///       +------------+   |
///       |    Exit    |<--+
///       +------------+
fn make_looping_cfg(
    entry_ext: ExtensionSet,
    bb1_ext: ExtensionSet,
    bb2_ext: ExtensionSet,
) -> Result<Hugr, Box<dyn Error>> {
    let hugr_delta = entry_ext
        .clone()
        .union(bb1_ext.clone())
        .union(bb2_ext.clone());

    let mut hugr = Hugr::new(NodeType::new_open(ops::CFG {
        signature: FunctionType::new(type_row![NAT], type_row![NAT])
            .with_extension_delta(hugr_delta),
    }));

    let root = hugr.root();

    let ([entry, entry_in, entry_out], exit) = create_entry_exit(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        entry_ext.clone(),
        type_row![NAT],
    )?;

    let entry_dfg = hugr.add_node_with_parent(
        entry,
        make_opaque(
            UNKNOWN_EXTENSION,
            FunctionType::new(vec![NAT], oneway(NAT)).with_extension_delta(entry_ext),
        ),
    );

    hugr.connect(entry_in, 0, entry_dfg, 0);
    hugr.connect(entry_dfg, 0, entry_out, 0);

    let bb1 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT], type_row![NAT]],
        bb1_ext.clone(),
    )?;

    let bb2 = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        bb2_ext.clone(),
    )?;

    hugr.connect(entry, 0, bb1, 0);
    hugr.connect(bb1, 0, bb2, 0);
    hugr.connect(bb1, 0, exit, 0);
    hugr.connect(bb2, 0, entry, 0);

    Ok(hugr)
}

#[test]
fn test_cfg_loops() -> Result<(), Box<dyn Error>> {
    let just_a = ExtensionSet::singleton(&A);
    let mut variants = Vec::new();
    for entry in [ExtensionSet::new(), just_a.clone()] {
        for bb1 in [ExtensionSet::new(), just_a.clone()] {
            for bb2 in [ExtensionSet::new(), just_a.clone()] {
                variants.push((entry.clone(), bb1.clone(), bb2.clone()));
            }
        }
    }
    for (bb0, bb1, bb2) in variants.into_iter() {
        let mut hugr = make_looping_cfg(bb0, bb1, bb2)?;
        hugr.update_validate(&PRELUDE_REGISTRY)?;
    }
    Ok(())
}

#[test]
/// A control flow graph consisting of an entry node and a single block
/// which adds a resource and links to both itself and the exit node.
fn simple_cfg_loop() -> Result<(), Box<dyn Error>> {
    let just_a = ExtensionSet::singleton(&A);

    let mut hugr = Hugr::new(NodeType::new(
        ops::CFG {
            signature: FunctionType::new(type_row![NAT], type_row![NAT]).with_extension_delta(A),
        },
        Some(A.into()),
    ));

    let root = hugr.root();

    let ([entry, entry_in, entry_out], exit) = create_entry_exit(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT]],
        ExtensionSet::new(),
        type_row![NAT],
    )?;

    let entry_mid = hugr.add_node_with_parent(
        entry,
        make_opaque(UNKNOWN_EXTENSION, FunctionType::new(vec![NAT], oneway(NAT))),
    );

    hugr.connect(entry_in, 0, entry_mid, 0);
    hugr.connect(entry_mid, 0, entry_out, 0);

    let bb = make_block(
        &mut hugr,
        root,
        type_row![NAT],
        vec![type_row![NAT], type_row![NAT]],
        just_a.clone(),
    )?;

    hugr.connect(entry, 0, bb, 0);
    hugr.connect(bb, 0, bb, 0);
    hugr.connect(bb, 0, exit, 0);

    hugr.update_validate(&PRELUDE_REGISTRY)?;

    Ok(())
}

/// This was stack-overflowing approx 50% of the time,
/// see https://github.com/CQCL/hugr/issues/633
#[test]
fn plus_on_self() -> Result<(), Box<dyn std::error::Error>> {
    let ext = ExtensionId::new("unknown1").unwrap();
    let ft = FunctionType::new_endo(type_row![QB_T, QB_T]).with_extension_delta(ext.clone());
    let mut dfg = DFGBuilder::new(ft.clone())?;

    // While https://github.com/CQCL/hugr/issues/388 is unsolved,
    // most operations have empty extension_reqs (not including their own extension).
    // Define some that do.
    let binop = CustomOp::new_opaque(OpaqueOp::new(
        ext.clone(),
        "2qb_op",
        String::new(),
        vec![],
        ft,
    ));
    let unary_sig = FunctionType::new_endo(type_row![QB_T]).with_extension_delta(ext.clone());
    let unop = CustomOp::new_opaque(OpaqueOp::new(
        ext,
        "1qb_op",
        String::new(),
        vec![],
        unary_sig,
    ));
    // Constrain q1,q2 as PLUS(ext1, inputs):
    let [q1, q2] = dfg
        .add_dataflow_op(binop.clone(), dfg.input_wires())?
        .outputs_arr();
    // Constrain q1 as PLUS(ext2, q2):
    let [q1] = dfg.add_dataflow_op(unop, [q1])?.outputs_arr();
    // Constrain q1 as EQUALS(q2) by using both together
    dfg.finish_hugr_with_outputs([q1, q2], &PRELUDE_REGISTRY)?;
    // The combined q1+q2 variable now has two PLUS constraints - on itself and the inputs.
    Ok(())
}

/// [plus_on_self] had about a 50% rate of failing with stack overflow.
/// So if we run 10 times, that would succeed about 1 run in 2^10, i.e. <0.1%
#[test]
fn plus_on_self_10_times() {
    [0; 10].iter().for_each(|_| plus_on_self().unwrap())
}

#[test]
// Test that logic for dealing with self-referential constraints doesn't
// fall over when a self-referencing group of metas also references a meta
// outside the group
fn sccs() {
    let hugr = Hugr::default();
    let mut ctx = UnificationContext::new(&hugr);
    // Make a strongly-connected component (loop)
    let m1 = ctx.fresh_meta();
    let m2 = ctx.fresh_meta();
    let m3 = ctx.fresh_meta();
    ctx.add_constraint(m1, Constraint::Plus(ExtensionSet::singleton(&A), m3));
    ctx.add_constraint(m2, Constraint::Plus(ExtensionSet::singleton(&B), m1));
    ctx.add_constraint(m3, Constraint::Plus(ExtensionSet::singleton(&A), m2));
    // And a second scc
    let m4 = ctx.fresh_meta();
    let m5 = ctx.fresh_meta();
    ctx.add_constraint(m4, Constraint::Plus(ExtensionSet::singleton(&C), m5));
    ctx.add_constraint(m5, Constraint::Plus(ExtensionSet::singleton(&C), m4));
    // Make second component depend upon first
    ctx.add_constraint(
        m4,
        Constraint::Plus(ExtensionSet::singleton(&UNKNOWN_EXTENSION), m3),
    );
    ctx.variables.insert(m1);
    ctx.variables.insert(m4);
    ctx.instantiate_variables();
    assert_eq!(
        ctx.get_solution(&m1),
        Some(&ExtensionSet::from_iter([A, B]))
    );
    assert_eq!(
        ctx.get_solution(&m4),
        Some(&ExtensionSet::from_iter([A, B, C, UNKNOWN_EXTENSION]))
    );
}

#[test]
/// Note: This test is relying on the builder's `define_function` doing the
/// right thing: it takes input resources via a [`Signature`], which it passes
/// to `create_with_io`, creating concrete resource sets.
/// Inference can still fail for a valid FuncDefn hugr created without using
/// the builder API.
fn simple_funcdefn() -> Result<(), Box<dyn Error>> {
    let mut builder = ModuleBuilder::new();
    let mut func_builder = builder.define_function(
        "F",
        FunctionType::new(vec![NAT], vec![NAT])
            .with_extension_delta(A)
            .into(),
    )?;

    let [w] = func_builder.input_wires_arr();
    let lift = func_builder.add_dataflow_op(
        Lift {
            type_row: type_row![NAT],
            new_extension: A,
        },
        [w],
    )?;
    let [w] = lift.outputs_arr();
    func_builder.finish_with_outputs([w])?;
    builder.finish_prelude_hugr()?;
    Ok(())
}

#[cfg(feature = "extension_inference")]
#[test]
fn funcdefn_signature_mismatch() -> Result<(), Box<dyn Error>> {
    let mut builder = ModuleBuilder::new();
    let mut func_builder = builder.define_function(
        "F",
        FunctionType::new(vec![NAT], vec![NAT])
            .with_extension_delta(A)
            .into(),
    )?;

    let [w] = func_builder.input_wires_arr();
    let lift = func_builder.add_dataflow_op(
        Lift {
            type_row: type_row![NAT],
            new_extension: B,
        },
        [w],
    )?;
    let [w] = lift.outputs_arr();
    func_builder.finish_with_outputs([w])?;
    let result = builder.finish_prelude_hugr();
    assert_matches!(
        result,
        Err(ValidationError::CantInfer(
            InferExtensionError::MismatchedConcreteWithLocations { .. }
        ))
    );
    Ok(())
}

#[cfg(feature = "extension_inference")]
#[test]
// Test that the difference between a FuncDefn's input and output nodes is being
// constrained to be the same as the extension delta in the FuncDefn signature.
// The FuncDefn here is declared to add resource "A", but its body just wires
// the input to the output.
fn funcdefn_signature_mismatch2() -> Result<(), Box<dyn Error>> {
    let mut builder = ModuleBuilder::new();
    let func_builder = builder.define_function(
        "F",
        FunctionType::new(vec![NAT], vec![NAT])
            .with_extension_delta(A)
            .into(),
    )?;

    let [w] = func_builder.input_wires_arr();
    func_builder.finish_with_outputs([w])?;
    let result = builder.finish_prelude_hugr();
    assert_matches!(result, Err(ValidationError::CantInfer(..)));
    Ok(())
}
