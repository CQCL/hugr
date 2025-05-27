use portgraph::PortOffset;
use rstest::{fixture, rstest};

use crate::{
    Hugr, HugrView,
    builder::{
        BuildError, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr, HugrBuilder,
        endo_sig, inout_sig,
    },
    extension::prelude::qb_t,
    ops::{
        Value,
        handle::{DataflowOpID, NodeHandle},
    },
    std_extensions::logic::LogicOp,
    type_row,
    types::Signature,
    utils::test_quantum_extension::cx_gate,
};

/// A Dataflow graph from two qubits to two qubits that applies two CX operations on them.
///
/// Returns the Hugr and the two CX node ids.
#[fixture]
pub(crate) fn sample_hugr() -> (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>) {
    let mut dfg = DFGBuilder::new(endo_sig(vec![qb_t(), qb_t()])).unwrap();

    let [q1, q2] = dfg.input_wires_arr();

    let n1 = dfg.add_dataflow_op(cx_gate(), [q1, q2]).unwrap();
    let [q1, q2] = n1.outputs_arr();
    let n2 = dfg.add_dataflow_op(cx_gate(), [q2, q1]).unwrap();
    dfg.add_other_wire(n1.node(), n2.node());

    (dfg.finish_hugr_with_outputs(n2.outputs()).unwrap(), n1, n2)
}

#[rstest]
fn node_connections(
    sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>),
) -> Result<(), BuildError> {
    let (h, n1, n2) = sample_hugr;
    let connections: Vec<_> = h.node_connections(n1.node(), n2.node()).collect();

    assert_eq!(
        &connections[..],
        &[
            [
                PortOffset::new_outgoing(0).into(),
                PortOffset::new_incoming(1).into()
            ],
            [
                PortOffset::new_outgoing(1).into(),
                PortOffset::new_incoming(0).into()
            ],
            // order edge
            [
                PortOffset::new_outgoing(2).into(),
                PortOffset::new_incoming(2).into()
            ],
        ]
    );
    Ok(())
}

/// Render some hugrs into dot format.
///
/// The first parameter `test_name` is required due to insta and rstest limitations.
/// See https://github.com/la10736/rstest/issues/183
#[rstest]
#[case::dfg("dot_dfg", sample_hugr().0)]
#[case::cfg("dot_cfg", crate::builder::test::simple_cfg_hugr())]
#[case::empty_dfg("dot_empty_dfg", crate::builder::test::simple_dfg_hugr())]
#[case::func("dot_func", crate::builder::test::simple_funcdef_hugr())]
#[case::module("dot_module", crate::builder::test::simple_module_hugr())]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn dot_string(#[case] test_name: &str, #[case] h: Hugr) {
    insta::assert_snapshot!(test_name, h.dot_string());
}

/// Render some hugrs into mermaid format.
///
/// The first parameter `test_name` is required due to insta and rstest limitations.
/// See https://github.com/la10736/rstest/issues/183
#[rstest]
#[case::dfg("mmd_dfg", sample_hugr().0)]
#[case::cfg("mmd_cfg", crate::builder::test::simple_cfg_hugr())]
#[case::empty_dfg("mmd_empty_dfg", crate::builder::test::simple_dfg_hugr())]
#[case::func("mmd_func", crate::builder::test::simple_funcdef_hugr())]
#[case::module("mmd_module", crate::builder::test::simple_module_hugr())]
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn mermaid_string(#[case] test_name: &str, #[case] h: Hugr) {
    insta::assert_snapshot!(test_name, h.mermaid_string());
}

#[rstest]
fn all_ports(sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>)) {
    use itertools::Itertools;
    let (h, n1, n2) = sample_hugr;

    let all_output_ports = h.all_linked_outputs(n2.node()).collect_vec();
    let all_ports = h.all_node_ports(n2.node()).collect_vec();

    assert_eq!(
        &all_output_ports[..],
        &[
            (n1.node(), 1.into()),
            (n1.node(), 0.into()),
            (n1.node(), 2.into()),
        ]
    );
    assert!(
        all_output_ports
            .iter()
            .all(|&(_, p)| all_ports.contains(&p.into()))
    );

    let all_linked_inputs = h.all_linked_inputs(n1.node()).collect_vec();
    assert_eq!(
        &all_linked_inputs[..],
        &[
            (n2.node(), 1.into()),
            (n2.node(), 0.into()),
            (n2.node(), 2.into()),
        ]
    );
}

#[test]
fn value_types() {
    use crate::builder::Container;
    use crate::extension::prelude::bool_t;

    use crate::utils::test_quantum_extension::h_gate;
    use itertools::Itertools;

    let mut dfg =
        DFGBuilder::new(inout_sig(vec![qb_t(), bool_t()], vec![bool_t(), qb_t()])).unwrap();

    let [q, b] = dfg.input_wires_arr();
    let n1 = dfg.add_dataflow_op(h_gate(), [q]).unwrap();
    let n2 = dfg.add_dataflow_op(LogicOp::Not, [b]).unwrap();
    dfg.add_other_wire(n1.node(), n2.node());
    let h = dfg
        .finish_hugr_with_outputs([n2.out_wire(0), n1.out_wire(0)])
        .unwrap();

    let [_, o] = h.get_io(h.entrypoint()).unwrap();
    let n1_out_types = h.out_value_types(n1.node()).collect_vec();

    assert_eq!(&n1_out_types[..], &[(0.into(), qb_t())]);
    let out_types = h.in_value_types(o).collect_vec();

    assert_eq!(&out_types[..], &[(0.into(), bool_t()), (1.into(), qb_t())]);
}

#[test]
fn static_targets() {
    use crate::extension::prelude::{ConstUsize, usize_t};
    use itertools::Itertools;
    let mut dfg = DFGBuilder::new(inout_sig(type_row![], vec![usize_t()])).unwrap();

    let c = dfg.add_constant(Value::extension(ConstUsize::new(1)));

    let load = dfg.load_const(&c);

    let h = dfg.finish_hugr_with_outputs([load]).unwrap();

    assert_eq!(h.static_source(load.node()), Some(c.node()));

    assert_eq!(
        &h.static_targets(c.node()).unwrap().collect_vec()[..],
        &[(load.node(), 0.into())]
    );
}

#[test]
fn test_dataflow_ports_only() {
    use crate::builder::DataflowSubContainer;
    use crate::extension::prelude::bool_t;
    use crate::hugr::views::PortIterator;

    use itertools::Itertools;

    let mut dfg = DFGBuilder::new(endo_sig(bool_t())).unwrap();
    let local_and = {
        let mut mb = dfg.module_root_builder();
        let local_and = mb
            .define_function("and", Signature::new(vec![bool_t(); 2], bool_t()))
            .unwrap();
        let first_input = local_and.input().out_wire(0);
        local_and.finish_with_outputs([first_input]).unwrap()
    };
    let [in_bool] = dfg.input_wires_arr();

    let not = dfg.add_dataflow_op(LogicOp::Not, [in_bool]).unwrap();
    let call = dfg
        .call(local_and.handle(), &[], [not.out_wire(0); 2])
        .unwrap();
    dfg.add_other_wire(not.node(), call.node());
    let h = dfg.finish_hugr_with_outputs(not.outputs()).unwrap();
    let filtered_ports = h
        .all_linked_outputs(call.node())
        .dataflow_ports_only(&h)
        .collect_vec();

    // should ignore the static input in to call, but report the two value ports
    // and the order port.
    assert_eq!(
        &filtered_ports[..],
        &[
            (not.node(), 0.into()),
            (not.node(), 0.into()),
            (not.node(), 1.into())
        ]
    );
}
