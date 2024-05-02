use portgraph::PortOffset;
use rstest::{fixture, rstest};

use crate::{
    builder::{BuildError, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::QB_T,
    ops::{
        handle::{DataflowOpID, NodeHandle},
        Value,
    },
    type_row,
    types::FunctionType,
    utils::test_quantum_extension::cx_gate,
    Hugr, HugrView,
};

#[fixture]
fn sample_hugr() -> (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>) {
    let mut dfg = DFGBuilder::new(FunctionType::new(
        type_row![QB_T, QB_T],
        type_row![QB_T, QB_T],
    ))
    .unwrap();

    let [q1, q2] = dfg.input_wires_arr();

    let n1 = dfg.add_dataflow_op(cx_gate(), [q1, q2]).unwrap();
    let [q1, q2] = n1.outputs_arr();
    let n2 = dfg.add_dataflow_op(cx_gate(), [q2, q1]).unwrap();
    dfg.add_other_wire(n1.node(), n2.node());

    (
        dfg.finish_prelude_hugr_with_outputs(n2.outputs()).unwrap(),
        n1,
        n2,
    )
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
#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
fn mermaid_string(#[case] test_name: &str, #[case] h: Hugr) {
    insta::assert_snapshot!(test_name, h.mermaid_string());
}

#[rstest]
fn all_ports(sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>)) {
    use itertools::Itertools;
    let (h, n1, n2) = sample_hugr;

    let all_output_ports = h.all_linked_outputs(n2.node()).collect_vec();

    assert_eq!(
        &all_output_ports[..],
        &[
            (n1.node(), 1.into()),
            (n1.node(), 0.into()),
            (n1.node(), 2.into()),
        ]
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
    use crate::extension::prelude::BOOL_T;
    use crate::std_extensions::logic::NotOp;
    use crate::utils::test_quantum_extension::h_gate;
    use itertools::Itertools;

    let mut dfg = DFGBuilder::new(FunctionType::new(
        type_row![QB_T, BOOL_T],
        type_row![BOOL_T, QB_T],
    ))
    .unwrap();

    let [q, b] = dfg.input_wires_arr();
    let n1 = dfg.add_dataflow_op(h_gate(), [q]).unwrap();
    let n2 = dfg.add_dataflow_op(NotOp, [b]).unwrap();
    dfg.add_other_wire(n1.node(), n2.node());
    let h = dfg
        .finish_prelude_hugr_with_outputs([n2.out_wire(0), n1.out_wire(0)])
        .unwrap();

    let [_, o] = h.get_io(h.root()).unwrap();
    let n1_out_types = h.out_value_types(n1.node()).collect_vec();

    assert_eq!(&n1_out_types[..], &[(0.into(), QB_T)]);
    let out_types = h.in_value_types(o).collect_vec();

    assert_eq!(&out_types[..], &[(0.into(), BOOL_T), (1.into(), QB_T)]);
}

#[test]
fn static_targets() {
    use crate::extension::prelude::{ConstUsize, PRELUDE_ID, USIZE_T};
    use itertools::Itertools;
    let mut dfg = DFGBuilder::new(
        FunctionType::new(type_row![], type_row![USIZE_T]).with_extension_delta(PRELUDE_ID),
    )
    .unwrap();

    let c = dfg.add_constant(Value::extension(ConstUsize::new(1)));

    let load = dfg.load_const(&c);

    let h = dfg.finish_prelude_hugr_with_outputs([load]).unwrap();

    assert_eq!(h.static_source(load.node()), Some(c.node()));

    assert_eq!(
        &h.static_targets(c.node()).unwrap().collect_vec()[..],
        &[(load.node(), 0.into())]
    )
}

#[test]
fn test_dataflow_ports_only() {
    use crate::builder::DataflowSubContainer;
    use crate::extension::{prelude::BOOL_T, PRELUDE_REGISTRY};
    use crate::hugr::views::PortIterator;
    use crate::std_extensions::logic::NotOp;
    use itertools::Itertools;

    let mut dfg = DFGBuilder::new(FunctionType::new(type_row![BOOL_T], type_row![BOOL_T])).unwrap();
    let local_and = {
        let local_and = dfg
            .define_function(
                "and",
                FunctionType::new(type_row![BOOL_T; 2], type_row![BOOL_T]).into(),
            )
            .unwrap();
        let first_input = local_and.input().out_wire(0);
        local_and.finish_with_outputs([first_input]).unwrap()
    };
    let [in_bool] = dfg.input_wires_arr();

    let not = dfg.add_dataflow_op(NotOp, [in_bool]).unwrap();
    let call = dfg
        .call(
            local_and.handle(),
            &[],
            [not.out_wire(0); 2],
            &PRELUDE_REGISTRY,
        )
        .unwrap();
    dfg.add_other_wire(not.node(), call.node());
    let h = dfg
        .finish_hugr_with_outputs(not.outputs(), &PRELUDE_REGISTRY)
        .unwrap();
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
    )
}
