use portgraph::PortOffset;
use rstest::{fixture, rstest};

use crate::{
    builder::{BuildError, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::QB_T,
    ops::handle::{DataflowOpID, NodeHandle},
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
    dfg.add_other_wire(n1.node(), n2.node()).unwrap();

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

#[cfg_attr(miri, ignore)] // Opening files is not supported in (isolated) miri
#[rstest]
fn dot_string(sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>)) {
    let (h, _, _) = sample_hugr;

    insta::assert_yaml_snapshot!(h.dot_string());
}

#[rustversion::since(1.75)] // uses impl in return position
#[rstest]
fn all_ports(sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>)) {
    use crate::hugr::Direction;
    use crate::Port;
    use itertools::Itertools;
    let (h, n1, n2) = sample_hugr;

    let all_output_ports = h.all_linked_outputs(n2.node()).collect_vec();

    assert_eq!(
        &all_output_ports[..],
        &[
            (
                n1.node(),
                Port::new(Direction::Outgoing, 1).as_outgoing().unwrap()
            ),
            (
                n1.node(),
                Port::new(Direction::Outgoing, 0).as_outgoing().unwrap()
            ),
            (
                n1.node(),
                Port::new(Direction::Outgoing, 2).as_outgoing().unwrap()
            ),
        ]
    );

    let all_linked_inputs = h.all_linked_inputs(n1.node()).collect_vec();

    assert_eq!(
        &all_linked_inputs[..],
        &[
            (
                n2.node(),
                Port::new(Direction::Incoming, 1).as_incoming().unwrap()
            ),
            (
                n2.node(),
                Port::new(Direction::Incoming, 0).as_incoming().unwrap()
            ),
            (
                n2.node(),
                Port::new(Direction::Incoming, 2).as_incoming().unwrap()
            ),
        ]
    );
}

#[rustversion::since(1.75)] // uses impl in return position
#[test]
fn value_types() {
    use crate::builder::Container;
    use crate::extension::prelude::BOOL_T;
    use crate::std_extensions::logic::test::not_op;
    use crate::utils::test_quantum_extension::h_gate;
    use itertools::Itertools;
    let mut dfg = DFGBuilder::new(FunctionType::new(
        type_row![QB_T, BOOL_T],
        type_row![BOOL_T, QB_T],
    ))
    .unwrap();

    let [q, b] = dfg.input_wires_arr();
    let n1 = dfg.add_dataflow_op(h_gate(), [q]).unwrap();
    let n2 = dfg.add_dataflow_op(not_op(), [b]).unwrap();
    dfg.add_other_wire(n1.node(), n2.node()).unwrap();
    let h = dfg
        .finish_prelude_hugr_with_outputs([n2.out_wire(0), n1.out_wire(0)])
        .unwrap();

    let [_, o] = h.get_io(h.root()).unwrap();
    let n1_out_types = h.out_value_types(n1.node()).collect_vec();

    assert_eq!(&n1_out_types[..], &[(0.into(), QB_T)]);
    let out_types = h.in_value_types(o).collect_vec();

    assert_eq!(&out_types[..], &[(0.into(), BOOL_T), (1.into(), QB_T)]);
}
