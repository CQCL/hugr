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

#[rstest]
fn dot_string(sample_hugr: (Hugr, BuildHandle<DataflowOpID>, BuildHandle<DataflowOpID>)) {
    let (h, _, _) = sample_hugr;

    insta::assert_yaml_snapshot!(h.dot_string());
}
