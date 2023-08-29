use portgraph::PortOffset;

use crate::{
    builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr},
    extension::{prelude::QB_T, prelude_registry},
    ops::handle::NodeHandle,
    std_extensions::quantum::test::cx_gate,
    type_row,
    types::FunctionType,
    HugrView,
};

#[test]
fn node_connections() -> Result<(), BuildError> {
    let mut dfg = DFGBuilder::new(FunctionType::new(
        type_row![QB_T, QB_T],
        type_row![QB_T, QB_T],
    ))?;

    let [q1, q2] = dfg.input_wires_arr();

    let n1 = dfg.add_dataflow_op(cx_gate(), [q1, q2])?;
    let [q1, q2] = n1.outputs_arr();
    let n2 = dfg.add_dataflow_op(cx_gate(), [q2, q1])?;

    let h = dfg.finish_hugr_with_outputs(n2.outputs(), &prelude_registry())?;

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
        ]
    );
    Ok(())
}
