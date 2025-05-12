use hugr_core::{
    Wire,
    builder::{BuildError, Dataflow},
    std_extensions::logic::LogicOp,
};

pub trait LogicOpBuilder: Dataflow {
    fn add_and(&mut self, x1: Wire, x2: Wire) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(LogicOp::And, [x1, x2])?.out_wire(0))
    }

    fn add_not(&mut self, x1: Wire) -> Result<Wire, BuildError> {
        Ok(self.add_dataflow_op(LogicOp::Not, [x1])?.out_wire(0))
    }
}

impl<D: Dataflow> LogicOpBuilder for D {}
