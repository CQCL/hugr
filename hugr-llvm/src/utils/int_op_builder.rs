use hugr_core::{
    Wire,
    builder::{BuildError, Dataflow},
    extension::simple_op::HasConcrete as _,
    std_extensions::arithmetic::int_ops::IntOpDef,
    types::TypeArg,
};

pub trait IntOpBuilder: Dataflow {
    fn add_iadd(
        &mut self,
        log_width: impl Into<TypeArg>,
        x1: Wire,
        x2: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = IntOpDef::iadd.instantiate(&[log_width.into()]).unwrap();
        Ok(self.add_dataflow_op(op, [x1, x2])?.out_wire(0))
    }

    fn add_ieq(
        &mut self,
        log_width: impl Into<TypeArg>,
        x1: Wire,
        x2: Wire,
    ) -> Result<Wire, BuildError> {
        // TODO Add an OpLoadError variant to BuildError
        let op = IntOpDef::ieq.instantiate(&[log_width.into()]).unwrap();
        Ok(self.add_dataflow_op(op, [x1, x2])?.out_wire(0))
    }
}

impl<D: Dataflow> IntOpBuilder for D {}
