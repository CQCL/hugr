use hugr_core::{
    builder::{Container, DFGBuilder, Dataflow, HugrBuilder, SubContainer},
    extension::{prelude::BOOL_T, EMPTY_REG},
    ops::{handle::NodeHandle, OpTrait, UnpackTuple, Value},
    type_row,
    types::{FunctionType, SumType},
    HugrView, OutgoingPort, Wire,
};

use hugr_core::partial_value::PartialValue;
use itertools::Itertools;

use super::*;

#[test]
fn test_make_tuple() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let v1 = builder.add_load_value(Value::false_val());
    let v2 = builder.add_load_value(Value::true_val());
    let v3 = builder.make_tuple([v1, v2]).unwrap();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);

    let x = machine.read_out_wire_value(&c, v3).unwrap();
    assert_eq!(x, Value::tuple([Value::false_val(), Value::true_val()]));
}

#[test]
fn test_unpack_tuple() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let v1 = builder.add_load_value(Value::false_val());
    let v2 = builder.add_load_value(Value::true_val());
    let v3 = builder.make_tuple([v1, v2]).unwrap();
    let [o1, o2] = builder
        .add_dataflow_op(UnpackTuple::new(type_row![BOOL_T, BOOL_T]), [v3])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);

    let o1_r = machine.read_out_wire_value(&c, o1).unwrap();
    assert_eq!(o1_r, Value::false_val());
    let o2_r = machine.read_out_wire_value(&c, o2).unwrap();
    assert_eq!(o2_r, Value::true_val());
}

#[test]
fn test_unpack_const() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let v1 = builder.add_load_value(Value::tuple([Value::true_val()]));
    let [o] = builder
        .add_dataflow_op(UnpackTuple::new(type_row![BOOL_T]), [v1])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);

    let o_r = machine.read_out_wire_value(&c, o).unwrap();
    assert_eq!(o_r, Value::true_val());
}

#[test]
fn test_tail_loop_never_iterates() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let r_v = Value::unit_sum(3, 6).unwrap();
    let r_w = builder.add_load_value(
        Value::sum(
            1,
            [r_v.clone()],
            SumType::new([type_row![], r_v.get_type().into()]),
        )
        .unwrap(),
    );
    let tlb = builder
        .tail_loop_builder([], [], vec![r_v.get_type()].into())
        .unwrap();
    let tail_loop = tlb.finish_with_outputs(r_w, []).unwrap();
    let [tl_o] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);
    // dbg!(&machine.tail_loop_io_node);
    // dbg!(&machine.out_wire_value);

    let o_r = machine.read_out_wire_value(&c, tl_o).unwrap();
    assert_eq!(o_r, r_v);
    assert_eq!(
        TailLoopTermination::SingleIteration,
        machine.tail_loop_terminates(&c, tail_loop.node())
    )
}

#[test]
fn test_tail_loop_always_iterates() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let r_w = builder
        .add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
    let tlb = builder
        .tail_loop_builder([], [], vec![BOOL_T].into())
        .unwrap();
    let tail_loop = tlb.finish_with_outputs(r_w, []).unwrap();
    let [tl_o] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);
    // dbg!(&machine.tail_loop_io_node);
    // dbg!(&machine.out_wire_value);

    let o_r = machine.read_out_wire_partial_value(&c, tl_o).unwrap();
    assert_eq!(o_r, PartialValue::Bottom);
    assert_eq!(
        TailLoopTermination::NeverTerminates,
        machine.tail_loop_terminates(&c, tail_loop.node())
    )
}

#[test]
fn test_tail_loop_iterates_twice() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    // let var_type = Type::new_sum([type_row![BOOL_T,BOOL_T], type_row![BOOL_T,BOOL_T]]);

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    // let r_w = builder
    //     .add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
    let tlb = builder
        .tail_loop_builder([], [(BOOL_T, false_w), (BOOL_T, true_w)], vec![].into())
        .unwrap();
    assert_eq!(
        tlb.loop_signature().unwrap().dataflow_signature().unwrap(),
        FunctionType::new_endo(type_row![BOOL_T, BOOL_T])
    );
    let [in_w1, in_w2] = tlb.input_wires_arr();
    let tail_loop = tlb.finish_with_outputs(in_w1, [in_w2, in_w1]).unwrap();

    // let optype = builder.hugr().get_optype(tail_loop.node());
    // for p in builder.hugr().node_outputs(tail_loop.node()) {
    //     use hugr_core::ops::OpType;
    //     println!("{:?}, {:?}", p, optype.port_kind(p));

    // }

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();
    // TODO once we can do conditionals put these wires inside `just_outputs` and
    // we should be able to propagate their values
    // let [o_w1, o_w2, _] = tail_loop.outputs_arr();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);
    // dbg!(&machine.tail_loop_io_node);
    // dbg!(&machine.out_wire_value);

    // TODO these hould be the propagated values
    // let o_r1 = machine.read_out_wire_value(&c, o_w1).unwrap();
    // assert_eq!(o_r1, Value::false_val());
    // let o_r2 = machine.read_out_wire_value(&c, o_w2).unwrap();
    // assert_eq!(o_r2, Value::true_val());
    assert_eq!(
        TailLoopTermination::Terminates,
        machine.tail_loop_terminates(&c, tail_loop.node())
    )
}
