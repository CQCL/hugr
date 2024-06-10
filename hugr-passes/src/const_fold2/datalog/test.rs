use hugr_core::{
    builder::{DFGBuilder, Dataflow, DataflowSubContainer, HugrBuilder, SubContainer}, extension::{prelude::BOOL_T, ExtensionSet, EMPTY_REG}, ops::{handle::NodeHandle, OpTrait, UnpackTuple, Value}, partial_value::PartialSum, type_row, types::{FunctionType, SumType}, Extension
};

use hugr_core::partial_value::PartialValue;

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
        TailLoopTermination::ExactlyZeroContinues,
        machine.tail_loop_terminates(&c, tail_loop.node())
    )
}

#[test]
fn test_tail_loop_always_iterates() {
    let mut builder = DFGBuilder::new(FunctionType::new_endo(&[])).unwrap();
    let r_w = builder
        .add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
    let true_w = builder.add_load_value(Value::true_val());

    let tlb = builder
        .tail_loop_builder([], [(BOOL_T,true_w)], vec![BOOL_T].into())
        .unwrap();

    // r_w has tag 0, so we always continue;
    // we put true in our "other_output", but we should not propagate this to
    // output because r_w never supports 1.
    let tail_loop = tlb.finish_with_outputs(r_w, [true_w]).unwrap();

    let [tl_o1, tl_o2] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);

    let o_r1 = machine.read_out_wire_partial_value(&c, tl_o1).unwrap();
    assert_eq!(o_r1, PartialValue::bottom());
    let o_r2 = machine.read_out_wire_partial_value(&c, tl_o2).unwrap();
    assert_eq!(o_r2, PartialValue::bottom());
    assert_eq!(
        TailLoopTermination::bottom(),
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
    let [o_w1, o_w2, _] = tail_loop.outputs_arr();

    let mut machine = Machine::new();
    let c = machine.run_hugr(&hugr);
    // dbg!(&machine.tail_loop_io_node);
    // dbg!(&machine.out_wire_value);

    // TODO these hould be the propagated values for now they will bt join(true,false)
    let o_r1 = machine.read_out_wire_partial_value(&c, o_w1).unwrap();
    // assert_eq!(o_r1, PartialValue::top());
    let o_r2 = machine.read_out_wire_partial_value(&c, o_w2).unwrap();
    // assert_eq!(o_r2, Value::true_val());
    assert_eq!(
        TailLoopTermination::Top,
        machine.tail_loop_terminates(&c, tail_loop.node())
    )
}

#[test]
fn conditional() {
    let variants = vec![type_row![], type_row![], type_row![BOOL_T]];
    let cond_t = Type::new_sum(variants.clone());
    let mut builder = DFGBuilder::new(FunctionType::new(Into::<TypeRow>::into(cond_t),type_row![])).unwrap();
    let [arg_w] = builder.input_wires_arr();

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    let mut cond_builder = builder.conditional_builder((variants, arg_w), [(BOOL_T,true_w)], type_row!(BOOL_T,BOOL_T), ExtensionSet::default()).unwrap();
    // will be unreachable
    let case1_b = cond_builder.case_builder(0).unwrap();
    let case1 = case1_b.finish_with_outputs([false_w,false_w]).unwrap();

    let case2_b = cond_builder.case_builder(1).unwrap();
    let [c2a] = case2_b.input_wires_arr();
    let case2 = case2_b.finish_with_outputs([false_w,c2a]).unwrap();

    let case3_b = cond_builder.case_builder(2).unwrap();
    let [c3_1,c3_2] = case3_b.input_wires_arr();
    let case3 = case3_b.finish_with_outputs([c3_1,false_w]).unwrap();

    let cond = cond_builder.finish_sub_container().unwrap();

    let [cond_o1,cond_o2] = cond.outputs_arr();

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::new();
    let arg_pv = PartialValue::variant(1, []).join(PartialValue::variant(2,[PartialValue::variant(0,[])]));
    machine.propolutate_out_wires([(arg_w, arg_pv)]);
    let c = machine.run_hugr(&hugr);

    let cond_r1 = machine.read_out_wire_value(&c, cond_o1).unwrap();
    assert_eq!(cond_r1, Value::false_val());
    assert!(machine.read_out_wire_value(&c, cond_o2).is_none());

    assert!(!machine.case_reachable(&c, case1.node()));
    assert!(machine.case_reachable(&c, case2.node()));
    assert!(machine.case_reachable(&c, case3.node()));
}
