use crate::const_fold2::HugrValueContext;

use ascent::{lattice::BoundedLattice, Lattice};
use hugr_core::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowSubContainer, HugrBuilder, SubContainer},
    extension::{
        prelude::{UnpackTuple, BOOL_T},
        ExtensionSet, EMPTY_REG,
    },
    ops::{handle::NodeHandle, DataflowOpTrait, Value},
    type_row,
    types::{Signature, SumType, Type},
    HugrView,
};

use super::{AbstractValue, Machine, PartialValue, TailLoopTermination};

#[test]
fn test_make_tuple() {
    let mut builder = DFGBuilder::new(endo_sig(vec![])).unwrap();
    let v1 = builder.add_load_value(Value::false_val());
    let v2 = builder.add_load_value(Value::true_val());
    let v3 = builder.make_tuple([v1, v2]).unwrap();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::default();
    machine.run(HugrValueContext::new(&hugr));

    let x = machine
        .read_out_wire(v3)
        .unwrap()
        .try_into_wire_value(&hugr, v3)
        .unwrap();
    assert_eq!(x, Value::tuple([Value::false_val(), Value::true_val()]));
}

#[test]
fn test_unpack_tuple_const() {
    let mut builder = DFGBuilder::new(endo_sig(vec![])).unwrap();
    let v = builder.add_load_value(Value::tuple([Value::false_val(), Value::true_val()]));
    let [o1, o2] = builder
        .add_dataflow_op(UnpackTuple::new(type_row![BOOL_T, BOOL_T]), [v])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::default();
    machine.run(HugrValueContext::new(&hugr));

    let o1_r = machine
        .read_out_wire(o1)
        .unwrap()
        .try_into_wire_value(&hugr, o1)
        .unwrap();
    assert_eq!(o1_r, Value::false_val());
    let o2_r = machine
        .read_out_wire(o2)
        .unwrap()
        .try_into_wire_value(&hugr, o2)
        .unwrap();
    assert_eq!(o2_r, Value::true_val());
}

#[test]
fn test_tail_loop_never_iterates() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
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

    let mut machine = Machine::default();
    machine.run(HugrValueContext::new(&hugr));

    let o_r = machine
        .read_out_wire(tl_o)
        .unwrap()
        .try_into_wire_value(&hugr, tl_o)
        .unwrap();
    assert_eq!(o_r, r_v);
    assert_eq!(
        Some(TailLoopTermination::NeverContinues),
        machine.tail_loop_terminates(&hugr, tail_loop.node())
    )
}

#[test]
fn test_tail_loop_always_iterates() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let r_w = builder
        .add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
    let true_w = builder.add_load_value(Value::true_val());

    let tlb = builder
        .tail_loop_builder([], [(BOOL_T, true_w)], vec![BOOL_T].into())
        .unwrap();

    // r_w has tag 0, so we always continue;
    // we put true in our "other_output", but we should not propagate this to
    // output because r_w never supports 1.
    let tail_loop = tlb.finish_with_outputs(r_w, [true_w]).unwrap();

    let [tl_o1, tl_o2] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::default();
    machine.run(HugrValueContext::new(&hugr));

    let o_r1 = machine.read_out_wire(tl_o1).unwrap();
    assert_eq!(o_r1, PartialValue::bottom());
    let o_r2 = machine.read_out_wire(tl_o2).unwrap();
    assert_eq!(o_r2, PartialValue::bottom());
    assert_eq!(
        Some(TailLoopTermination::NeverBreaks),
        machine.tail_loop_terminates(&hugr, tail_loop.node())
    );
    assert_eq!(machine.tail_loop_terminates(&hugr, hugr.root()), None);
}

#[test]
fn test_tail_loop_iterates_twice() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    // let var_type = Type::new_sum([type_row![BOOL_T,BOOL_T], type_row![BOOL_T,BOOL_T]]);

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    // let r_w = builder
    //     .add_load_value(Value::sum(0, [], SumType::new([type_row![], BOOL_T.into()])).unwrap());
    let tlb = builder
        .tail_loop_builder_exts(
            [],
            [(BOOL_T, false_w), (BOOL_T, true_w)],
            vec![].into(),
            ExtensionSet::new(),
        )
        .unwrap();
    assert_eq!(
        tlb.loop_signature().unwrap().signature(),
        Signature::new_endo(type_row![BOOL_T, BOOL_T])
    );
    let [in_w1, in_w2] = tlb.input_wires_arr();
    let tail_loop = tlb.finish_with_outputs(in_w1, [in_w2, in_w1]).unwrap();

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();
    // TODO once we can do conditionals put these wires inside `just_outputs` and
    // we should be able to propagate their values...ALAN wtf? loop control type IS bool ATM
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let mut machine = Machine::default();
    machine.run(HugrValueContext::new(&hugr));

    let true_or_false = PartialValue::new_variant(0, []).join(PartialValue::new_variant(1, []));
    // TODO these should be the propagated values for now they will be join(true,false) - ALAN wtf?
    let o_r1 = machine.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, true_or_false);
    let o_r2 = machine.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, true_or_false);
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        machine.tail_loop_terminates(&hugr, tail_loop.node())
    );
    assert_eq!(machine.tail_loop_terminates(&hugr, hugr.root()), None);
}

#[test]
fn conditional() {
    let variants = vec![type_row![], type_row![], type_row![BOOL_T]];
    let cond_t = Type::new_sum(variants.clone());
    let mut builder = DFGBuilder::new(Signature::new(cond_t, type_row![])).unwrap();
    let [arg_w] = builder.input_wires_arr();

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    let mut cond_builder = builder
        .conditional_builder(
            (variants, arg_w),
            [(BOOL_T, true_w)],
            type_row!(BOOL_T, BOOL_T),
        )
        .unwrap();
    // will be unreachable
    let case1_b = cond_builder.case_builder(0).unwrap();
    let case1 = case1_b.finish_with_outputs([false_w, false_w]).unwrap();

    let case2_b = cond_builder.case_builder(1).unwrap();
    let [c2a] = case2_b.input_wires_arr();
    let case2 = case2_b.finish_with_outputs([false_w, c2a]).unwrap();

    let case3_b = cond_builder.case_builder(2).unwrap();
    let [c3_1, _c3_2] = case3_b.input_wires_arr();
    let case3 = case3_b.finish_with_outputs([c3_1, false_w]).unwrap();

    let cond = cond_builder.finish_sub_container().unwrap();

    let [cond_o1, cond_o2] = cond.outputs_arr();

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::default();
    let arg_pv = PartialValue::new_variant(1, []).join(PartialValue::new_variant(
        2,
        [PartialValue::new_variant(0, [])],
    ));
    machine.propolutate_out_wires([(arg_w, arg_pv)]);
    machine.run(HugrValueContext::new(&hugr));

    let cond_r1 = machine
        .read_out_wire(cond_o1)
        .unwrap()
        .try_into_wire_value(&hugr, cond_o1)
        .unwrap();
    assert_eq!(cond_r1, Value::false_val());
    assert!(machine
        .read_out_wire(cond_o2)
        .unwrap()
        .try_into_wire_value(&hugr, cond_o2)
        .is_err());

    assert_eq!(machine.case_reachable(&hugr, case1.node()), Some(false)); // arg_pv is variant 1 or 2 only
    assert_eq!(machine.case_reachable(&hugr, case2.node()), Some(true));
    assert_eq!(machine.case_reachable(&hugr, case3.node()), Some(true));
    assert_eq!(machine.case_reachable(&hugr, cond.node()), None);
}
