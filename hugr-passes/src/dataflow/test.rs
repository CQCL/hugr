use std::convert::Infallible;

use ascent::{Lattice, lattice::BoundedLattice};

use hugr_core::builder::{CFGBuilder, DataflowHugr, ModuleBuilder, inout_sig};
use hugr_core::ops::{CallIndirect, TailLoop};
use hugr_core::types::{ConstTypeError, TypeRow};
use hugr_core::{Hugr, Node, Wire};
use hugr_core::{
    HugrView,
    builder::{DFGBuilder, Dataflow, DataflowSubContainer, HugrBuilder, SubContainer, endo_sig},
    extension::prelude::{UnpackTuple, bool_t},
    ops::{DataflowOpTrait, Tag, Value, handle::NodeHandle},
    type_row,
    types::{Signature, SumType, Type},
};
use rstest::{fixture, rstest};

use super::{
    AbstractValue, AsConcrete, ConstLoader, DFContext, LoadedFunction, Machine, PartialValue, Sum,
    TailLoopTermination,
};

// ------- Minimal implementation of DFContext and AbstractValue -------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Void {}

impl AbstractValue for Void {}

struct TestContext;

impl ConstLoader<Void> for TestContext {
    type Node = Node;
}
impl DFContext<Void> for TestContext {}

// This allows testing creation of tuple/sum Values (only)
impl<N> AsConcrete<Void, N> for Value {
    type ValErr = Infallible;

    type SumErr = ConstTypeError;

    fn from_value(v: Void) -> Result<Self, Infallible> {
        match v {}
    }

    fn from_sum(value: Sum<Self>) -> Result<Self, Self::SumErr> {
        Self::sum(value.tag, value.values, value.st)
    }

    fn from_func(func: LoadedFunction<N>) -> Result<Self, crate::dataflow::LoadedFunction<N>> {
        Err(func)
    }
}

fn pv_false() -> PartialValue<Void> {
    PartialValue::new_variant(0, [])
}

fn pv_true() -> PartialValue<Void> {
    PartialValue::new_variant(1, [])
}

fn pv_true_or_false() -> PartialValue<Void> {
    pv_true().join(pv_false())
}

#[test]
fn test_make_tuple() {
    let mut builder = DFGBuilder::new(endo_sig(vec![])).unwrap();
    let v1 = builder.add_load_value(Value::false_val());
    let v2 = builder.add_load_value(Value::true_val());
    let v3 = builder.make_tuple([v1, v2]).unwrap();
    let hugr = builder.finish_hugr().unwrap();

    let results = Machine::new(&hugr).run(TestContext, []);

    let x: Value = results.try_read_wire_concrete(v3).unwrap();
    assert_eq!(x, Value::tuple([Value::false_val(), Value::true_val()]));
}

#[test]
fn test_unpack_tuple_const() {
    let mut builder = DFGBuilder::new(endo_sig(vec![])).unwrap();
    let v = builder.add_load_value(Value::tuple([Value::false_val(), Value::true_val()]));
    let [o1, o2] = builder
        .add_dataflow_op(UnpackTuple::new(vec![bool_t(); 2].into()), [v])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr().unwrap();

    let results = Machine::new(&hugr).run(TestContext, []);

    let o1_r: Value = results.try_read_wire_concrete(o1).unwrap();
    assert_eq!(o1_r, Value::false_val());
    let o2_r: Value = results.try_read_wire_concrete(o2).unwrap();
    assert_eq!(o2_r, Value::true_val());
}

#[test]
fn test_tail_loop_never_iterates() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let r_v = Value::unit_sum(3, 6).unwrap();
    let r_w = builder.add_load_value(r_v.clone());
    let tag = Tag::new(
        TailLoop::BREAK_TAG,
        vec![type_row![], r_v.get_type().into()],
    );
    let tagged = builder.add_dataflow_op(tag, [r_w]).unwrap();

    let tlb = builder
        .tail_loop_builder([], [], vec![r_v.get_type()].into())
        .unwrap();
    let tail_loop = tlb.finish_with_outputs(tagged.out_wire(0), []).unwrap();
    let [tl_o] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr().unwrap();

    let results = Machine::new(&hugr).run(TestContext, []);

    let o_r: Value = results.try_read_wire_concrete(tl_o).unwrap();
    assert_eq!(o_r, r_v);
    assert_eq!(
        Some(TailLoopTermination::NeverContinues),
        results.tail_loop_terminates(tail_loop.node())
    );
}

#[test]
fn test_tail_loop_always_iterates() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let r_w = builder.add_load_value(
        Value::sum(
            TailLoop::CONTINUE_TAG,
            [],
            SumType::new([type_row![], bool_t().into()]),
        )
        .unwrap(),
    );
    let true_w = builder.add_load_value(Value::true_val());

    let tlb = builder
        .tail_loop_builder([], [(bool_t(), true_w)], vec![bool_t()].into())
        .unwrap();

    // r_w has tag 0, so we always continue;
    // we put true in our "other_output", but we should not propagate this to
    // output because r_w never supports 1.
    let tail_loop = tlb.finish_with_outputs(r_w, [true_w]).unwrap();

    let [tl_o1, tl_o2] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr().unwrap();

    let results = Machine::new(&hugr).run(TestContext, []);

    let o_r1 = results.read_out_wire(tl_o1).unwrap();
    assert_eq!(o_r1, PartialValue::bottom());
    let o_r2 = results.read_out_wire(tl_o2).unwrap();
    assert_eq!(o_r2, PartialValue::bottom());
    assert_eq!(
        Some(TailLoopTermination::NeverBreaks),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.entrypoint()), None);
}

#[test]
fn test_tail_loop_two_iters() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    let tlb = builder
        .tail_loop_builder([], [(bool_t(), false_w), (bool_t(), true_w)], type_row![])
        .unwrap();
    assert_eq!(
        tlb.loop_signature().unwrap().signature().as_ref(),
        &Signature::new_endo(vec![bool_t(); 2])
    );
    let [in_w1, in_w2] = tlb.input_wires_arr();
    let tail_loop = tlb.finish_with_outputs(in_w1, [in_w2, in_w1]).unwrap();

    let hugr = builder.finish_hugr().unwrap();
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let results = Machine::new(&hugr).run(TestContext, []);

    let o_r1 = results.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true_or_false());
    let o_r2 = results.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_true_or_false());
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.entrypoint()), None);
}

#[test]
fn test_tail_loop_containing_conditional() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let control_variants = vec![vec![bool_t(); 2].into(); 2];
    let control_t = Type::new_sum(control_variants.clone());
    let body_out_variants = vec![TypeRow::from(control_t.clone()), vec![bool_t(); 2].into()];

    let init = builder.add_load_value(
        Value::sum(
            0,
            [Value::false_val(), Value::true_val()],
            SumType::new(control_variants.clone()),
        )
        .unwrap(),
    );

    let mut tlb = builder
        .tail_loop_builder([(control_t, init)], [], vec![bool_t(); 2].into())
        .unwrap();
    let tl = tlb.loop_signature().unwrap().clone();
    let [in_w] = tlb.input_wires_arr();

    // Branch on in_wire, so first iter 0(false, true)...
    let mut cond = tlb
        .conditional_builder(
            (control_variants.clone(), in_w),
            [],
            Type::new_sum(body_out_variants.clone()).into(),
        )
        .unwrap();
    let mut case0_b = cond.case_builder(0).unwrap();
    let [a, b] = case0_b.input_wires_arr();
    // Builds value for next iter as 1(true, false) by flipping arguments
    let [next_input] = case0_b
        .add_dataflow_op(Tag::new(1, control_variants), [b, a])
        .unwrap()
        .outputs_arr();
    let cont = case0_b.make_continue(tl.clone(), [next_input]).unwrap();
    case0_b.finish_with_outputs([cont]).unwrap();
    // Second iter 1(true, false) => exit with (true, false)
    let mut case1_b = cond.case_builder(1).unwrap();
    let loop_res = case1_b.make_break(tl, case1_b.input_wires()).unwrap();
    case1_b.finish_with_outputs([loop_res]).unwrap();
    let [r] = cond.finish_sub_container().unwrap().outputs_arr();

    let tail_loop = tlb.finish_with_outputs(r, []).unwrap();

    let hugr = builder.finish_hugr().unwrap();
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let results = Machine::new(&hugr).run(TestContext, []);

    let o_r1 = results.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true());
    let o_r2 = results.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_false());
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.entrypoint()), None);
}

#[test]
fn test_conditional() {
    let variants = vec![type_row![], type_row![], bool_t().into()];
    let cond_t = Type::new_sum(variants.clone());
    let mut builder = DFGBuilder::new(Signature::new(cond_t, type_row![])).unwrap();
    let [arg_w] = builder.input_wires_arr();

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    let mut cond_builder = builder
        .conditional_builder(
            (variants, arg_w),
            [(bool_t(), true_w)],
            vec![bool_t(); 2].into(),
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

    let hugr = builder.finish_hugr().unwrap();

    let arg_pv = PartialValue::new_variant(1, []).join(PartialValue::new_variant(
        2,
        [PartialValue::new_variant(0, [])],
    ));
    let results = Machine::new(&hugr).run(TestContext, [(0.into(), arg_pv)]);

    let cond_r1: Value = results.try_read_wire_concrete(cond_o1).unwrap();
    assert_eq!(cond_r1, Value::false_val());
    assert!(results.try_read_wire_concrete::<Value>(cond_o2).is_err());

    assert_eq!(results.case_reachable(case1.node()), Some(false)); // arg_pv is variant 1 or 2 only
    assert_eq!(results.case_reachable(case2.node()), Some(true));
    assert_eq!(results.case_reachable(case3.node()), Some(true));
    assert_eq!(results.case_reachable(cond.node()), None);
}

// A Hugr being a function on bools: (x, y) => (x XOR y, x AND y)
#[fixture]
fn xor_and_cfg() -> Hugr {
    //        Entry       branch on first arg, passes arguments on unchanged
    //       /T   F\
    //      A --T-> B     A(x=true, y) branch on second arg, passing (first arg == true, false)
    //       \F    /      B(w,v) => X(v,w)
    //        > X <
    // Inputs received:
    // Entry    A       B       X
    // F,F      -       F,F     F,F
    // F,T      -       F,T     T,F
    // T,F      T,F     -       T,F
    // T,T      T,T     T,F     F,T
    let mut builder =
        CFGBuilder::new(Signature::new(vec![bool_t(); 2], vec![bool_t(); 2])).unwrap();

    // entry (x, y) => (if x then A else B)(x=true, y)
    let entry = builder
        .entry_builder(vec![type_row![]; 2], vec![bool_t(); 2].into())
        .unwrap();
    let [in_x, in_y] = entry.input_wires_arr();
    let entry = entry.finish_with_outputs(in_x, [in_x, in_y]).unwrap();

    // A(x==true, y) => (if y then B else X)(x, false)
    let mut a = builder
        .block_builder(
            vec![bool_t(); 2].into(),
            vec![type_row![]; 2],
            vec![bool_t(); 2].into(),
        )
        .unwrap();
    let [in_x, in_y] = a.input_wires_arr();
    let false_w1 = a.add_load_value(Value::false_val());
    let a = a.finish_with_outputs(in_y, [in_x, false_w1]).unwrap();

    // B(w, v) => X(v, w)
    let mut b = builder
        .block_builder(
            vec![bool_t(); 2].into(),
            [type_row![]],
            vec![bool_t(); 2].into(),
        )
        .unwrap();
    let [in_w, in_v] = b.input_wires_arr();
    let [control] = b
        .add_dataflow_op(Tag::new(0, vec![type_row![]]), [])
        .unwrap()
        .outputs_arr();
    let b = b.finish_with_outputs(control, [in_v, in_w]).unwrap();

    let x = builder.exit_block();

    let [fals, tru]: [usize; 2] = [0, 1];
    builder.branch(&entry, tru, &a).unwrap(); // if true
    builder.branch(&entry, fals, &b).unwrap(); // if false
    builder.branch(&a, tru, &b).unwrap(); // if true
    builder.branch(&a, fals, &x).unwrap(); // if false
    builder.branch(&b, 0, &x).unwrap();
    builder.finish_hugr().unwrap()
}

#[rstest]
#[case(pv_true(), pv_true(), pv_false(), pv_true())]
#[case(pv_true(), pv_false(), pv_true(), pv_false())]
#[case(pv_true(), pv_true_or_false(), pv_true_or_false(), pv_true_or_false())]
#[case(pv_true(), PartialValue::Top, pv_true_or_false(), pv_true_or_false())]
#[case(pv_false(), pv_true(), pv_true(), pv_false())]
#[case(pv_false(), pv_false(), pv_false(), pv_false())]
#[case(pv_false(), pv_true_or_false(), pv_true_or_false(), pv_false())]
#[case(pv_false(), PartialValue::Top, PartialValue::Top, pv_false())] // if !inp0 then out0=inp1
#[case(pv_true_or_false(), pv_true(), pv_true_or_false(), pv_true_or_false())]
#[case(pv_true_or_false(), pv_false(), pv_true_or_false(), pv_true_or_false())]
#[case(PartialValue::Top, pv_true(), pv_true_or_false(), PartialValue::Top)]
#[case(PartialValue::Top, pv_false(), PartialValue::Top, PartialValue::Top)]
fn test_cfg(
    #[case] inp0: PartialValue<Void>,
    #[case] inp1: PartialValue<Void>,
    #[case] out0: PartialValue<Void>,
    #[case] out1: PartialValue<Void>,
    xor_and_cfg: Hugr,
) {
    let root = xor_and_cfg.entrypoint();
    let results = Machine::new(&xor_and_cfg).run(TestContext, [(0.into(), inp0), (1.into(), inp1)]);

    assert_eq!(results.read_out_wire(Wire::new(root, 0)).unwrap(), out0);
    assert_eq!(results.read_out_wire(Wire::new(root, 1)).unwrap(), out1);
}

#[rstest]
#[case(pv_true(), pv_true(), pv_true())]
#[case(pv_false(), pv_false(), pv_false())]
#[case(pv_true(), pv_false(), pv_true_or_false())] // Two calls alias
fn test_call(
    #[case] inp0: PartialValue<Void>,
    #[case] inp1: PartialValue<Void>,
    #[case] out: PartialValue<Void>,
) {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![bool_t(); 2])).unwrap();
    let func_defn = {
        let mut mb = builder.module_root_builder();
        let func_bldr = mb
            .define_function("id", Signature::new_endo(bool_t()))
            .unwrap();
        let [v] = func_bldr.input_wires_arr();
        func_bldr.finish_with_outputs([v]).unwrap()
    };
    let [a, b] = builder.input_wires_arr();
    let [a2] = builder
        .call(func_defn.handle(), &[], [a])
        .unwrap()
        .outputs_arr();
    let [b2] = builder
        .call(func_defn.handle(), &[], [b])
        .unwrap()
        .outputs_arr();
    let hugr = builder.finish_hugr_with_outputs([a2, b2]).unwrap();

    let results = Machine::new(&hugr).run(TestContext, [(0.into(), inp0), (1.into(), inp1)]);

    let [res0, res1] = [0, 1].map(|i| {
        results
            .read_out_wire(Wire::new(hugr.entrypoint(), i))
            .unwrap()
    });
    // The two calls alias so both results will be the same:
    assert_eq!(res0, out);
    assert_eq!(res1, out);
}

#[test]
fn test_region() {
    let mut builder = DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t(); 2])).unwrap();
    let [in_w] = builder.input_wires_arr();
    let cst_w = builder.add_load_const(Value::false_val());
    // Create a nested DFG which gets in_w passed as an input, but has a nonlocal edge
    // from the LoadConstant
    let nested = builder
        .dfg_builder(Signature::new(bool_t(), vec![bool_t(); 2]), [in_w])
        .unwrap();
    let [nested_in] = nested.input_wires_arr();
    let nested = nested.finish_with_outputs([nested_in, cst_w]).unwrap();
    let hugr = builder.finish_hugr_with_outputs(nested.outputs()).unwrap();
    let [nested_input, _] = hugr.get_io(nested.node()).unwrap();
    let whole_hugr_results = Machine::new(&hugr).run(TestContext, [(0.into(), pv_true())]);
    let sub_hugr_results =
        Machine::new(hugr.with_entrypoint(nested.node())).run(TestContext, [(0.into(), pv_true())]);
    for (wire, val) in [
        (Wire::new(nested_input, 0), Some(pv_true())),
        (Wire::new(nested.node(), 0), Some(pv_true())),
        (Wire::new(nested.node(), 1), Some(pv_false())),
    ] {
        assert_eq!(whole_hugr_results.read_out_wire(wire), val);
        assert_eq!(sub_hugr_results.read_out_wire(wire), val);
    }

    for (wire, val) in [
        (cst_w, pv_false()),
        (Wire::new(hugr.entrypoint(), 0), pv_true()),
        (Wire::new(hugr.entrypoint(), 1), pv_false()),
    ] {
        assert_eq!(whole_hugr_results.read_out_wire(wire), Some(val));
        assert_eq!(sub_hugr_results.read_out_wire(wire), None);
    }
}

#[test]
fn test_module() {
    let mut modb = ModuleBuilder::new();
    let leaf_fn = modb
        .define_function("leaf", Signature::new_endo(vec![bool_t(); 2]))
        .unwrap();
    let outs = leaf_fn.input_wires();
    let leaf_fn = leaf_fn.finish_with_outputs(outs).unwrap();

    let mut f2 = modb
        .define_function("f2", Signature::new(bool_t(), vec![bool_t(); 2]))
        .unwrap();
    let [inp] = f2.input_wires_arr();
    let cst_true = f2.add_load_value(Value::true_val());
    let f2_call = f2.call(leaf_fn.handle(), &[], [inp, cst_true]).unwrap();
    let f2 = f2.finish_with_outputs(f2_call.outputs()).unwrap();

    let mut main = modb
        .define_function("main", Signature::new(bool_t(), vec![bool_t(); 2]))
        .unwrap();
    let [inp] = main.input_wires_arr();
    let cst_false = main.add_load_value(Value::false_val());
    let main_call = main.call(leaf_fn.handle(), &[], [inp, cst_false]).unwrap();
    let main = main.finish_with_outputs(main_call.outputs()).unwrap();
    let hugr = modb.finish_hugr().unwrap();
    let [f2_inp, _] = hugr.get_io(f2.node()).unwrap();

    let results_just_main = {
        let mut mach = Machine::new(&hugr);
        mach.prepopulate_inputs(main.node(), [(0.into(), pv_true())])
            .unwrap();
        mach.run(TestContext, [])
    };
    assert_eq!(
        results_just_main.read_out_wire(Wire::new(f2_inp, 0)),
        Some(PartialValue::Bottom)
    );
    for call in [f2_call, main_call] {
        // The first output of the Call comes from `main` because no value was fed in from f2
        assert_eq!(
            results_just_main.read_out_wire(Wire::new(call.node(), 0)),
            Some(pv_true())
        );
        // (Without reachability) the second output of the Call is the join of the two constant inputs from the two calls
        assert_eq!(
            results_just_main.read_out_wire(Wire::new(call.node(), 1)),
            Some(pv_true_or_false())
        );
    }

    let results_two_calls = {
        let mut m = Machine::new(&hugr);
        m.prepopulate_inputs(f2.node(), [(0.into(), pv_true())])
            .unwrap();
        m.prepopulate_inputs(main.node(), [(0.into(), pv_false())])
            .unwrap();
        m.run(TestContext, [])
    };

    for call in [f2_call, main_call] {
        assert_eq!(
            results_two_calls.read_out_wire(Wire::new(call.node(), 0)),
            Some(pv_true_or_false())
        );
        assert_eq!(
            results_two_calls.read_out_wire(Wire::new(call.node(), 1)),
            Some(pv_true_or_false())
        );
    }
}

#[rstest]
#[case(pv_false(), pv_false())]
#[case(pv_false(), pv_true())]
#[case(pv_true(), pv_false())]
#[case(pv_true(), pv_true())]
fn call_indirect(#[case] inp1: PartialValue<Void>, #[case] inp2: PartialValue<Void>) {
    let b2b = || Signature::new_endo(bool_t());
    let mut dfb = DFGBuilder::new(inout_sig(vec![bool_t(); 3], vec![bool_t(); 2])).unwrap();

    let [id1, id2] = ["id1", "[id2]"].map(|name| {
        let mut mb = dfb.module_root_builder();
        let fb = mb.define_function(name, b2b()).unwrap();
        let [inp] = fb.input_wires_arr();
        fb.finish_with_outputs([inp]).unwrap()
    });

    let [inp_direct, which, inp_indirect] = dfb.input_wires_arr();
    let [res1] = dfb
        .call(id1.handle(), &[], [inp_direct])
        .unwrap()
        .outputs_arr();

    // We'll unconditionally load both functions, to demonstrate that it's
    // the CallIndirect that matters, not just which functions are loaded.
    let lf1 = dfb.load_func(id1.handle(), &[]).unwrap();
    let lf2 = dfb.load_func(id2.handle(), &[]).unwrap();
    let bool_func = || Type::new_function(b2b());
    let mut cond = dfb
        .conditional_builder(
            (vec![type_row![]; 2], which),
            [(bool_func(), lf1), (bool_func(), lf2)],
            bool_func().into(),
        )
        .unwrap();
    let case_false = cond.case_builder(0).unwrap();
    let [f0, _f1] = case_false.input_wires_arr();
    case_false.finish_with_outputs([f0]).unwrap();
    let case_true = cond.case_builder(1).unwrap();
    let [_f0, f1] = case_true.input_wires_arr();
    case_true.finish_with_outputs([f1]).unwrap();
    let [tgt] = cond.finish_sub_container().unwrap().outputs_arr();
    let [res2] = dfb
        .add_dataflow_op(CallIndirect { signature: b2b() }, [tgt, inp_indirect])
        .unwrap()
        .outputs_arr();
    let h = dfb.finish_hugr_with_outputs([res1, res2]).unwrap();

    let run = |which| {
        Machine::new(&h).run(
            TestContext,
            [
                (0.into(), inp1.clone()),
                (1.into(), which),
                (2.into(), inp2.clone()),
            ],
        )
    };
    let (w1, w2) = (Wire::new(h.entrypoint(), 0), Wire::new(h.entrypoint(), 1));

    // 1. Test with `which` unknown -> second output unknown
    let results = run(PartialValue::Top);
    assert_eq!(results.read_out_wire(w1), Some(inp1.clone()));
    assert_eq!(results.read_out_wire(w2), Some(PartialValue::Top));

    // 2. Test with `which` selecting second function -> both passthrough
    let results = run(pv_true());
    assert_eq!(results.read_out_wire(w1), Some(inp1.clone()));
    assert_eq!(results.read_out_wire(w2), Some(inp2.clone()));

    //3. Test with `which` selecting first function -> alias
    let results = run(pv_false());
    let out = Some(inp1.join(inp2));
    assert_eq!(results.read_out_wire(w1), out);
    assert_eq!(results.read_out_wire(w2), out);
}
