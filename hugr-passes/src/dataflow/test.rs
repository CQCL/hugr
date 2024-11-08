use ascent::{lattice::BoundedLattice, Lattice};

use hugr_core::builder::{CFGBuilder, Container, DataflowHugr};
use hugr_core::hugr::views::{DescendantsGraph, HierarchyView};
use hugr_core::ops::handle::DfgID;
use hugr_core::ops::TailLoop;
use hugr_core::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowSubContainer, HugrBuilder, SubContainer},
    extension::{
        prelude::{UnpackTuple, BOOL_T},
        ExtensionSet, EMPTY_REG,
    },
    ops::{handle::NodeHandle, DataflowOpTrait, Tag, Value},
    type_row,
    types::{Signature, SumType, Type},
    HugrView,
};
use hugr_core::{Hugr, Wire};
use rstest::{fixture, rstest};

use super::{AbstractValue, ConstLoader, DFContext, Machine, PartialValue, TailLoopTermination};

// ------- Minimal implementation of DFContext and AbstractValue -------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Void {}

impl AbstractValue for Void {}

struct TestContext<H>(H);

impl<H: HugrView> std::ops::Deref for TestContext<H> {
    type Target = H;
    fn deref(&self) -> &H {
        &self.0
    }
}
impl<H> ConstLoader<Void> for TestContext<H> {}
impl<H: HugrView> DFContext<Void> for TestContext<H> {
    type View = H;
}

// This allows testing creation of tuple/sum Values (only)
impl From<Void> for Value {
    fn from(v: Void) -> Self {
        match v {}
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
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let results = Machine::default().run(TestContext(hugr), []);

    let x: Value = results.try_read_wire_value(v3).unwrap();
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

    let results = Machine::default().run(TestContext(hugr), []);

    let o1_r: Value = results.try_read_wire_value(o1).unwrap();
    assert_eq!(o1_r, Value::false_val());
    let o2_r: Value = results.try_read_wire_value(o2).unwrap();
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
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let results = Machine::default().run(TestContext(hugr), []);

    let o_r: Value = results.try_read_wire_value(tl_o).unwrap();
    assert_eq!(o_r, r_v);
    assert_eq!(
        Some(TailLoopTermination::NeverContinues),
        results.tail_loop_terminates(tail_loop.node())
    )
}

#[test]
fn test_tail_loop_always_iterates() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let r_w = builder.add_load_value(
        Value::sum(
            TailLoop::CONTINUE_TAG,
            [],
            SumType::new([type_row![], BOOL_T.into()]),
        )
        .unwrap(),
    );
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

    let results = Machine::default().run(TestContext(&hugr), []);

    let o_r1 = results.read_out_wire(tl_o1).unwrap();
    assert_eq!(o_r1, PartialValue::bottom());
    let o_r2 = results.read_out_wire(tl_o2).unwrap();
    assert_eq!(o_r2, PartialValue::bottom());
    assert_eq!(
        Some(TailLoopTermination::NeverBreaks),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.root()), None);
}

#[test]
fn test_tail_loop_two_iters() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();

    let true_w = builder.add_load_value(Value::true_val());
    let false_w = builder.add_load_value(Value::false_val());

    let tlb = builder
        .tail_loop_builder_exts(
            [],
            [(BOOL_T, false_w), (BOOL_T, true_w)],
            type_row![],
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
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let results = Machine::default().run(TestContext(&hugr), []);

    let o_r1 = results.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true_or_false());
    let o_r2 = results.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_true_or_false());
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.root()), None);
}

#[test]
fn test_tail_loop_containing_conditional() {
    let mut builder = DFGBuilder::new(Signature::new_endo(vec![])).unwrap();
    let control_variants = vec![type_row![BOOL_T;2]; 2];
    let control_t = Type::new_sum(control_variants.clone());
    let body_out_variants = vec![control_t.clone().into(), type_row![BOOL_T; 2]];

    let init = builder.add_load_value(
        Value::sum(
            0,
            [Value::false_val(), Value::true_val()],
            SumType::new(control_variants.clone()),
        )
        .unwrap(),
    );

    let mut tlb = builder
        .tail_loop_builder([(control_t, init)], [], type_row![BOOL_T; 2])
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

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let results = Machine::default().run(TestContext(&hugr), []);

    let o_r1 = results.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true());
    let o_r2 = results.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_false());
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        results.tail_loop_terminates(tail_loop.node())
    );
    assert_eq!(results.tail_loop_terminates(hugr.root()), None);
}

#[test]
fn test_conditional() {
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

    let arg_pv = PartialValue::new_variant(1, []).join(PartialValue::new_variant(
        2,
        [PartialValue::new_variant(0, [])],
    ));
    let results = Machine::default().run(TestContext(hugr), [(0.into(), arg_pv)]);

    let cond_r1: Value = results.try_read_wire_value(cond_o1).unwrap();
    assert_eq!(cond_r1, Value::false_val());
    assert!(results.try_read_wire_value::<Value, _, _>(cond_o2).is_err());

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
        CFGBuilder::new(Signature::new(type_row![BOOL_T; 2], type_row![BOOL_T; 2])).unwrap();

    // entry (x, y) => (if x then A else B)(x=true, y)
    let entry = builder
        .entry_builder(vec![type_row![]; 2], type_row![BOOL_T;2])
        .unwrap();
    let [in_x, in_y] = entry.input_wires_arr();
    let entry = entry.finish_with_outputs(in_x, [in_x, in_y]).unwrap();

    // A(x==true, y) => (if y then B else X)(x, false)
    let mut a = builder
        .block_builder(
            type_row![BOOL_T; 2],
            vec![type_row![]; 2],
            type_row![BOOL_T; 2],
        )
        .unwrap();
    let [in_x, in_y] = a.input_wires_arr();
    let false_w1 = a.add_load_value(Value::false_val());
    let a = a.finish_with_outputs(in_y, [in_x, false_w1]).unwrap();

    // B(w, v) => X(v, w)
    let mut b = builder
        .block_builder(type_row![BOOL_T; 2], [type_row![]], type_row![BOOL_T; 2])
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
    builder.finish_hugr(&EMPTY_REG).unwrap()
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
    let root = xor_and_cfg.root();
    let results = Machine::default().run(
        TestContext(xor_and_cfg),
        [(0.into(), inp0), (1.into(), inp1)],
    );

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
    let mut builder = DFGBuilder::new(Signature::new_endo(type_row![BOOL_T; 2])).unwrap();
    let func_bldr = builder
        .define_function("id", Signature::new_endo(BOOL_T))
        .unwrap();
    let [v] = func_bldr.input_wires_arr();
    let func_defn = func_bldr.finish_with_outputs([v]).unwrap();
    let [a, b] = builder.input_wires_arr();
    let [a2] = builder
        .call(func_defn.handle(), &[], [a], &EMPTY_REG)
        .unwrap()
        .outputs_arr();
    let [b2] = builder
        .call(func_defn.handle(), &[], [b], &EMPTY_REG)
        .unwrap()
        .outputs_arr();
    let hugr = builder
        .finish_hugr_with_outputs([a2, b2], &EMPTY_REG)
        .unwrap();

    let results = Machine::default().run(TestContext(&hugr), [(0.into(), inp0), (1.into(), inp1)]);

    let [res0, res1] = [0, 1].map(|i| results.read_out_wire(Wire::new(hugr.root(), i)).unwrap());
    // The two calls alias so both results will be the same:
    assert_eq!(res0, out);
    assert_eq!(res1, out);
}

#[test]
fn test_region() {
    let mut builder =
        DFGBuilder::new(Signature::new(type_row![BOOL_T], type_row![BOOL_T;2])).unwrap();
    let [in_w] = builder.input_wires_arr();
    let cst_w = builder.add_load_const(Value::false_val());
    let nested = builder
        .dfg_builder(Signature::new_endo(type_row![BOOL_T; 2]), [in_w, cst_w])
        .unwrap();
    let nested_ins = nested.input_wires();
    let nested = nested.finish_with_outputs(nested_ins).unwrap();
    let hugr = builder
        .finish_prelude_hugr_with_outputs(nested.outputs())
        .unwrap();
    let [nested_input, _] = hugr.get_io(nested.node()).unwrap();
    let whole_hugr_results = Machine::default().run(TestContext(&hugr), [(0.into(), pv_true())]);
    assert_eq!(
        whole_hugr_results.read_out_wire(Wire::new(nested_input, 0)),
        Some(pv_true())
    );
    assert_eq!(
        whole_hugr_results.read_out_wire(Wire::new(nested_input, 1)),
        Some(pv_false())
    );
    assert_eq!(
        whole_hugr_results.read_out_wire(Wire::new(hugr.root(), 0)),
        Some(pv_true())
    );
    assert_eq!(
        whole_hugr_results.read_out_wire(Wire::new(hugr.root(), 1)),
        Some(pv_false())
    );

    let subview = DescendantsGraph::<DfgID>::try_new(&hugr, nested.node()).unwrap();
    // Do not provide a value on the second input (constant false in the whole hugr, above)
    let sub_hugr_results = Machine::default().run(TestContext(subview), [(0.into(), pv_true())]);
    assert_eq!(
        sub_hugr_results.read_out_wire(Wire::new(nested_input, 0)),
        Some(pv_true())
    );
    assert_eq!(
        sub_hugr_results.read_out_wire(Wire::new(nested_input, 1)),
        Some(PartialValue::Top)
    );
    for w in [0, 1] {
        assert_eq!(
            sub_hugr_results.read_out_wire(Wire::new(hugr.root(), w)),
            None
        );
    }
}
