use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ascent::{lattice::BoundedLattice, Lattice};

use hugr_core::builder::CFGBuilder;
use hugr_core::types::TypeRow;
use hugr_core::Wire;
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
use rstest::rstest;

use super::{AbstractValue, BaseValue, DFContext, Machine, PartialValue, TailLoopTermination};

// ------- Minimal implementation of DFContext and BaseValue -------
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Void {}

impl BaseValue for Void {}

struct TestContext<H>(Arc<H>);

// Deriving Clone requires H:HugrView to implement Clone,
// but we don't need that as we only clone the Arc.
impl<H: HugrView> Clone for TestContext<H> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<H: HugrView> std::ops::Deref for TestContext<H> {
    type Target = hugr_core::Hugr;

    fn deref(&self) -> &Self::Target {
        self.0.base_hugr()
    }
}

// Any value used in an Ascent program must be hashable.
// However, there should only be one DFContext, so its hash is immaterial.
impl<H: HugrView> Hash for TestContext<H> {
    fn hash<I: Hasher>(&self, _state: &mut I) {}
}

impl<H: HugrView> PartialEq for TestContext<H> {
    fn eq(&self, other: &Self) -> bool {
        // Any AscentProgram should have only one DFContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        true
    }
}

impl<H: HugrView> Eq for TestContext<H> {}

impl<H: HugrView> PartialOrd for TestContext<H> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Any AscentProgram should have only one DFContext (maybe cloned)
        assert!(Arc::ptr_eq(&self.0, &other.0));
        Some(std::cmp::Ordering::Equal)
    }
}

impl<H: HugrView> DFContext<PartialValue<Void>> for TestContext<H> {
    fn interpret_leaf_op(
        &self,
        _node: hugr_core::Node,
        _ins: &[PartialValue<Void>],
    ) -> Option<Vec<PartialValue<Void>>> {
        None
    }
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

    let mut machine = Machine::default();
    machine.propolutate_out_wires([(v1, pv_false()), (v2, pv_true())]);
    machine.run(TestContext(Arc::new(&hugr)));

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
    machine.propolutate_out_wires([(v, PartialValue::new_variant(0, [pv_false(), pv_true()]))]);
    machine.run(TestContext(Arc::new(&hugr)));

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
    let r_w = builder.add_load_value(r_v.clone());
    let tag = Tag::new(1, vec![type_row![], r_v.get_type().into()]);
    let tagged = builder.add_dataflow_op(tag, [r_w]).unwrap();

    let tlb = builder
        .tail_loop_builder([], [], vec![r_v.get_type()].into())
        .unwrap();
    let tail_loop = tlb.finish_with_outputs(tagged.out_wire(0), []).unwrap();
    let [tl_o] = tail_loop.outputs_arr();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let mut machine = Machine::default();
    machine.propolutate_out_wires([(r_w, PartialValue::new_variant(3, []))]);
    machine.run(TestContext(Arc::new(&hugr)));

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
    machine.run(TestContext(Arc::new(&hugr)));

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

    let mut machine = Machine::default();
    machine.propolutate_out_wires([(true_w, pv_true()), (false_w, pv_false())]);
    machine.run(TestContext(Arc::new(&hugr)));

    let o_r1 = machine.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true_or_false());
    let o_r2 = machine.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_true_or_false());
    assert_eq!(
        Some(TailLoopTermination::BreaksAndContinues),
        machine.tail_loop_terminates(&hugr, tail_loop.node())
    );
    assert_eq!(machine.tail_loop_terminates(&hugr, hugr.root()), None);
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
    let cont = case0_b
        .add_dataflow_op(Tag::new(0, body_out_variants.clone()), [next_input])
        .unwrap();
    case0_b.finish_with_outputs(cont.outputs()).unwrap();
    // Second iter 1(true, false) => exit with (true, false)
    let mut case1_b = cond.case_builder(1).unwrap();
    let loop_res = case1_b
        .add_dataflow_op(Tag::new(1, body_out_variants), case1_b.input_wires())
        .unwrap();
    case1_b.finish_with_outputs(loop_res.outputs()).unwrap();
    let [r] = cond.finish_sub_container().unwrap().outputs_arr();

    let tail_loop = tlb.finish_with_outputs(r, []).unwrap();

    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();
    let [o_w1, o_w2] = tail_loop.outputs_arr();

    let mut machine = Machine::default();
    machine.propolutate_out_wires([(init, PartialValue::new_variant(0, [pv_false(), pv_true()]))]);
    machine.run(TestContext(Arc::new(&hugr)));

    let o_r1 = machine.read_out_wire(o_w1).unwrap();
    assert_eq!(o_r1, pv_true());
    let o_r2 = machine.read_out_wire(o_w2).unwrap();
    assert_eq!(o_r2, pv_false());
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
    machine.run(TestContext(Arc::new(&hugr)));

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

#[rstest]
#[case(pv_true(), pv_true(), pv_true())] // OK
#[case(pv_true(), pv_false(), pv_false())] // OK
#[case(pv_false(), pv_true(), pv_false())] // OK
#[case(pv_false(), pv_false(), pv_true())] // OK
#[case(PartialValue::top(), pv_true(), PartialValue::top())] // Result should be true_or_false? TOP means all inputs inside cases are TOP
#[case(PartialValue::top(), pv_false(), pv_true_or_false())] // OK
fn cfg(
    #[case] inp0: PartialValue<Void>,
    #[case] inp1: PartialValue<Void>,
    #[case] outp: PartialValue<Void>,
) {
    //        Entry
    //       /0   1\
    //      A --1-> B
    //       \0    /
    //        > X <
    let mut builder = CFGBuilder::new(Signature::new(type_row![BOOL_T;2], BOOL_T)).unwrap();

    // entry (i, j) => if i {B(j)} else {A(j, i, true)}, note that (j, i, true) == (j, false, true)
    let entry_outs = [type_row![BOOL_T;3], type_row![BOOL_T]];
    let mut entry = builder
        .entry_builder(entry_outs.clone(), type_row![])
        .unwrap();
    let [in_i, in_j] = entry.input_wires_arr();
    let mut cond = entry
        .conditional_builder(
            (vec![type_row![]; 2], in_i),
            [],
            Type::new_sum(entry_outs.clone()).into(),
        )
        .unwrap();
    let mut if_i_true = cond.case_builder(1).unwrap();
    let br_to_b = if_i_true
        .add_dataflow_op(Tag::new(1, entry_outs.to_vec()), [in_j])
        .unwrap();
    if_i_true.finish_with_outputs(br_to_b.outputs()).unwrap();
    let mut if_i_false = cond.case_builder(0).unwrap();
    let true_w = if_i_false.add_load_value(Value::true_val());
    let br_to_a = if_i_false
        .add_dataflow_op(Tag::new(0, entry_outs.into()), [in_j, in_i, true_w])
        .unwrap();
    if_i_false.finish_with_outputs(br_to_a.outputs()).unwrap();

    let [res] = cond.finish_sub_container().unwrap().outputs_arr();
    let entry = entry.finish_with_outputs(res, []).unwrap();

    // A(w, y, z) => if w {B(y)} else {X(z)}
    let a_outs = vec![type_row![BOOL_T]; 2];
    let mut a = builder
        .block_builder(
            type_row![BOOL_T; 3],
            vec![type_row![BOOL_T]; 2],
            type_row![],
        )
        .unwrap();
    let [in_w, in_y, in_z] = a.input_wires_arr();
    let mut cond = a
        .conditional_builder(
            (vec![type_row![]; 2], in_w),
            [],
            Type::new_sum(a_outs.clone()).into(),
        )
        .unwrap();
    let mut if_w_true = cond.case_builder(1).unwrap();
    let br_to_b = if_w_true
        .add_dataflow_op(Tag::new(1, a_outs.clone()), [in_y])
        .unwrap();
    if_w_true.finish_with_outputs(br_to_b.outputs()).unwrap();
    let mut if_w_false = cond.case_builder(0).unwrap();
    let br_to_x = if_w_false
        .add_dataflow_op(Tag::new(0, a_outs), [in_z])
        .unwrap();
    if_w_false.finish_with_outputs(br_to_x.outputs()).unwrap();
    let [res] = cond.finish_sub_container().unwrap().outputs_arr();
    let a = a.finish_with_outputs(res, []).unwrap();

    // B(v) => X(v)
    let mut b = builder
        .block_builder(type_row![BOOL_T], [type_row![BOOL_T]], type_row![])
        .unwrap();
    let [control] = b
        .add_dataflow_op(Tag::new(0, vec![type_row![BOOL_T]]), b.input_wires())
        .unwrap()
        .outputs_arr();
    let b = b.finish_with_outputs(control, []).unwrap();

    let x = builder.exit_block();

    builder.branch(&entry, 0, &a).unwrap();
    builder.branch(&entry, 1, &b).unwrap();
    builder.branch(&a, 0, &x).unwrap();
    builder.branch(&a, 1, &b).unwrap();
    builder.branch(&b, 0, &x).unwrap();
    let hugr = builder.finish_hugr(&EMPTY_REG).unwrap();

    let [entry_input, _] = hugr.get_io(entry.node()).unwrap();
    let [in_w0, in_w1] = [0, 1].map(|i| Wire::new(entry_input, i));

    let mut machine = Machine::default();
    machine.propolutate_out_wires([(in_w0, inp0), (in_w1, inp1), (true_w, pv_true())]);
    machine.run(TestContext(Arc::new(&hugr)));

    assert_eq!(
        machine.read_out_wire(Wire::new(hugr.root(), 0)).unwrap(),
        outp
    );
}
