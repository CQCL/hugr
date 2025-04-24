//! Callbacks for use with [ReplaceTypes::replace_consts_parametrized]
//! and [DelegatingLinearizer::register_callback](super::DelegatingLinearizer::register_callback)

use hugr_core::builder::{endo_sig, inout_sig, DFGBuilder, Dataflow, DataflowHugr};
use hugr_core::extension::prelude::{option_type, UnwrapBuilder};
use hugr_core::extension::ExtensionSet;
use hugr_core::ops::constant::CustomConst;
use hugr_core::ops::{constant::OpaqueValue, Value};
use hugr_core::ops::{OpTrait, OpType, Tag};
use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
use hugr_core::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
use hugr_core::std_extensions::collections::array::{Array, ArrayKind, GenericArrayValue};
use hugr_core::std_extensions::collections::list::ListValue;
use hugr_core::std_extensions::collections::value_array::{
    value_array_type, VArrayOpDef, VArrayRepeat, VArrayScan, ValueArray,
};
use hugr_core::types::{SumType, Transformable, Type, TypeArg};
use hugr_core::{type_row, Hugr, HugrView};
use itertools::Itertools;

use super::{
    CallbackHandler, LinearizeError, Linearizer, NodeTemplate, ReplaceTypes, ReplaceTypesError,
};

/// Handler for [ListValue] constants that updates the element type and
/// recursively [ReplaceTypes::change_value]s the elements of the list.
/// Included in [ReplaceTypes::default].
pub fn list_const(
    val: &OpaqueValue,
    repl: &ReplaceTypes,
) -> Result<Option<Value>, ReplaceTypesError> {
    let Some(lv) = val.value().downcast_ref::<ListValue>() else {
        return Ok(None);
    };
    let mut elem_t = lv.get_element_type().clone();
    if !elem_t.transform(repl)? {
        // No change to type, so values should not change either
        return Ok(None);
    }

    let mut vals: Vec<Value> = lv.get_contents().to_vec();
    for v in vals.iter_mut() {
        repl.change_value(v)?;
    }
    Ok(Some(ListValue::new(elem_t, vals).into()))
}

/// Handler for [GenericArrayValue] constants that recursively
/// [ReplaceTypes::change_value]s the elements of the list.
/// Included in [ReplaceTypes::default].
pub fn generic_array_const<AK: ArrayKind>(
    val: &OpaqueValue,
    repl: &ReplaceTypes,
) -> Result<Option<Value>, ReplaceTypesError>
where
    GenericArrayValue<AK>: CustomConst,
{
    let Some(av) = val.value().downcast_ref::<GenericArrayValue<AK>>() else {
        return Ok(None);
    };
    let mut elem_t = av.get_element_type().clone();
    if !elem_t.transform(repl)? {
        // No change to type, so values should not change either
        return Ok(None);
    }

    let mut vals: Vec<Value> = av.get_contents().to_vec();
    for v in vals.iter_mut() {
        repl.change_value(v)?;
    }
    Ok(Some(GenericArrayValue::<AK>::new(elem_t, vals).into()))
}

/// Handler for [ArrayValue] constants that recursively
/// [ReplaceTypes::change_value]s the elements of the list.
/// Included in [ReplaceTypes::default].
///
/// [ArrayValue]: hugr_core::std_extensions::collections::array::ArrayValue
pub fn array_const(
    val: &OpaqueValue,
    repl: &ReplaceTypes,
) -> Result<Option<Value>, ReplaceTypesError> {
    generic_array_const::<Array>(val, repl)
}

/// Handler for [VArrayValue] constants that recursively
/// [ReplaceTypes::change_value]s the elements of the list.
/// Included in [ReplaceTypes::default].
///
/// [VArrayValue]: hugr_core::std_extensions::collections::value_array::VArrayValue
pub fn value_array_const(
    val: &OpaqueValue,
    repl: &ReplaceTypes,
) -> Result<Option<Value>, ReplaceTypesError> {
    generic_array_const::<ValueArray>(val, repl)
}

fn runtime_reqs(h: &Hugr) -> ExtensionSet {
    h.signature(h.root()).unwrap().runtime_reqs.clone()
}

/// Handler for copying/discarding value arrays if their elements have become linear.
/// Included in [ReplaceTypes::default] and [DelegatingLinearizer::default].
///
/// [DelegatingLinearizer::default]: super::DelegatingLinearizer::default
pub fn linearize_value_array(
    args: &[TypeArg],
    num_outports: usize,
    lin: &CallbackHandler,
) -> Result<NodeTemplate, LinearizeError> {
    // Require known length i.e. usable only after monomorphization, due to no-variables limitation
    // restriction on NodeTemplate::CompoundOp
    let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = args else {
        panic!("Illegal TypeArgs to array: {:?}", args)
    };
    if num_outports == 0 {
        // "Simple" discard - first map each element to unit (via type-specific discard):
        let map_fn = {
            let mut dfb = DFGBuilder::new(inout_sig(ty.clone(), Type::UNIT)).unwrap();
            let [to_discard] = dfb.input_wires_arr();
            lin.copy_discard_op(ty, 0)?
                .add(&mut dfb, [to_discard])
                .map_err(|e| LinearizeError::NestedTemplateError(ty.clone(), e))?;
            let ret = dfb.add_load_value(Value::unary_unit_sum());
            dfb.finish_hugr_with_outputs([ret]).unwrap()
        };
        // Now array.scan that over the input array to get an array of unit (which can be discarded)
        let array_scan = VArrayScan::new(ty.clone(), Type::UNIT, vec![], *n, runtime_reqs(&map_fn));
        let in_type = value_array_type(*n, ty.clone());
        return Ok(NodeTemplate::CompoundOp(Box::new({
            let mut dfb = DFGBuilder::new(inout_sig(in_type, type_row![])).unwrap();
            let [in_array] = dfb.input_wires_arr();
            let map_fn = dfb.add_load_value(Value::Function {
                hugr: Box::new(map_fn),
            });
            // scan has one output, an array of unit, so just ignore/discard that
            dfb.add_dataflow_op(array_scan, [in_array, map_fn]).unwrap();
            dfb.finish_hugr_with_outputs([]).unwrap()
        })));
    };
    // The num_outports>1 case will simplify, and unify with the previous, when we have a
    // more general ArrayScan https://github.com/CQCL/hugr/issues/2041. In the meantime:
    let num_new = num_outports - 1;
    let array_ty = value_array_type(*n, ty.clone());
    let mut dfb = DFGBuilder::new(inout_sig(
        array_ty.clone(),
        vec![array_ty.clone(); num_outports],
    ))
    .unwrap();

    // 1. make num_new array<SZ, Option<T>>, initialized to None...
    let option_sty = option_type(ty.clone());
    let option_ty = Type::from(option_sty.clone());
    let arrays_of_none = {
        let fn_none = {
            let mut dfb = DFGBuilder::new(inout_sig(vec![], option_ty.clone())).unwrap();
            let none = dfb
                .add_dataflow_op(Tag::new(0, vec![type_row![], ty.clone().into()]), [])
                .unwrap();
            dfb.finish_hugr_with_outputs(none.outputs()).unwrap()
        };
        let repeats =
            vec![VArrayRepeat::new(option_ty.clone(), *n, runtime_reqs(&fn_none)); num_new];
        let fn_none = dfb.add_load_value(Value::function(fn_none).unwrap());
        repeats
            .into_iter()
            .map(|rpt| {
                let [arr] = dfb.add_dataflow_op(rpt, [fn_none]).unwrap().outputs_arr();
                arr
            })
            .collect::<Vec<_>>()
    };

    // 2. use a scan through the input array, copying the element num_outputs times;
    // return the first copy, and put each of the other copies into one of the array<option>
    let i64_t = INT_TYPES[6].to_owned();
    let option_array = value_array_type(*n, option_ty.clone());
    let copy_elem = {
        let mut io = vec![ty.clone(), i64_t.clone()];
        io.extend(vec![option_array.clone(); num_new]);
        let mut dfb = DFGBuilder::new(endo_sig(io)).unwrap();
        let mut inputs = dfb.input_wires();
        let elem = inputs.next().unwrap();
        let idx = inputs.next().unwrap();
        let opt_arrays = inputs.collect::<Vec<_>>();
        let [idx_usz] = dfb
            .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [idx])
            .unwrap()
            .outputs_arr();
        let mut copies = lin
            .copy_discard_op(ty, num_outports)?
            .add(&mut dfb, [elem])
            .map_err(|e| LinearizeError::NestedTemplateError(ty.clone(), e))?
            .outputs();
        let copy0 = copies.next().unwrap(); // We'll return this directly

        // Wrap each remaining copy into an option
        let set_op = OpType::from(VArrayOpDef::set.to_concrete(option_ty.clone(), *n));
        let either_st = set_op.dataflow_signature().unwrap().output[0]
            .as_sum()
            .unwrap()
            .clone();
        let opt_arrays = opt_arrays
            .into_iter()
            .zip_eq(copies)
            .map(|(opt_array, copy1)| {
                let [tag] = dfb
                    .add_dataflow_op(Tag::new(1, vec![type_row![], ty.clone().into()]), [copy1])
                    .unwrap()
                    .outputs_arr();
                let [set_result] = dfb
                    .add_dataflow_op(set_op.clone(), [opt_array, idx_usz, tag])
                    .unwrap()
                    .outputs_arr();
                // set should always be successful
                let [none, opt_array] = dfb
                    .build_unwrap_sum(1, either_st.clone(), set_result)
                    .unwrap();
                //the removed element is an option, which should always be none (and thus discardable)
                let [] = dfb
                    .build_unwrap_sum(0, SumType::new_option(ty.clone()), none)
                    .unwrap();
                opt_array
            })
            .collect::<Vec<_>>(); // stop borrowing dfb

        let cst1 = dfb.add_load_value(ConstInt::new_u(6, 1).unwrap());
        let [new_idx] = dfb
            .add_dataflow_op(IntOpDef::iadd.with_log_width(6), [idx, cst1])
            .unwrap()
            .outputs_arr();
        dfb.finish_hugr_with_outputs([copy0, new_idx].into_iter().chain(opt_arrays))
            .unwrap()
    };
    let [in_array] = dfb.input_wires_arr();
    let scan1 = VArrayScan::new(
        ty.clone(),
        ty.clone(),
        std::iter::once(i64_t)
            .chain(vec![option_array; num_new])
            .collect(),
        *n,
        runtime_reqs(&copy_elem),
    );

    let copy_elem = dfb.add_load_value(Value::function(copy_elem).unwrap());
    let cst0 = dfb.add_load_value(ConstInt::new_u(6, 0).unwrap());

    let mut outs = dfb
        .add_dataflow_op(
            scan1,
            [in_array, copy_elem, cst0]
                .into_iter()
                .chain(arrays_of_none),
        )
        .unwrap()
        .outputs();
    let out_array1 = outs.next().unwrap();
    let _idx_out = outs.next().unwrap();
    let opt_arrays = outs;

    //3. Scan each array-of-options, 'unwrapping' each element into a non-option
    let unwrap_elem = {
        let mut dfb =
            DFGBuilder::new(inout_sig(Type::from(option_ty.clone()), ty.clone())).unwrap();
        let [opt] = dfb.input_wires_arr();
        let [val] = dfb.build_unwrap_sum(1, option_sty.clone(), opt).unwrap();
        dfb.finish_hugr_with_outputs([val]).unwrap()
    };

    let unwrap_scan = VArrayScan::new(
        option_ty.clone(),
        ty.clone(),
        vec![],
        *n,
        runtime_reqs(&unwrap_elem),
    );
    let unwrap_elem = dfb.add_load_value(Value::function(unwrap_elem).unwrap());

    let out_arrays = std::iter::once(out_array1)
        .chain(opt_arrays.map(|opt_array| {
            let [out_array] = dfb
                .add_dataflow_op(unwrap_scan.clone(), [opt_array, unwrap_elem])
                .unwrap()
                .outputs_arr();
            out_array
        }))
        .collect::<Vec<_>>();

    Ok(NodeTemplate::CompoundOp(Box::new(
        dfb.finish_hugr_with_outputs(out_arrays).unwrap(),
    )))
}
