//! Callbacks for use with [ReplaceTypes::replace_consts_parametrized]
//! and [ReplaceTypes::linearize_parametric]

use hugr_core::builder::{endo_sig, inout_sig, DFGBuilder, Dataflow, DataflowHugr};
use hugr_core::extension::prelude::{option_type, UnwrapBuilder};
use hugr_core::extension::ExtensionSet;
use hugr_core::ops::{constant::OpaqueValue, Value};
use hugr_core::ops::{OpTrait, OpType, Tag};
use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
use hugr_core::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
use hugr_core::std_extensions::collections::array::{
    array_type, ArrayOpDef, ArrayRepeat, ArrayScan,
};
use hugr_core::std_extensions::collections::list::ListValue;
use hugr_core::types::{SumType, Transformable, Type, TypeArg};
use hugr_core::{type_row, Hugr, HugrView};
use itertools::Itertools;

use super::{LinearizeError, Linearizer, OpReplacement, ReplaceTypes, ReplaceTypesError};

/// Handler for [ListValue] constants that recursively [ReplaceTypes::change_value]s
/// the elements of the list
pub fn list_const(
    val: &OpaqueValue,
    repl: &ReplaceTypes,
) -> Result<Option<Value>, ReplaceTypesError> {
    let Some(lv) = val.value().downcast_ref::<ListValue>() else {
        return Ok(None);
    };
    let mut vals: Vec<Value> = lv.get_contents().to_vec();
    let mut ch = false;
    for v in vals.iter_mut() {
        ch |= repl.change_value(v)?;
    }
    // If none of the values has changed, assume the Type hasn't (Values have a single known type)
    if !ch {
        return Ok(None);
    };

    let mut elem_t = lv.get_element_type().clone();
    elem_t.transform(repl)?;
    Ok(Some(ListValue::new(elem_t, vals).into()))
}

fn runtime_reqs(h: &Hugr) -> ExtensionSet {
    h.signature(h.root()).unwrap().runtime_reqs.clone()
}

/// Handler for copying/discarding arrays, for use with [ReplaceTypes::linearize_parametric] of
/// [array_type_def](hugr_core::std_extensions::collections::array::array_type_def)
pub fn linearize_array(
    args: &[TypeArg],
    num_outports: usize,
    lin: &Linearizer,
) -> Result<OpReplacement, LinearizeError> {
    // Require known length i.e. usable only after monomorphization, due to no-variables limitation
    // restriction on OpReplacement::CompoundOp
    let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = args else {
        panic!("Illegal TypeArgs to array: {:?}", args)
    };
    if num_outports == 0 {
        // Make a function that maps the linear element to unit
        let map_fn = {
            let mut dfb = DFGBuilder::new(inout_sig(ty.clone(), Type::UNIT)).unwrap();
            let [to_discard] = dfb.input_wires_arr();
            lin.copy_discard_op(ty, 0)?
                .add(&mut dfb, [to_discard])
                .unwrap();
            let ret = dfb.add_load_value(Value::unary_unit_sum());
            dfb.finish_hugr_with_outputs([ret]).unwrap()
        };
        // Now array.scan that over the input array to get an array of unit (which can be discarded)
        let array_scan = ArrayScan::new(ty.clone(), Type::UNIT, vec![], *n, runtime_reqs(&map_fn));
        let in_type = array_type(*n, ty.clone());
        Ok(OpReplacement::CompoundOp(Box::new({
            let mut dfb = DFGBuilder::new(inout_sig(in_type, type_row![])).unwrap();
            let [in_array] = dfb.input_wires_arr();
            let map_fn = dfb.add_load_value(Value::Function {
                hugr: Box::new(map_fn),
            });
            // scan has one output, an array of unit, so just ignore/discard that
            dfb.add_dataflow_op(array_scan, [in_array, map_fn]).unwrap();
            dfb.finish_hugr_with_outputs([]).unwrap()
        })))
    } else {
        let i64_t = INT_TYPES[6].to_owned();
        let option_sty = option_type(ty.clone());
        let option_ty = Type::from(option_sty.clone());
        let option_array = array_type(*n, option_ty.clone());
        let copy_elem = {
            let mut io = vec![ty.clone(), i64_t.clone()];
            io.extend(vec![option_array.clone(); num_outports - 1]);
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
                .unwrap()
                .outputs();
            let copy0 = copies.next().unwrap(); // We'll return this directly
                                                // Wrap remaining copies into an option
            let set_op = OpType::from(ArrayOpDef::set.to_concrete(option_ty.clone(), *n));
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
        let unwrap_elem = {
            let mut dfb =
                DFGBuilder::new(inout_sig(Type::from(option_ty.clone()), ty.clone())).unwrap();
            let [opt] = dfb.input_wires_arr();
            let [val] = dfb.build_unwrap_sum(1, option_sty.clone(), opt).unwrap();
            dfb.finish_hugr_with_outputs([val]).unwrap()
        };
        let array_ty = array_type(*n, ty.clone());
        let mut dfb =
            DFGBuilder::new(inout_sig(array_ty.clone(), vec![array_ty.clone(); 2])).unwrap();
        let [in_array] = dfb.input_wires_arr();
        // First, Scan `copy_elem` to get the original array back and an array of Some's of the copies
        let scan1 = ArrayScan::new(
            ty.clone(),
            ty.clone(),
            vec![i64_t, option_array],
            *n,
            runtime_reqs(&copy_elem),
        );
        let copy_elem = dfb.add_load_value(Value::function(copy_elem).unwrap());
        let cst0 = dfb.add_load_value(ConstInt::new_u(6, 0).unwrap());
        let [array_of_none] = {
            let fn_none = {
                let mut dfb = DFGBuilder::new(inout_sig(vec![], option_ty.clone())).unwrap();
                let none = dfb
                    .add_dataflow_op(Tag::new(0, vec![type_row![], ty.clone().into()]), [])
                    .unwrap();
                dfb.finish_hugr_with_outputs(none.outputs()).unwrap()
            };
            let rpt = ArrayRepeat::new(option_ty.clone(), *n, runtime_reqs(&fn_none));
            let fn_none = dfb.add_load_value(Value::function(fn_none).unwrap());
            dfb.add_dataflow_op(rpt, [fn_none]).unwrap()
        }
        .outputs_arr();
        let [out_array1, _idx_out, opt_array] = dfb
            .add_dataflow_op(scan1, [in_array, copy_elem, cst0, array_of_none])
            .unwrap()
            .outputs_arr();

        let scan2 = ArrayScan::new(
            option_ty.clone(),
            ty.clone(),
            vec![],
            *n,
            runtime_reqs(&unwrap_elem),
        );
        let unwrap_elem = dfb.add_load_value(Value::function(unwrap_elem).unwrap());
        let [out_array2] = dfb
            .add_dataflow_op(scan2, [opt_array, unwrap_elem])
            .unwrap()
            .outputs_arr();

        Ok(OpReplacement::CompoundOp(Box::new(
            dfb.finish_hugr_with_outputs([out_array1, out_array2])
                .unwrap(),
        )))
    }
}
