use std::{collections::HashMap, sync::Arc};

use hugr_core::builder::{
    endo_sig, inout_sig, ConditionalBuilder, DFGBuilder, Dataflow, DataflowHugr,
    DataflowSubContainer, HugrBuilder,
};
use hugr_core::extension::prelude::{option_type, UnwrapBuilder};
use hugr_core::extension::{ExtensionSet, SignatureError, TypeDef};
use hugr_core::ops::{OpTrait, OpType, Tag, Value};
use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
use hugr_core::std_extensions::arithmetic::int_ops::IntOpDef;
use hugr_core::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
use hugr_core::std_extensions::collections::array::{
    array_type, ArrayOpDef, ArrayRepeat, ArrayScan,
};
use hugr_core::types::{SumType, Type, TypeArg, TypeEnum, TypeRow};
use hugr_core::{
    hugr::hugrmut::HugrMut, type_row, Hugr, HugrView, IncomingPort, Node, OutgoingPort,
};
use itertools::Itertools;

use super::{OpReplacement, ParametricType};

/// Configuration for inserting copy and discard operations for linear types
/// outports of which are sources of multiple or 0 edges.
#[derive(Clone, Default)]
pub struct Linearizer {
    // Keyed by lowered type, as only needed when there is an op outputting such
    copy_discard: HashMap<Type, (OpReplacement, OpReplacement)>,
    // Copy/discard of parametric types handled by a function that receives the new/lowered type.
    // We do not allow overriding copy/discard of non-extension types, but that
    // can be achieved by *firstly* lowering to a custom linear type, with copy/discard
    // inserted; *secondly* by lowering that to the desired non-extension linear type,
    // including lowering of the copy/discard operations to...whatever.
    copy_discard_parametric: HashMap<
        ParametricType,
        (
            Arc<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
            Arc<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
        ),
    >,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum LinearizeError {
    #[error("Need copy op for {_0}")]
    NeedCopy(Type),
    #[error("Need discard op for {_0}")]
    NeedDiscard(Type),
    #[error("Cannot add nonlocal edge for linear type from {src} (with parent {src_parent}) to {tgt} (with parent {tgt_parent})")]
    NoLinearNonLocalEdges {
        src: Node,
        src_parent: Node,
        tgt: Node,
        tgt_parent: Node,
    },
    /// SignatureError's can happen when converting nested types e.g. Sums
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    /// We cannot linearize (insert copy and discard functions) for
    /// [Variable](TypeEnum::Variable)s, [Row variables](TypeEnum::RowVar),
    /// or [Alias](TypeEnum::Alias)es.
    #[error("Cannot linearize type {_0}")]
    UnsupportedType(Type),
    /// Neither does linearization make sense for copyable types
    #[error("Type {_0} is copyable")]
    CopyableType(Type),
}

impl Linearizer {
    /// Registers a type for linearization by providing copy and discard operations.
    ///
    /// # Errors
    ///
    /// If `typ` is copyable, it is returned as an `Err`.
    pub fn register(
        &mut self,
        typ: Type,
        copy: OpReplacement,
        discard: OpReplacement,
    ) -> Result<(), Type> {
        if typ.copyable() {
            Err(typ)
        } else {
            self.copy_discard.insert(typ, (copy, discard));
            Ok(())
        }
    }

    /// Registers that instances of a parametrized [TypeDef] should be linearized
    /// by providing functions that generate copy and discard functions given the [TypeArg]s.
    pub fn register_parametric(
        &mut self,
        src: &TypeDef,
        copy_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
        discard_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
    ) {
        // We could look for `src`s TypeDefBound being explicit Copyable, otherwise
        // it depends on the arguments. Since there is no method to get the TypeDefBound
        // from a TypeDef, leaving this for now.
        self.copy_discard_parametric
            .insert(src.into(), (Arc::from(copy_fn), Arc::from(discard_fn)));
    }

    /// Insert copy or discard operations (as appropriate) enough to wire `src_port` of `src_node`
    /// up to all `targets`.
    ///
    /// # Errors
    ///
    /// Most variants of [LinearizeError] can be raised, specifically including
    /// [LinearizeError::CopyableType] if the type is [Copyable], in which case the Hugr
    /// will be unchanged.
    ///
    /// [Copyable]: hugr_core::types::TypeBound::Copyable
    pub fn insert_copy_discard(
        &self,
        hugr: &mut impl HugrMut,
        mut src_node: Node,
        mut src_port: OutgoingPort,
        typ: &Type, // Or better to get the signature ourselves??
        targets: &[(Node, IncomingPort)],
    ) -> Result<(), LinearizeError> {
        let (last_node, last_inport) = match targets.last() {
            None => {
                let parent = hugr.get_parent(src_node).unwrap();
                (self.discard_op(typ)?.add_hugr(hugr, parent), 0.into())
            }
            Some(last) => *last,
        };

        if targets.len() > 1 {
            // Fail fast if the edges are nonlocal. (TODO transform to local edges!)
            let src_parent = hugr
                .get_parent(src_node)
                .expect("Root node cannot have out edges");
            if let Some((tgt, tgt_parent)) = targets.iter().find_map(|(tgt, _)| {
                let tgt_parent = hugr
                    .get_parent(*tgt)
                    .expect("Root node cannot have incoming edges");
                (tgt_parent != src_parent).then_some((*tgt, tgt_parent))
            }) {
                return Err(LinearizeError::NoLinearNonLocalEdges {
                    src: src_node,
                    src_parent,
                    tgt,
                    tgt_parent,
                });
            }

            let copy_op = self.copy_op(typ)?;

            for (tgt_node, tgt_port) in &targets[..targets.len() - 1] {
                let n = copy_op
                    .clone()
                    .add_hugr(hugr, hugr.get_parent(src_node).unwrap());
                hugr.connect(src_node, src_port, n, 0);
                hugr.connect(n, 0, *tgt_node, *tgt_port);
                (src_node, src_port) = (n, 1.into());
            }
        }
        hugr.connect(src_node, src_port, last_node, last_inport);
        Ok(())
    }

    /// Gets an [OpReplacement] for copying a value of type `typ`, i.e.
    /// a recipe for a node with one input of that type and two outputs.
    pub fn copy_op(&self, typ: &Type) -> Result<OpReplacement, LinearizeError> {
        if typ.copyable() {
            return Err(LinearizeError::CopyableType(typ.clone()));
        };
        if let Some((copy, _)) = self.copy_discard.get(typ) {
            return Ok(copy.clone());
        }
        match typ.as_type_enum() {
            TypeEnum::Sum(sum_type) => {
                let variants = sum_type
                    .variants()
                    .map(|trv| trv.clone().try_into())
                    .collect::<Result<Vec<TypeRow>, _>>()?;
                let mut cb = ConditionalBuilder::new(
                    variants.clone(),
                    vec![],
                    vec![sum_type.clone().into(); 2],
                )
                .unwrap();
                for (tag, variant) in variants.iter().enumerate() {
                    let mut case_b = cb.case_builder(tag).unwrap();
                    let mut orig_elems = vec![];
                    let mut copy_elems = vec![];
                    for (inp, ty) in case_b.input_wires().zip_eq(variant.iter()) {
                        let [orig_elem, copy_elem] = if ty.copyable() {
                            [inp, inp]
                        } else {
                            self.copy_op(ty)?
                                .add(&mut case_b, [inp])
                                .unwrap()
                                .outputs_arr()
                        };
                        orig_elems.push(orig_elem);
                        copy_elems.push(copy_elem);
                    }
                    let t = Tag::new(tag, variants.clone());
                    let [orig] = case_b
                        .add_dataflow_op(t.clone(), orig_elems)
                        .unwrap()
                        .outputs_arr();
                    let [copy] = case_b.add_dataflow_op(t, copy_elems).unwrap().outputs_arr();
                    case_b.finish_with_outputs([orig, copy]).unwrap();
                }
                Ok(OpReplacement::CompoundOp(Box::new(
                    cb.finish_hugr().unwrap(),
                )))
            }
            TypeEnum::Extension(cty) => {
                let (copy_fn, _) = self
                    .copy_discard_parametric
                    .get(&cty.into())
                    .ok_or_else(|| LinearizeError::NeedCopy(typ.clone()))?;
                copy_fn(cty.args(), self)
            }
            TypeEnum::Function(_) => panic!("Ruled out above as copyable"),
            _ => Err(LinearizeError::UnsupportedType(typ.clone())),
        }
    }

    /// Gets an [OpReplacement] for discarding a value of type `typ`, i.e.
    /// a recipe for a node with one input of that type and no outputs.
    pub fn discard_op(&self, typ: &Type) -> Result<OpReplacement, LinearizeError> {
        if typ.copyable() {
            return Err(LinearizeError::CopyableType(typ.clone()));
        };
        if let Some((_, discard)) = self.copy_discard.get(typ) {
            return Ok(discard.clone());
        }
        match typ.as_type_enum() {
            TypeEnum::Sum(sum_type) => {
                let variants = sum_type
                    .variants()
                    .map(|trv| trv.clone().try_into())
                    .collect::<Result<Vec<TypeRow>, _>>()?;
                let mut cb = ConditionalBuilder::new(variants.clone(), vec![], vec![]).unwrap();
                for (idx, variant) in variants.into_iter().enumerate() {
                    let mut case_b = cb.case_builder(idx).unwrap();
                    for (inp, ty) in case_b.input_wires().zip_eq(variant.iter()) {
                        if !ty.copyable() {
                            self.discard_op(ty)?.add(&mut case_b, [inp]).unwrap();
                        }
                    }
                    case_b.finish_with_outputs([]).unwrap();
                }
                Ok(OpReplacement::CompoundOp(Box::new(
                    cb.finish_hugr().unwrap(),
                )))
            }
            TypeEnum::Extension(cty) => {
                let (_, discard_fn) = self
                    .copy_discard_parametric
                    .get(&cty.into())
                    .ok_or_else(|| LinearizeError::NeedDiscard(typ.clone()))?;
                discard_fn(cty.args(), self)
            }
            TypeEnum::Function(_) => panic!("Ruled out above as copyable"),
            _ => Err(LinearizeError::UnsupportedType(typ.clone())),
        }
    }
}

fn runtime_reqs(h: &Hugr) -> ExtensionSet {
    h.signature(h.root()).unwrap().runtime_reqs.clone()
}

pub fn discard_array(args: &[TypeArg], lin: &Linearizer) -> Result<OpReplacement, LinearizeError> {
    // Require known length i.e. usable only after monomorphization, due to no-variables limitation
    // restriction on OpReplacement::CompoundOp
    let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = args else {
        panic!("Illegal TypeArgs to array: {:?}", args)
    };
    // Make a function that maps the linear element to unit
    let map_fn = {
        let mut dfb = DFGBuilder::new(inout_sig(ty.clone(), Type::UNIT)).unwrap();
        let [to_discard] = dfb.input_wires_arr();
        lin.discard_op(ty)?.add(&mut dfb, [to_discard]).unwrap();
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
}

pub fn copy_array(args: &[TypeArg], lin: &Linearizer) -> Result<OpReplacement, LinearizeError> {
    // Require known length i.e. usable only after monomorphization, due to no-variables limitation
    // restriction on OpReplacement::CompoundOp
    let [TypeArg::BoundedNat { n }, TypeArg::Type { ty }] = args else {
        panic!("Illegal TypeArgs to array: {:?}", args)
    };
    let i64_t = INT_TYPES[6].to_owned();
    let option_sty = option_type(ty.clone());
    let option_ty = Type::from(option_sty.clone());
    let option_array = array_type(*n, option_ty.clone());
    let copy_elem = {
        let mut dfb = DFGBuilder::new(endo_sig(vec![
            ty.clone(),
            i64_t.clone(),
            option_array.clone(),
        ]))
        .unwrap();
        let [elem, idx, opt_array] = dfb.input_wires_arr();
        let [idx_usz] = dfb
            .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [idx])
            .unwrap()
            .outputs_arr();
        let [copy0, copy1] = lin
            .copy_op(ty)?
            .add(&mut dfb, [elem])
            .unwrap()
            .outputs_arr();

        //let variants = option_ty.variants().cloned().map(|row| row.try_into().unwrap()).collect();
        let [tag] = dfb
            .add_dataflow_op(Tag::new(1, vec![type_row![], ty.clone().into()]), [copy1])
            .unwrap()
            .outputs_arr();
        let set_op = OpType::from(ArrayOpDef::set.to_concrete(option_ty.clone(), *n));
        let TypeEnum::Sum(either_st) = set_op.dataflow_signature().unwrap().output[0]
            .as_type_enum()
            .clone()
        else {
            panic!("ArrayOp::set has one output of sum type")
        };
        let [set_result] = dfb
            .add_dataflow_op(set_op, [opt_array, idx_usz, tag])
            .unwrap()
            .outputs_arr();
        // set should always be successful
        let [none, opt_array] = dfb.build_unwrap_sum(1, either_st, set_result).unwrap();
        //the removed element is an option, which should always be none (and thus discardable)
        let [] = dfb
            .build_unwrap_sum(0, SumType::new_option(ty.clone()), none)
            .unwrap();

        let cst1 = dfb.add_load_value(ConstInt::new_u(6, 1).unwrap());
        let [new_idx] = dfb
            .add_dataflow_op(IntOpDef::iadd.with_log_width(6), [idx, cst1])
            .unwrap()
            .outputs_arr();
        dfb.finish_hugr_with_outputs([copy0, new_idx, opt_array])
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
    let mut dfb = DFGBuilder::new(inout_sig(array_ty.clone(), vec![array_ty.clone(); 2])).unwrap();
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

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::iter::successors;
    use std::sync::Arc;

    use hugr_core::builder::{
        endo_sig, inout_sig, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    };

    use hugr_core::extension::prelude::{option_type, usize_t};
    use hugr_core::extension::{simple_op::MakeExtensionOp, TypeDefBound, Version};
    use hugr_core::ops::{handle::NodeHandle, DataflowOpTrait, ExtensionOp, NamedOp, OpName};
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::std_extensions::collections::array::{
        array_type, array_type_def, ArrayOpDef, ArrayRepeat, ArrayScan, ArrayScanDef,
    };
    use hugr_core::types::{Signature, Type, TypeEnum, TypeRow};
    use hugr_core::{hugr::IdentList, type_row, Extension, HugrView};
    use itertools::Itertools;

    use crate::replace_types::OpReplacement;
    use crate::ReplaceTypes;

    const LIN_T: &str = "Lin";

    fn ext_lowerer() -> (Arc<Extension>, ReplaceTypes) {
        // Extension with a linear type, a copy and discard op
        let e = Extension::new_arc(
            IdentList::new_unchecked("TestExt"),
            Version::new(0, 0, 0),
            |e, w| {
                let lin = Type::new_extension(
                    e.add_type(LIN_T.into(), vec![], String::new(), TypeDefBound::any(), w)
                        .unwrap()
                        .instantiate([])
                        .unwrap(),
                );
                e.add_op(
                    "copy".into(),
                    String::new(),
                    Signature::new(lin.clone(), vec![lin.clone(); 2]),
                    w,
                )
                .unwrap();
                e.add_op(
                    "discard".into(),
                    String::new(),
                    Signature::new(lin, vec![]),
                    w,
                )
                .unwrap();
            },
        );

        let lin_t = Type::new_extension(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());

        // Configure to lower usize_t to the linear type above
        let copy_op = ExtensionOp::new(e.get_op("copy").unwrap().clone(), []).unwrap();
        let discard_op = ExtensionOp::new(e.get_op("discard").unwrap().clone(), []).unwrap();
        let mut lowerer = ReplaceTypes::default();
        let usize_custom_t = usize_t().as_extension().unwrap().clone();
        lowerer.replace_type(usize_custom_t, lin_t.clone());
        lowerer
            .linearize(
                lin_t,
                OpReplacement::SingleOp(copy_op.into()),
                OpReplacement::SingleOp(discard_op.into()),
            )
            .unwrap();
        (e, lowerer)
    }

    #[test]
    fn single_values() {
        let (_e, lowerer) = ext_lowerer();
        // Build Hugr - uses first input three times, discards second input (both usize)
        let mut outer = DFGBuilder::new(inout_sig(
            vec![usize_t(); 2],
            vec![usize_t(), array_type(2, usize_t())],
        ))
        .unwrap();
        let [inp, _] = outer.input_wires_arr();
        let new_array = outer
            .add_dataflow_op(ArrayOpDef::new_array.to_concrete(usize_t(), 2), [inp, inp])
            .unwrap();
        let [arr] = new_array.outputs_arr();
        let mut h = outer.finish_hugr_with_outputs([inp, arr]).unwrap();

        assert!(lowerer.run(&mut h).unwrap());

        let ext_ops = h.nodes().filter_map(|n| h.get_optype(n).as_extension_op());
        let mut counts = HashMap::<OpName, u32>::new();
        for e in ext_ops {
            *counts.entry(e.name()).or_default() += 1;
        }
        assert_eq!(
            counts,
            HashMap::from([
                ("TestExt.copy".into(), 2),
                ("TestExt.discard".into(), 1),
                ("collections.array.new_array".into(), 1)
            ])
        );
    }

    #[test]
    fn sums() {
        let i8_t = || INT_TYPES[3].clone();
        let sum_ty = Type::new_sum([vec![i8_t()], vec![usize_t(); 2]]);
        let mut outer = DFGBuilder::new(endo_sig(sum_ty.clone())).unwrap();
        let [inp] = outer.input_wires_arr();
        let inner = outer
            .dfg_builder(inout_sig(sum_ty, vec![]), [inp])
            .unwrap()
            .finish_with_outputs([])
            .unwrap();
        let mut h = outer.finish_hugr_with_outputs([inp]).unwrap();

        let (e, lowerer) = ext_lowerer();
        assert!(lowerer.run(&mut h).unwrap());

        let lin_t = Type::from(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());
        let sum_ty = Type::new_sum([vec![i8_t()], vec![lin_t.clone(); 2]]);
        let copy_out: TypeRow = vec![sum_ty.clone(); 2].into();
        let count_tags = |n| h.children(n).filter(|n| h.get_optype(*n).is_tag()).count();

        // Check we've inserted one Conditional into outer (for copy) and inner (for discard)...
        for (dfg, num_tags, out_row, ext_op_name) in [
            (inner.node(), 0, type_row![], "TestExt.discard"),
            (h.root(), 2, copy_out, "TestExt.copy"),
        ] {
            let [cond] = h
                .children(dfg)
                .filter(|n| h.get_optype(*n).is_conditional())
                .collect_array()
                .unwrap();
            let [case0, case1] = h.children(cond).collect_array().unwrap();
            // first is for empty - the only input is Copyable so can be directly wired or ignored
            assert_eq!(h.children(case0).count(), 2 + num_tags); // Input, Output
            assert_eq!(count_tags(case0), num_tags);
            let case0 = h.get_optype(case0).as_case().unwrap();
            assert_eq!(case0.signature.io(), (&vec![i8_t()].into(), &out_row));

            // second is for two elements
            assert_eq!(h.children(case1).count(), 4 + num_tags); // Input, Output, two leaf copies/discards:
            assert_eq!(count_tags(case1), num_tags);
            assert_eq!(
                h.children(case1)
                    .filter_map(|n| h.get_optype(n).as_extension_op().map(ExtensionOp::name))
                    .collect_vec(),
                vec![ext_op_name; 2]
            );
            assert_eq!(
                h.get_optype(case1).as_case().unwrap().signature.io(),
                (&vec![lin_t.clone(); 2].into(), &out_row)
            );
        }
    }

    #[test]
    fn array() {
        let (e, mut lowerer) = ext_lowerer();

        lowerer.linearize_parametric(
            array_type_def(),
            Box::new(super::copy_array),
            Box::new(super::discard_array),
        );
        let lin_t = Type::from(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());
        let opt_lin_ty = Type::from(option_type(lin_t.clone()));
        let mut dfb = DFGBuilder::new(endo_sig(array_type(5, usize_t()))).unwrap();
        let [array_in] = dfb.input_wires_arr();
        // The outer DFG passes the input array into (1) a DFG that discards it
        let discard = dfb
            .dfg_builder(
                Signature::new(array_type(5, usize_t()), type_row![]),
                [array_in],
            )
            .unwrap()
            .finish_with_outputs([])
            .unwrap();
        // and (2) its own output
        let mut h = dfb.finish_hugr_with_outputs([array_in]).unwrap();

        assert!(lowerer.run(&mut h).unwrap());

        let (discard_ops, copy_ops): (Vec<_>, Vec<_>) = h
            .nodes()
            .filter_map(|n| h.get_optype(n).as_extension_op().map(|e| (n, e)))
            .partition(|(n, _)| {
                successors(Some(*n), |n| h.get_parent(*n)).contains(&discard.node())
            });
        {
            let [(n, ext_op)] = discard_ops.try_into().unwrap();
            assert!(ArrayScanDef::from_extension_op(ext_op).is_ok());
            assert_eq!(
                ext_op.signature().output,
                TypeRow::from(vec![array_type(5, Type::UNIT)])
            );
            assert_eq!(h.linked_inputs(n, 0).next(), None);
        }
        assert_eq!(copy_ops.len(), 3);
        let copy_ops = copy_ops.into_iter().map(|(_, e)| e).collect_vec();
        let rpt = *copy_ops
            .iter()
            .find(|e| ArrayRepeat::from_extension_op(e).is_ok())
            .unwrap();
        assert_eq!(
            rpt.signature().output(),
            &TypeRow::from(array_type(5, opt_lin_ty.clone()))
        );
        let scan0 = copy_ops
            .iter()
            .find_map(|e| {
                ArrayScan::from_extension_op(e)
                    .ok()
                    .filter(|sc| sc.acc_tys.is_empty())
            })
            .unwrap();
        assert_eq!(scan0.src_ty, opt_lin_ty);
        assert_eq!(scan0.tgt_ty, lin_t);

        let scan2 = *copy_ops
            .iter()
            .find(|e| ArrayScan::from_extension_op(e).is_ok_and(|sc| !sc.acc_tys.is_empty()))
            .unwrap();
        let sig = scan2.signature().into_owned();
        assert_eq!(
            sig.output,
            TypeRow::from(vec![
                array_type(5, lin_t.clone()),
                INT_TYPES[6].to_owned(),
                array_type(5, option_type(lin_t.clone()).into())
            ])
        );
        assert_eq!(sig.input[0], sig.output[0]);
        assert!(matches!(sig.input[1].as_type_enum(), TypeEnum::Function(_)));
        assert_eq!(sig.input[2..], sig.output[1..]);
    }
}
