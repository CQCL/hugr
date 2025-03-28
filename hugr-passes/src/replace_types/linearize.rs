use std::iter::repeat;
use std::{collections::HashMap, sync::Arc};

use hugr_core::builder::{
    inout_sig, ConditionalBuilder, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    HugrBuilder,
};
use hugr_core::extension::{SignatureError, TypeDef};
use hugr_core::types::{CustomType, Type, TypeArg, TypeBound, TypeEnum, TypeRow};
use hugr_core::{hugr::hugrmut::HugrMut, ops::Tag, IncomingPort, Node, OutgoingPort};
use itertools::Itertools;

use super::{OpReplacement, ParametricType};

/// Configuration for inserting copy and discard operations for linear types
/// outports of which are sources of multiple or 0 edges.
#[derive(Clone, Default)]
pub struct Linearizer {
    // Keyed by lowered type, as only needed when there is an op outputting such
    copy_discard: HashMap<CustomType, (OpReplacement, OpReplacement)>,
    // Copy/discard of parametric types handled by a function that receives the new/lowered type.
    // We do not allow overriding copy/discard of non-extension types, but that
    // can be achieved by *firstly* lowering to a custom linear type, with copy/discard
    // inserted; *secondly* by lowering that to the desired non-extension linear type,
    // including lowering of the copy/discard operations to...whatever.
    copy_discard_parametric: HashMap<
        ParametricType,
        Arc<dyn Fn(&[TypeArg], usize, &Linearizer) -> Result<OpReplacement, LinearizeError>>,
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
        typ: CustomType,
        copy: OpReplacement,
        discard: OpReplacement,
    ) -> Result<(), CustomType> {
        if typ.bound() == TypeBound::Copyable {
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
        copy_discard_fn: impl Fn(&[TypeArg], usize, &Linearizer) -> Result<OpReplacement, LinearizeError>
            + 'static,
    ) {
        // We could look for `src`s TypeDefBound being explicit Copyable, otherwise
        // it depends on the arguments. Since there is no method to get the TypeDefBound
        // from a TypeDef, leaving this for now.
        self.copy_discard_parametric
            .insert(src.into(), Arc::new(copy_discard_fn));
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
        src_node: Node,
        src_port: OutgoingPort,
        typ: &Type, // Or better to get the signature ourselves??
        targets: &[(Node, IncomingPort)],
    ) -> Result<(), LinearizeError> {
        let (tgt_node, tgt_inport) = if targets.len() == 1 {
            *targets.first().unwrap()
        } else {
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
            let copy_discard_op = self
                .copy_discard_op(typ, targets.len())?
                .add_hugr(hugr, src_parent);
            for (n, (tgt_node, tgt_port)) in targets.iter().enumerate() {
                hugr.connect(copy_discard_op, n, *tgt_node, *tgt_port);
            }
            (copy_discard_op, 0.into())
        };
        hugr.connect(src_node, src_port, tgt_node, tgt_inport);
        Ok(())
    }

    /// Gets an [OpReplacement] for copying or discarding a value of type `typ`, i.e.
    /// a recipe for a node with one input of that type and the specified number of
    /// outports. Note that `num_outports` should never be 1 (as no node is required)
    ///
    /// # Panics
    ///
    /// if `num_outports == 1`
    pub fn copy_discard_op(
        &self,
        typ: &Type,
        num_outports: usize,
    ) -> Result<OpReplacement, LinearizeError> {
        if typ.copyable() {
            return Err(LinearizeError::CopyableType(typ.clone()));
        };
        assert!(num_outports != 1);

        match typ.as_type_enum() {
            TypeEnum::Sum(sum_type) => {
                let variants = sum_type
                    .variants()
                    .map(|trv| trv.clone().try_into())
                    .collect::<Result<Vec<TypeRow>, _>>()?;
                let mut cb = ConditionalBuilder::new(
                    variants.clone(),
                    vec![],
                    vec![sum_type.clone().into(); num_outports],
                )
                .unwrap();
                for (tag, variant) in variants.iter().enumerate() {
                    let mut case_b = cb.case_builder(tag).unwrap();
                    let mut elems_for_copy = vec![vec![]; num_outports];
                    for (inp, ty) in case_b.input_wires().zip_eq(variant.iter()) {
                        let inp_copies = if ty.copyable() {
                            repeat(inp).take(num_outports).collect::<Vec<_>>()
                        } else {
                            self.copy_discard_op(ty, num_outports)?
                                .add(&mut case_b, [inp])
                                .unwrap()
                                .outputs()
                                .collect()
                        };
                        for (src, elems) in inp_copies.into_iter().zip_eq(elems_for_copy.iter_mut())
                        {
                            elems.push(src)
                        }
                    }
                    let t = Tag::new(tag, variants.clone());
                    let outputs = elems_for_copy
                        .into_iter()
                        .map(|elems| {
                            let [copy] = case_b
                                .add_dataflow_op(t.clone(), elems)
                                .unwrap()
                                .outputs_arr();
                            copy
                        })
                        .collect::<Vec<_>>(); // must collect to end borrow of `case_b` by closure
                    case_b.finish_with_outputs(outputs).unwrap();
                }
                Ok(OpReplacement::CompoundOp(Box::new(
                    cb.finish_hugr().unwrap(),
                )))
            }
            TypeEnum::Extension(cty) => match self.copy_discard.get(cty) {
                Some((copy, discard)) => Ok(if num_outports == 0 {
                    discard.clone()
                } else {
                    let mut dfb =
                        DFGBuilder::new(inout_sig(typ.clone(), vec![typ.clone(); num_outports]))
                            .unwrap();
                    let [mut src] = dfb.input_wires_arr();
                    let mut outputs = vec![];
                    for _ in 0..num_outports - 1 {
                        let [out0, out1] = copy.clone().add(&mut dfb, [src]).unwrap().outputs_arr();
                        outputs.push(out0);
                        src = out1;
                    }
                    outputs.push(src);
                    OpReplacement::CompoundOp(Box::new(
                        dfb.finish_hugr_with_outputs(outputs).unwrap(),
                    ))
                }),
                None => {
                    let copy_discard_fn = self
                        .copy_discard_parametric
                        .get(&cty.into())
                        .ok_or_else(|| LinearizeError::NeedCopy(typ.clone()))?;
                    copy_discard_fn(cty.args(), num_outports, self)
                }
            },
            TypeEnum::Function(_) => panic!("Ruled out above as copyable"),
            _ => Err(LinearizeError::UnsupportedType(typ.clone())),
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use hugr_core::builder::{
        endo_sig, inout_sig, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    };

    use hugr_core::extension::prelude::usize_t;
    use hugr_core::extension::{TypeDefBound, Version};
    use hugr_core::hugr::views::{DescendantsGraph, HierarchyView};
    use hugr_core::ops::{handle::NodeHandle, ExtensionOp, NamedOp, OpName};
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::std_extensions::collections::array::{array_type, ArrayOpDef};
    use hugr_core::types::{Signature, Type, TypeRow};
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

        let lin_custom_t = e.get_type(LIN_T).unwrap().instantiate([]).unwrap();
        let lin_t = Type::new_extension(lin_custom_t.clone());

        // Configure to lower usize_t to the linear type above
        let copy_op = ExtensionOp::new(e.get_op("copy").unwrap().clone(), []).unwrap();
        let discard_op = ExtensionOp::new(e.get_op("discard").unwrap().clone(), []).unwrap();
        let mut lowerer = ReplaceTypes::default();
        let usize_custom_t = usize_t().as_extension().unwrap().clone();
        lowerer.replace_type(usize_custom_t, lin_t.clone());
        lowerer
            .linearize(
                lin_custom_t,
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
                DescendantsGraph::<hugr_core::Node>::try_new(&h, case1)
                    .unwrap()
                    .nodes()
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
}
