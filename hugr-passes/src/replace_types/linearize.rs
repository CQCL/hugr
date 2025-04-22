use std::borrow::Cow;
use std::iter::repeat;
use std::{collections::HashMap, sync::Arc};

use hugr_core::builder::{
    inout_sig, ConditionalBuilder, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    HugrBuilder,
};
use hugr_core::extension::{SignatureError, TypeDef};
use hugr_core::std_extensions::collections::array::array_type_def;
use hugr_core::types::{CustomType, Signature, Type, TypeArg, TypeEnum, TypeRow};
use hugr_core::{hugr::hugrmut::HugrMut, ops::Tag, HugrView, IncomingPort, Node, Wire};
use itertools::Itertools;

use super::{handlers::linearize_value_array, NodeTemplate, ParametricType};

/// Trait for things that know how to wire up linear outports to other than one
/// target.  Used to restore Hugr validity when a [ReplaceTypes](super::ReplaceTypes)
/// results in types of such outports changing from [Copyable] to linear (i.e.
/// [hugr_core::types::TypeBound::Any]).
///
/// Note that this is not really effective before [monomorphization]: if a
/// function polymorphic over a [Copyable] becomes called with a
/// non-Copyable type argument, [Linearizer] cannot insert copy/discard
/// operations for such a case. However, following [monomorphization], there
/// would be a specific instantiation of the function for the
/// type-that-becomes-linear, into which copy/discard can be inserted.
///
/// [monomorphization]: crate::monomorphize()
/// [Copyable]: hugr_core::types::TypeBound::Copyable
pub trait Linearizer {
    /// Insert copy or discard operations (as appropriate) enough to wire `src`
    /// up to all `targets`.
    ///
    /// The default implementation
    /// * if `targets.len() == 1`, wires `src` to the unique target
    /// * otherwise, makes a single call to [Self::copy_discard_op], inserts that op,
    ///   and wires its outputs 1:1 to each target
    ///
    /// # Errors
    ///
    /// Most variants of [LinearizeError] can be raised, specifically including
    /// [LinearizeError::CopyableType] if the type is [Copyable], in which case the Hugr
    /// will be unchanged.
    ///
    /// [Copyable]: hugr_core::types::TypeBound::Copyable
    ///
    /// # Panics
    ///
    /// if `src` is not a valid Wire (does not identify a dataflow out-port)
    fn insert_copy_discard(
        &self,
        hugr: &mut impl HugrMut,
        src: Wire,
        targets: &[(Node, IncomingPort)],
    ) -> Result<(), LinearizeError> {
        let sig = hugr.signature(src.node()).unwrap();
        let typ = sig.port_type(src.source()).unwrap();
        let (tgt_node, tgt_inport) = if targets.len() == 1 {
            *targets.first().unwrap()
        } else {
            // Fail fast if the edges are nonlocal. (TODO transform to local edges!)
            let src_parent = hugr
                .get_parent(src.node())
                .expect("Root node cannot have out edges");
            if let Some((tgt, tgt_parent)) = targets.iter().find_map(|(tgt, _)| {
                let tgt_parent = hugr
                    .get_parent(*tgt)
                    .expect("Root node cannot have incoming edges");
                (tgt_parent != src_parent).then_some((*tgt, tgt_parent))
            }) {
                return Err(LinearizeError::NoLinearNonLocalEdges {
                    src: src.node(),
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
        hugr.connect(src.node(), src.source(), tgt_node, tgt_inport);
        Ok(())
    }

    /// Gets an [NodeTemplate] for copying or discarding a value of type `typ`, i.e.
    /// a recipe for a node with one input of that type and the specified number of
    /// outports.
    ///
    /// Implementations are free to panic if `num_outports == 1`, such calls should never
    /// occur as source/target can be directly wired without any node/op being required.
    fn copy_discard_op(
        &self,
        typ: &Type,
        num_outports: usize,
    ) -> Result<NodeTemplate, LinearizeError>;
}

/// A configuration for implementing [Linearizer] by delegating to
/// type-specific callbacks, and by  composing them in order to handle compound types
/// such as [TypeEnum::Sum]s.
#[derive(Clone)]
pub struct DelegatingLinearizer {
    // Keyed by lowered type, as only needed when there is an op outputting such
    copy_discard: HashMap<CustomType, (NodeTemplate, NodeTemplate)>,
    // Copy/discard of parametric types handled by a function that receives the new/lowered type.
    // We do not allow overriding copy/discard of non-extension types, but that
    // can be achieved by *firstly* lowering to a custom linear type, with copy/discard
    // inserted; *secondly* by lowering that to the desired non-extension linear type,
    // including lowering of the copy/discard operations to...whatever.
    copy_discard_parametric: HashMap<
        ParametricType,
        Arc<dyn Fn(&[TypeArg], usize, &CallbackHandler) -> Result<NodeTemplate, LinearizeError>>,
    >,
}

impl Default for DelegatingLinearizer {
    fn default() -> Self {
        let mut res = Self::new_empty();
        res.register_callback(array_type_def(), linearize_value_array);
        res
    }
}

/// Implementation of [Linearizer] passed to callbacks, (e.g.) so that callbacks for
/// handling collection types can use it to generate copy/discards of elements.
// (Note, this is its own type just to give a bit of room for future expansion,
// rather than passing a &DelegatingLinearizer directly)
pub struct CallbackHandler<'a>(#[allow(dead_code)] &'a DelegatingLinearizer);

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum LinearizeError {
    #[error("Need copy/discard op for {_0}")]
    NeedCopyDiscard(Type),
    #[error("Copy/discard op for {typ} with {num_outports} outputs had wrong signature {sig:?}")]
    WrongSignature {
        typ: Type,
        num_outports: usize,
        sig: Option<Signature>,
    },
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

impl DelegatingLinearizer {
    /// Makes a new instance. Unlike [Self::default], this does not understand
    /// any extension types, even those in the prelude.
    pub fn new_empty() -> Self {
        Self {
            copy_discard: Default::default(),
            copy_discard_parametric: Default::default(),
        }
    }

    /// Configures this instance that the specified monomorphic type can be copied and/or
    /// discarded via the provided [NodeTemplate]s - directly or as part of a compound type
    /// e.g. [TypeEnum::Sum].
    /// `copy` should have exactly one inport, of type `src`, and two outports, of same type;
    /// `discard` should have exactly one inport, of type 'src', and no outports.
    ///
    /// # Errors
    ///
    /// * [LinearizeError::CopyableType] If `typ` is
    ///   [Copyable](hugr_core::types::TypeBound::Copyable)
    /// * [LinearizeError::WrongSignature] if `copy` or `discard` do not have the
    ///   expected inputs or outputs
    pub fn register_simple(
        &mut self,
        cty: CustomType,
        copy: NodeTemplate,
        discard: NodeTemplate,
    ) -> Result<(), LinearizeError> {
        let typ = Type::new_extension(cty.clone());
        if typ.copyable() {
            return Err(LinearizeError::CopyableType(typ));
        }
        check_sig(&copy, &typ, 2)?;
        check_sig(&discard, &typ, 0)?;
        self.copy_discard.insert(cty, (copy, discard));
        Ok(())
    }

    /// Configures this instance that instances of the specified [TypeDef] (perhaps
    /// polymorphic) can be copied and/or discarded by using the provided callback
    /// to generate a [NodeTemplate] for an appropriate copy/discard operation.
    ///
    /// The callback is given
    /// * the type arguments (as appropriate for the [TypeDef], so perhaps empty)
    /// * the desired number of outports (this will never be 1)
    /// * A [CallbackHandler] that the callback can use it to generate
    ///   `copy`/`discard` ops for other types (e.g. the elements of a collection),
    ///   as part of an [NodeTemplate::CompoundOp].
    ///
    /// Note that [Self::register_simple] takes precedence when the `src` types overlap.
    pub fn register_callback(
        &mut self,
        src: &TypeDef,
        copy_discard_fn: impl Fn(&[TypeArg], usize, &CallbackHandler) -> Result<NodeTemplate, LinearizeError>
            + 'static,
    ) {
        // We could look for `src`s TypeDefBound being explicit Copyable, otherwise
        // it depends on the arguments. Since there is no method to get the TypeDefBound
        // from a TypeDef, leaving this for now.
        self.copy_discard_parametric
            .insert(src.into(), Arc::new(copy_discard_fn));
    }
}

fn check_sig(tmpl: &NodeTemplate, typ: &Type, num_outports: usize) -> Result<(), LinearizeError> {
    let sig = tmpl.signature();
    if sig.as_ref().is_some_and(|sig| {
        sig.io() == (&typ.clone().into(), &vec![typ.clone(); num_outports].into())
    }) {
        Ok(())
    } else {
        Err(LinearizeError::WrongSignature {
            typ: typ.clone(),
            num_outports,
            sig: sig.map(Cow::into_owned),
        })
    }
}

impl Linearizer for DelegatingLinearizer {
    fn copy_discard_op(
        &self,
        typ: &Type,
        num_outports: usize,
    ) -> Result<NodeTemplate, LinearizeError> {
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
                Ok(NodeTemplate::CompoundOp(Box::new(
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
                    NodeTemplate::CompoundOp(Box::new(
                        dfb.finish_hugr_with_outputs(outputs).unwrap(),
                    ))
                }),
                None => {
                    let copy_discard_fn = self
                        .copy_discard_parametric
                        .get(&cty.into())
                        .ok_or_else(|| LinearizeError::NeedCopyDiscard(typ.clone()))?;
                    let tmpl = copy_discard_fn(cty.args(), num_outports, &CallbackHandler(self))?;
                    check_sig(&tmpl, typ, num_outports)?;
                    Ok(tmpl)
                }
            },
            TypeEnum::Function(_) => panic!("Ruled out above as copyable"),
            _ => Err(LinearizeError::UnsupportedType(typ.clone())),
        }
    }
}

impl Linearizer for CallbackHandler<'_> {
    fn copy_discard_op(
        &self,
        typ: &Type,
        num_outports: usize,
    ) -> Result<NodeTemplate, LinearizeError> {
        self.0.copy_discard_op(typ, num_outports)
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::iter::successors;
    use std::sync::Arc;

    use hugr_core::builder::{inout_sig, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer};

    use hugr_core::extension::prelude::{option_type, usize_t};
    use hugr_core::extension::simple_op::MakeExtensionOp;
    use hugr_core::extension::{
        CustomSignatureFunc, OpDef, SignatureError, SignatureFunc, TypeDefBound, Version,
    };
    use hugr_core::hugr::views::{DescendantsGraph, HierarchyView};
    use hugr_core::ops::handle::NodeHandle;
    use hugr_core::ops::{DataflowOpTrait, ExtensionOp, NamedOp, OpName, OpType};
    use hugr_core::std_extensions::arithmetic::int_types::INT_TYPES;
    use hugr_core::std_extensions::collections::value_array::{
        value_array_type, value_array_type_def, VArrayOpDef, VArrayRepeat, VArrayScan,
        VArrayScanDef,
    };
    use hugr_core::types::type_param::TypeParam;
    use hugr_core::types::{
        FuncValueType, PolyFuncTypeRV, Signature, Type, TypeArg, TypeEnum, TypeRow,
    };
    use hugr_core::{hugr::IdentList, type_row, Extension, Hugr, HugrView, Node};
    use itertools::Itertools;
    use rstest::rstest;

    use crate::replace_types::handlers::linearize_value_array;
    use crate::replace_types::{LinearizeError, NodeTemplate, ReplaceTypesError};
    use crate::ReplaceTypes;

    const LIN_T: &str = "Lin";

    struct NWayCopySigFn(Type);
    impl CustomSignatureFunc for NWayCopySigFn {
        fn compute_signature<'o, 'a: 'o>(
            &'a self,
            arg_values: &[TypeArg],
            _def: &'o OpDef,
        ) -> Result<PolyFuncTypeRV, SignatureError> {
            let [TypeArg::BoundedNat { n }] = arg_values else {
                panic!()
            };
            let outs = vec![self.0.clone(); *n as usize];
            Ok(FuncValueType::new(self.0.clone(), outs).into())
        }

        fn static_params(&self) -> &[TypeParam] {
            const JUST_NAT: &[TypeParam] = &[TypeParam::max_nat()];
            JUST_NAT
        }
    }

    fn ext_lowerer() -> (Arc<Extension>, ReplaceTypes) {
        // Extension with a linear type, an n-way parametric copy op, and a discard op
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
                    "discard".into(),
                    String::new(),
                    Signature::new(lin.clone(), vec![]),
                    w,
                )
                .unwrap();
                e.add_op(
                    "copy".into(),
                    String::new(),
                    SignatureFunc::CustomFunc(Box::new(NWayCopySigFn(lin))),
                    w,
                )
                .unwrap();
            },
        );

        let lin_custom_t = e.get_type(LIN_T).unwrap().instantiate([]).unwrap();

        // Configure to lower usize_t to the linear type above, using a 2-way copy only
        let copy_op = ExtensionOp::new(e.get_op("copy").unwrap().clone(), [2.into()]).unwrap();
        let discard_op = ExtensionOp::new(e.get_op("discard").unwrap().clone(), []).unwrap();
        let mut lowerer = ReplaceTypes::default();
        let usize_custom_t = usize_t().as_extension().unwrap().clone();
        lowerer.replace_type(usize_custom_t, Type::new_extension(lin_custom_t.clone()));
        lowerer
            .linearizer()
            .register_simple(
                lin_custom_t,
                NodeTemplate::SingleOp(copy_op.into()),
                NodeTemplate::SingleOp(discard_op.into()),
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
            vec![usize_t(), value_array_type(2, usize_t())],
        ))
        .unwrap();
        let [inp, _] = outer.input_wires_arr();
        let new_array = outer
            .add_dataflow_op(VArrayOpDef::new_array.to_concrete(usize_t(), 2), [inp, inp])
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
                ("collections.value_array.new_array".into(), 1)
            ])
        );
    }

    fn copy_n_discard_one(ty: Type, n: usize) -> (Hugr, Node) {
        let mut outer = DFGBuilder::new(inout_sig(ty.clone(), vec![ty.clone(); n - 1])).unwrap();
        let [inp] = outer.input_wires_arr();
        let inner = outer
            .dfg_builder(inout_sig(ty, vec![]), [inp])
            .unwrap()
            .finish_with_outputs([])
            .unwrap();
        let h = outer.finish_hugr_with_outputs(vec![inp; n - 1]).unwrap();
        (h, inner.node())
    }

    #[rstest]
    fn sums_2way_copy(#[values(2, 3, 4)] num_copies: usize) {
        let (mut h, inner) = copy_n_discard_one(option_type(usize_t()).into(), num_copies);

        let (e, lowerer) = ext_lowerer();
        assert!(lowerer.run(&mut h).unwrap());

        let lin_t = Type::from(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());
        let sum_ty: Type = option_type(lin_t.clone()).into();
        let count_tags = |n| h.children(n).filter(|n| h.get_optype(*n).is_tag()).count();

        // Check we've inserted one Conditional into outer (for copy) and inner (for discard)...
        for (dfg, num_tags, expected_ext_ops) in [
            (inner.node(), 0, vec!["TestExt.discard"]),
            (h.root(), num_copies, vec!["TestExt.copy"; num_copies - 1]), // 2 copy nodes -> 3 outputs, etc.
        ] {
            let [(cond_node, cond)] = h
                .children(dfg)
                .filter_map(|n| h.get_optype(n).as_conditional().map(|c| (n, c)))
                .collect_array()
                .unwrap();
            assert_eq!(
                cond.signature().output(),
                &TypeRow::from(vec![sum_ty.clone(); num_tags])
            );
            let [case0, case1] = h.children(cond_node).collect_array().unwrap();
            // first is for empty variant
            assert_eq!(h.children(case0).count(), 2 + num_tags); // Input, Output
            assert_eq!(count_tags(case0), num_tags);

            // second is for variant of a LIN_T
            assert_eq!(h.children(case1).count(), 3 + num_tags); // Input, Output, copy/discard
            assert_eq!(count_tags(case1), num_tags);
            let ext_ops = DescendantsGraph::<Node>::try_new(&h, case1)
                .unwrap()
                .nodes()
                .filter_map(|n| h.get_optype(n).as_extension_op().map(ExtensionOp::name))
                .collect_vec();
            assert_eq!(ext_ops, expected_ext_ops);
        }
    }

    #[rstest]
    fn sum_nway_copy(#[values(2, 5, 9)] num_copies: usize) {
        let i8_t = || INT_TYPES[3].clone();
        let sum_ty = Type::new_sum([vec![i8_t()], vec![usize_t(); 2]]);

        let (mut h, inner) = copy_n_discard_one(sum_ty, num_copies);
        let (e, _) = ext_lowerer();
        let mut lowerer = ReplaceTypes::default();
        let lin_t_def = e.get_type(LIN_T).unwrap();
        lowerer.replace_type(
            usize_t().as_extension().unwrap().clone(),
            lin_t_def.instantiate([]).unwrap().into(),
        );
        let opdef = e.get_op("copy").unwrap();
        let opdef2 = opdef.clone();
        lowerer
            .linearizer()
            .register_callback(lin_t_def, move |args, num_outs, _| {
                assert!(args.is_empty());
                Ok(NodeTemplate::SingleOp(
                    ExtensionOp::new(opdef2.clone(), [(num_outs as u64).into()])
                        .unwrap()
                        .into(),
                ))
            });
        assert!(lowerer.run(&mut h).unwrap());

        let lin_t = Type::from(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());
        let sum_ty = Type::new_sum([vec![i8_t()], vec![lin_t.clone(); 2]]);
        let count_tags = |n| h.children(n).filter(|n| h.get_optype(*n).is_tag()).count();

        // Check we've inserted one Conditional into outer (for copy) and inner (for discard)...
        for (dfg, num_tags) in [(inner.node(), 0), (h.root(), num_copies)] {
            let [cond] = h
                .children(dfg)
                .filter(|n| h.get_optype(*n).is_conditional())
                .collect_array()
                .unwrap();
            let [case0, case1] = h.children(cond).collect_array().unwrap();
            let out_row = vec![sum_ty.clone(); num_tags].into();
            // first is for empty variant - the only input is Copyable so can be directly wired or ignored
            assert_eq!(h.children(case0).count(), 2 + num_tags); // Input, Output
            assert_eq!(count_tags(case0), num_tags);
            let case0 = h.get_optype(case0).as_case().unwrap();
            assert_eq!(case0.signature.io(), (&vec![i8_t()].into(), &out_row));

            // second is for variant of two elements
            assert_eq!(h.children(case1).count(), 4 + num_tags); // Input, Output, two leaf copies/discards:
            assert_eq!(count_tags(case1), num_tags);
            let ext_ops = h
                .children(case1)
                .filter_map(|n| h.get_optype(n).as_extension_op())
                .collect_vec();
            let expected_op = ExtensionOp::new(opdef.clone(), [(num_tags as u64).into()]).unwrap();
            assert_eq!(ext_ops, vec![&expected_op; 2]);

            let case1 = h.get_optype(case1).as_case().unwrap();
            assert_eq!(
                case1.signature.io(),
                (&vec![lin_t.clone(); 2].into(), &out_row)
            );
        }
    }

    #[test]
    fn bad_sig() {
        // Change usize to QB_T
        let (ext, _) = ext_lowerer();
        let lin_ct = ext.get_type(LIN_T).unwrap().instantiate([]).unwrap();
        let lin_t = Type::from(lin_ct.clone());
        let copy3 = OpType::from(
            ExtensionOp::new(ext.get_op("copy").unwrap().clone(), [3.into()]).unwrap(),
        );
        let copy2 = ExtensionOp::new(ext.get_op("copy").unwrap().clone(), [2.into()]).unwrap();
        let discard = ExtensionOp::new(ext.get_op("discard").unwrap().clone(), []).unwrap();
        let mut replacer = ReplaceTypes::default();
        replacer.replace_type(usize_t().as_extension().unwrap().clone(), lin_t.clone());

        let bad_copy = replacer.linearizer().register_simple(
            lin_ct.clone(),
            NodeTemplate::SingleOp(copy3.clone()),
            NodeTemplate::SingleOp(discard.clone().into()),
        );
        let sig3 = Some(
            Signature::new(lin_t.clone(), vec![lin_t.clone(); 3])
                .with_extension_delta(ext.name().clone()),
        );
        assert_eq!(
            bad_copy,
            Err(LinearizeError::WrongSignature {
                typ: lin_t.clone(),
                num_outports: 2,
                sig: sig3.clone()
            })
        );

        let bad_discard = replacer.linearizer().register_simple(
            lin_ct.clone(),
            NodeTemplate::SingleOp(copy2.into()),
            NodeTemplate::SingleOp(copy3.clone()),
        );

        assert_eq!(
            bad_discard,
            Err(LinearizeError::WrongSignature {
                typ: lin_t.clone(),
                num_outports: 0,
                sig: sig3.clone()
            })
        );

        // Try parametrized instead, but this version always returns 3 outports
        replacer
            .linearizer()
            .register_callback(ext.get_type(LIN_T).unwrap(), move |_args, _, _| {
                Ok(NodeTemplate::SingleOp(copy3.clone()))
            });

        // A hugr that copies a usize
        let dfb = DFGBuilder::new(inout_sig(usize_t(), vec![usize_t(); 2])).unwrap();
        let [inp] = dfb.input_wires_arr();
        let mut h = dfb.finish_hugr_with_outputs([inp, inp]).unwrap();

        assert_eq!(
            replacer.run(&mut h),
            Err(ReplaceTypesError::LinearizeError(
                LinearizeError::WrongSignature {
                    typ: lin_t.clone(),
                    num_outports: 2,
                    sig: sig3.clone()
                }
            ))
        );
    }

    #[rstest]
    fn value_array(#[values(2, 3, 4)] num_outports: usize) {
        let num_new = num_outports - 1;
        let (e, mut lowerer) = ext_lowerer();

        lowerer
            .linearizer()
            .register_callback(value_array_type_def(), linearize_value_array);
        let lin_t = Type::from(e.get_type(LIN_T).unwrap().instantiate([]).unwrap());
        let opt_lin_ty = Type::from(option_type(lin_t.clone()));
        let array_ty = || value_array_type(5, usize_t());
        let mut dfb = DFGBuilder::new(inout_sig(array_ty(), vec![array_ty(); num_new])).unwrap();
        let [array_in] = dfb.input_wires_arr();
        // The outer DFG passes the input array into (1) a DFG that discards it
        let discard = dfb
            .dfg_builder(
                Signature::new(value_array_type(5, usize_t()), type_row![]),
                [array_in],
            )
            .unwrap()
            .finish_with_outputs([])
            .unwrap();
        // and (2) its own output
        let mut h = dfb
            .finish_hugr_with_outputs(vec![array_in; num_new])
            .unwrap();

        assert!(lowerer.run(&mut h).unwrap());

        let (discard_ops, copy_ops): (Vec<_>, Vec<_>) = h
            .nodes()
            .filter_map(|n| h.get_optype(n).as_extension_op().map(|e| (n, e)))
            .partition(|(n, _)| {
                successors(Some(*n), |n| h.get_parent(*n)).contains(&discard.node())
            });
        {
            let [(n, ext_op)] = discard_ops.try_into().unwrap();
            assert!(VArrayScanDef::from_extension_op(ext_op).is_ok());
            assert_eq!(
                ext_op.signature().output,
                TypeRow::from(vec![value_array_type(5, Type::UNIT)])
            );
            assert_eq!(h.linked_inputs(n, 0).next(), None);
        }
        assert_eq!(copy_ops.len(), num_new * 2 + 1); // 1 middle scan; 1repeat+1unwrap per new
        let copy_ops = copy_ops.into_iter().map(|(_, e)| e).collect_vec();
        let rpts = copy_ops
            .iter()
            .copied()
            .filter(|e| VArrayRepeat::from_extension_op(e).is_ok())
            .collect_vec();
        assert_eq!(rpts.len(), num_new);
        for rpt in rpts {
            assert_eq!(
                rpt.signature().output(),
                &TypeRow::from(value_array_type(5, opt_lin_ty.clone()))
            );
        }
        let unwrap_scans = copy_ops
            .iter()
            .filter_map(|e| {
                VArrayScan::from_extension_op(e)
                    .ok()
                    .filter(|sc| sc.acc_tys.is_empty())
            })
            .collect_vec();
        assert_eq!(unwrap_scans.len(), num_new);
        for scan in unwrap_scans {
            assert_eq!(scan.src_ty, opt_lin_ty);
            assert_eq!(scan.tgt_ty, lin_t);
        }

        let copy_sig = copy_ops
            .into_iter()
            .find(|e| VArrayScan::from_extension_op(e).is_ok_and(|sc| !sc.acc_tys.is_empty()))
            .unwrap()
            .signature()
            .into_owned();
        assert_eq!(
            copy_sig.output,
            TypeRow::from(
                [value_array_type(5, lin_t.clone()), INT_TYPES[6].to_owned()]
                    .into_iter()
                    .chain(vec![
                        value_array_type(5, option_type(lin_t.clone()).into());
                        num_new
                    ])
                    .collect_vec()
            )
        );
        assert_eq!(copy_sig.input[0], copy_sig.output[0]);
        assert!(matches!(
            copy_sig.input[1].as_type_enum(),
            TypeEnum::Function(_)
        ));
        assert_eq!(copy_sig.input[2..], copy_sig.output[1..]);
    }
}
