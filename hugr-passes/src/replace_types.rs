#![allow(clippy::type_complexity)]
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use hugr_core::extension::{ExtensionId, OpDef, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{OpaqueValue, Sum};
use hugr_core::ops::{
    AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock, ExtensionOp,
    FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpType, Output, Tag, TailLoop, Value,
    CFG, DFG,
};
use hugr_core::types::{CustomType, Transformable, Type, TypeArg, TypeEnum, TypeTransformer};
use hugr_core::{Hugr, Node};

use crate::validation::{ValidatePassError, ValidationLevel};

/// A thing to which an Op can be lowered, i.e. with which a node can be replaced.
#[derive(Clone, Debug, PartialEq)]
pub enum OpReplacement {
    /// Keep the same node (inputs/outputs, modulo lowering of types therein), change only the op
    SingleOp(OpType),
    /// Defines a sub-Hugr to splice in place of the op - a [CFG](OpType::CFG),
    /// [Conditional](OpType::Conditional) or [DFG](OpType::DFG), which must have
    /// the same (lowered) inputs and outputs as the original op.
    // Not a FuncDefn, nor Case/DataflowBlock
    /// Note this will be of limited use before [monomorphization](super::monomorphize()) because
    /// the sub-Hugr will not be able to use type variables present in the op.
    // TODO: store also a vec<TypeParam>, and update Hugr::validate to take &[TypeParam]s
    // (defaulting to empty list) - see https://github.com/CQCL/hugr/issues/709
    CompoundOp(Box<Hugr>),
    // TODO allow also Call to a Node in the existing Hugr
    // (can't see any other way to achieve multiple calls to the same decl.
    // So client should add the functions before lowering, then remove unused ones afterwards.)
}

impl OpReplacement {
    fn replace(&self, hugr: &mut impl HugrMut, n: Node) {
        assert_eq!(hugr.children(n).count(), 0);
        let new_optype = match self.clone() {
            OpReplacement::SingleOp(op_type) => op_type,
            OpReplacement::CompoundOp(new_h) => {
                let new_root = hugr.insert_hugr(n, *new_h).new_root;
                let children = hugr.children(new_root).collect::<Vec<_>>();
                let root_opty = hugr.remove_node(new_root);
                for ch in children {
                    hugr.set_parent(ch, n);
                }
                root_opty
            }
        };
        *hugr.optype_mut(n) = new_optype;
    }
}

#[derive(Clone, Default)]
pub struct ReplaceTypes {
    /// Handles simple cases like T1 -> T2.
    /// If T1 is Copyable and T2 Linear, then error will be raised if we find e.g.
    /// ArrayOfCopyables(T1). This would require an additional entry for that.
    type_map: HashMap<CustomType, Type>,
    /// Parametric types are handled by a function which receives the lowered typeargs.
    param_types: HashMap<ParametricType, Arc<dyn Fn(&[TypeArg]) -> Option<Type>>>,
    // Handles simple cases Op1 -> Op2.
    op_map: HashMap<OpHashWrapper, OpReplacement>,
    // Called after lowering typeargs; return None to use original OpDef
    param_ops: HashMap<ParametricOp, Arc<dyn Fn(&[TypeArg]) -> Option<OpReplacement>>>,
    consts: HashMap<CustomType, Arc<dyn Fn(&OpaqueValue) -> Value>>,
    param_consts: HashMap<ParametricType, Arc<dyn Fn(&OpaqueValue) -> Option<Value>>>,
    validation: ValidationLevel,
}

impl TypeTransformer for ReplaceTypes {
    type Err = ReplaceTypesError;

    fn apply_custom(&self, ct: &CustomType) -> Result<Option<Type>, Self::Err> {
        Ok(if let Some(res) = self.type_map.get(ct) {
            Some(res.clone())
        } else if let Some(dest_fn) = self.param_types.get(&ct.into()) {
            // `ct` has not had args transformed
            let mut nargs = ct.args().to_vec();
            // We don't care if `nargs` are changed, we're just calling `dest_fn`
            nargs
                .iter_mut()
                .try_for_each(|ta| ta.transform(self).map(|_ch| ()))?;
            dest_fn(&nargs)
        } else {
            None
        })
    }
}

#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ReplaceTypesError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    #[error(transparent)]
    ValidationError(#[from] ValidatePassError),
}

impl ReplaceTypes {
    /// Sets the validation level used before and after the pass is run.
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Configures this instance to change occurrences of type `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* [TypeDef], this takes
    /// precedence over [Self::replace_parametrized_type] where the `src`s overlap. Thus, this
    /// should only be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as
    /// substitution (parametric polymorphism) happening later will not respect this lowering.
    pub fn replace_type(&mut self, src: CustomType, dest: Type) {
        // We could check that 'dest' is copyable or 'src' is linear, but since we can't
        // check that for parametrized types, we'll be consistent and not check here either.
        self.type_map.insert(src, dest);
    }

    /// Configures this instance to change occurrences of a parametrized type `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been lowered (e.g. they may not
    /// fit the bounds of the original type). The callback may return `None` to indicate
    /// no change (in which case the supplied/lowered TypeArgs will be given to `src`).
    pub fn replace_parametrized_type(
        &mut self,
        src: &TypeDef,
        dest_fn: impl Fn(&[TypeArg]) -> Option<Type> + 'static,
    ) {
        // No way to check that dest_fn never produces a linear type.
        // We could require copy/discard-generators if src is Copyable, or *might be*
        // (depending on arguments - i.e. if src's TypeDefBound is anything other than
        // `TypeDefBound::Explicit(TypeBound::Copyable)`) but that seems an annoying
        // overapproximation. Moreover, these depend upon the *return type* of the Fn.
        self.param_types.insert(src.into(), Arc::new(dest_fn));
    }

    /// Configures this instance to change occurrences of `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* [OpDef], this takes
    /// precedence over [Self::lower_parametric_op] where the `src`s overlap. Thus, this
    /// should only be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as
    /// substitution (parametric polymorphism) happening later will not respect this
    /// lowering.
    pub fn replace_op(&mut self, src: &ExtensionOp, dest: OpReplacement) {
        self.op_map.insert(OpHashWrapper::from(src), dest);
    }

    /// Configures this instance to change occurrences of a parametrized op `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been lowered (e.g. they may not
    /// fit the bounds of the original op).
    ///
    /// If the Callback returns None, the new typeargs will be applied to the original op.
    pub fn replace_parametrized_op(
        &mut self,
        src: &OpDef,
        dest_fn: impl Fn(&[TypeArg]) -> Option<OpReplacement> + 'static,
    ) {
        self.param_ops.insert(src.into(), Arc::new(dest_fn));
    }

    /// Configures this instance to change [Const]s of type `src_ty`, using
    /// a callback that is passed the value of the constant (of that type).
    ///
    /// Note that if `src_ty` is an instance of a *parametrized* [TypeDef],
    /// this takes precedence over [Self::lower_consts_parametric] where
    /// the `src_ty`s overlap.
    pub fn replace_consts(
        &mut self,
        src_ty: CustomType,
        const_fn: impl Fn(&OpaqueValue) -> Value + 'static,
    ) {
        self.consts.insert(src_ty.clone(), Arc::new(const_fn));
    }

    /// Configures this instance to change [Const]s of all types that are instances
    /// of a parametrized typedef `src_ty`, using a callback that is passed the
    /// value of the constant (the [OpaqueValue] contains the [TypeArg]s).
    pub fn replace_consts_parametrized(
        &mut self,
        src_ty: &TypeDef,
        const_fn: impl Fn(&OpaqueValue) -> Option<Value> + 'static,
    ) {
        self.param_consts.insert(src_ty.into(), Arc::new(const_fn));
    }

    /// Run the pass using specified configuration.
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<bool, ReplaceTypesError> {
        self.validation
            .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
    }

    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ReplaceTypesError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            changed |= self.change_node(hugr, n)?;
        }
        Ok(changed)
    }

    fn change_node(&self, hugr: &mut impl HugrMut, n: Node) -> Result<bool, ReplaceTypesError> {
        match hugr.optype_mut(n) {
            OpType::FuncDefn(FuncDefn { signature, .. })
            | OpType::FuncDecl(FuncDecl { signature, .. }) => signature.body_mut().transform(self),
            OpType::LoadConstant(LoadConstant { datatype: ty })
            | OpType::AliasDefn(AliasDefn { definition: ty, .. }) => ty.transform(self),

            OpType::ExitBlock(ExitBlock { cfg_outputs: types })
            | OpType::Input(Input { types })
            | OpType::Output(Output { types }) => types.transform(self),
            OpType::LoadFunction(LoadFunction {
                func_sig,
                type_args,
                instantiation,
            })
            | OpType::Call(Call {
                func_sig,
                type_args,
                instantiation,
            }) => {
                let change = func_sig.body_mut().transform(self)? | type_args.transform(self)?;
                if change {
                    let new_inst = func_sig
                        .instantiate(type_args)
                        .map_err(ReplaceTypesError::SignatureError)?;
                    *instantiation = new_inst;
                }
                Ok(change)
            }
            OpType::Case(Case { signature })
            | OpType::CFG(CFG { signature })
            | OpType::DFG(DFG { signature })
            | OpType::CallIndirect(CallIndirect { signature }) => signature.transform(self),
            OpType::Tag(Tag { variants, .. }) => variants.transform(self),
            OpType::Conditional(Conditional {
                other_inputs: row1,
                outputs: row2,
                sum_rows,
                ..
            })
            | OpType::DataflowBlock(DataflowBlock {
                inputs: row1,
                other_outputs: row2,
                sum_rows,
                ..
            }) => Ok(row1.transform(self)? | row2.transform(self)? | sum_rows.transform(self)?),
            OpType::TailLoop(TailLoop {
                just_inputs,
                just_outputs,
                rest,
                ..
            }) => Ok(just_inputs.transform(self)?
                | just_outputs.transform(self)?
                | rest.transform(self)?),

            OpType::Const(Const { value, .. }) => self.change_value(value),
            OpType::ExtensionOp(ext_op) => Ok(
                if let Some(replacement) = self.op_map.get(&OpHashWrapper::from(&*ext_op)) {
                    replacement.replace(hugr, n); // Copy/discard insertion done by caller
                    true
                } else {
                    let def = ext_op.def_arc();
                    let mut args = ext_op.args().to_vec();
                    let ch = args.transform(self)?;
                    if let Some(replacement) = self
                        .param_ops
                        .get(&def.as_ref().into())
                        .and_then(|rep_fn| rep_fn(&args))
                    {
                        replacement.replace(hugr, n);
                        true
                    } else {
                        if ch {
                            *ext_op = ExtensionOp::new(def.clone(), args)?;
                        }
                        ch
                    }
                },
            ),

            OpType::OpaqueOp(_) => panic!("OpaqueOp should not be in a Hugr"),

            OpType::AliasDecl(_) | OpType::Module(_) => Ok(false),
            _ => todo!(),
        }
    }

    fn change_value(&self, value: &mut Value) -> Result<bool, ReplaceTypesError> {
        match value {
            Value::Sum(Sum {
                values, sum_type, ..
            }) => {
                let mut any_change = false;
                for value in values {
                    any_change |= self.change_value(value)?;
                }
                any_change |= sum_type.transform(self)?;
                Ok(any_change)
            }
            Value::Extension { e } => Ok('changed: {
                if let TypeEnum::Extension(exty) = e.get_type().as_type_enum() {
                    if let Some(new_const) =
                        self.consts.get(exty).map(|const_fn| const_fn(e)).or(self
                            .param_consts
                            .get(&exty.into())
                            .and_then(|const_fn| const_fn(e)))
                    {
                        *value = new_const;
                        break 'changed true;
                    }
                }
                false
            }),
            Value::Function { hugr } => self.run_no_validate(&mut **hugr),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct OpHashWrapper {
    ext_name: ExtensionId,
    op_name: String, // Only because SmolStr not in hugr-passes yet
    args: Vec<TypeArg>,
}

impl From<&ExtensionOp> for OpHashWrapper {
    fn from(op: &ExtensionOp) -> Self {
        Self {
            ext_name: op.def().extension_id().clone(),
            op_name: op.def().name().to_string(),
            args: op.args().to_vec(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ParametricType(ExtensionId, String);

impl From<&TypeDef> for ParametricType {
    fn from(value: &TypeDef) -> Self {
        Self(value.extension_id().clone(), value.name().to_string())
    }
}

impl From<&CustomType> for ParametricType {
    fn from(value: &CustomType) -> Self {
        Self(value.extension().clone(), value.name().to_string())
    }
}

// Separate from above for clarity
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ParametricOp(ExtensionId, String);

impl From<&OpDef> for ParametricOp {
    fn from(value: &OpDef) -> Self {
        Self(value.extension_id().clone(), value.name().to_string())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use hugr_core::builder::{
        inout_sig, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        HugrBuilder, ModuleBuilder, SubContainer, TailLoopBuilder,
    };
    use hugr_core::extension::prelude::{
        bool_t, option_type, qb_t, usize_t, ConstUsize, UnwrapBuilder,
    };
    use hugr_core::extension::simple_op::MakeExtensionOp;
    use hugr_core::extension::{TypeDefBound, Version};

    use hugr_core::ops::{ExtensionOp, NamedOp, OpTrait, OpType, Tag, Value};
    use hugr_core::std_extensions::arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES};
    use hugr_core::std_extensions::collections::array::{
        array_type, ArrayOp, ArrayOpDef, ArrayValue,
    };
    use hugr_core::std_extensions::collections::list::{
        list_type, list_type_def, ListOp, ListValue,
    };

    use hugr_core::types::{
        PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeEnum, TypeRow,
    };
    use hugr_core::{hugr::IdentList, type_row, Extension, HugrView};
    use itertools::Itertools;

    use super::{ReplaceTypes, OpReplacement};

    const PACKED_VEC: &str = "PackedVec";
    const READ: &str = "read";

    fn i64_t() -> Type {
        INT_TYPES[6].clone()
    }

    fn read_op(ext: &Arc<Extension>, t: Type) -> ExtensionOp {
        ExtensionOp::new(ext.get_op(READ).unwrap().clone(), [t.into()]).unwrap()
    }

    fn just_elem_type(args: &[TypeArg]) -> &Type {
        let [TypeArg::Type { ty }] = args else {
            panic!("Expected just elem type")
        };
        ty
    }

    fn ext() -> Arc<Extension> {
        Extension::new_arc(
            IdentList::new("TestExt").unwrap(),
            Version::new(0, 0, 1),
            |ext, w| {
                let pv_of_var = ext
                    .add_type(
                        PACKED_VEC.into(),
                        vec![TypeBound::Any.into()],
                        String::new(),
                        TypeDefBound::from_params(vec![0]),
                        w,
                    )
                    .unwrap()
                    .instantiate(vec![Type::new_var_use(0, TypeBound::Copyable).into()])
                    .unwrap();
                ext.add_op(
                    READ.into(),
                    "".into(),
                    PolyFuncType::new(
                        vec![TypeBound::Copyable.into()],
                        Signature::new(
                            vec![pv_of_var.into(), i64_t()],
                            Type::new_var_use(0, TypeBound::Any),
                        ),
                    ),
                    w,
                )
                .unwrap();
                ext.add_op(
                    "lowered_read_bool".into(),
                    "".into(),
                    Signature::new(vec![i64_t(); 2], bool_t()),
                    w,
                )
                .unwrap();
            },
        )
    }

    fn lowerer(ext: &Arc<Extension>) -> ReplaceTypes {
        fn lowered_read(args: &[TypeArg]) -> Option<OpReplacement> {
            let ty = just_elem_type(args);
            let mut dfb = DFGBuilder::new(inout_sig(
                vec![array_type(64, ty.clone()), i64_t()],
                ty.clone(),
            ))
            .unwrap();
            let [val, idx] = dfb.input_wires_arr();
            let [idx] = dfb
                .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [idx])
                .unwrap()
                .outputs_arr();
            let [opt] = dfb
                .add_dataflow_op(ArrayOpDef::get.to_concrete(ty.clone(), 64), [val, idx])
                .unwrap()
                .outputs_arr();
            let [res] = dfb
                .build_unwrap_sum(1, option_type(Type::from(ty.clone())), opt)
                .unwrap();
            Some(OpReplacement::CompoundOp(Box::new(
                dfb.finish_hugr_with_outputs([res]).unwrap(),
            )))
        }
        let pv = ext.get_type(PACKED_VEC).unwrap();
        let mut lw = ReplaceTypes::default();
        lw.replace_type(pv.instantiate([bool_t().into()]).unwrap(), i64_t());
        lw.replace_parametrized_type(
            pv,
            Box::new(|args: &[TypeArg]| Some(array_type(64, just_elem_type(args).clone()))),
        );
        lw.replace_op(
            &read_op(ext, bool_t()),
            OpReplacement::SingleOp(
                ExtensionOp::new(ext.get_op("lowered_read_bool").unwrap().clone(), [])
                    .unwrap()
                    .into(),
            ),
        );
        lw.replace_parametrized_op(ext.get_op(READ).unwrap().as_ref(), Box::new(lowered_read));
        lw
    }

    #[test]
    fn module_func_cfg_call() {
        let ext = ext();
        let coln = ext.get_type(PACKED_VEC).unwrap();
        let c_int = Type::from(coln.instantiate([i64_t().into()]).unwrap());
        let c_bool = Type::from(coln.instantiate([bool_t().into()]).unwrap());
        let mut mb = ModuleBuilder::new();
        let sig = Signature::new_endo(Type::new_var_use(0, TypeBound::Any));
        let fb = mb
            .define_function("id", PolyFuncType::new([TypeBound::Any.into()], sig))
            .unwrap();
        let inps = fb.input_wires();
        let id = fb.finish_with_outputs(inps).unwrap();

        let sig = Signature::new(vec![i64_t(), c_int.clone(), c_bool.clone()], bool_t())
            .with_extension_delta(ext.name.clone());
        let mut fb = mb.define_function("main", sig).unwrap();
        let [idx, indices, bools] = fb.input_wires_arr();
        let [indices] = fb
            .call(id.handle(), &[c_int.into()], [indices])
            .unwrap()
            .outputs_arr();
        let [idx2] = fb
            .add_dataflow_op(read_op(&ext, i64_t()), [indices, idx])
            .unwrap()
            .outputs_arr();
        let mut cfg = fb
            .cfg_builder([(i64_t(), idx2), (c_bool.clone(), bools)], bool_t().into())
            .unwrap();
        let mut entry = cfg.entry_builder([bool_t().into()], type_row![]).unwrap();
        let [idx2, bools] = entry.input_wires_arr();
        let [bools] = entry
            .call(id.handle(), &[c_bool.into()], [bools])
            .unwrap()
            .outputs_arr();
        let bool_read_op = entry
            .add_dataflow_op(read_op(&ext, bool_t()), [bools, idx2])
            .unwrap();
        let [tagged] = entry
            .add_dataflow_op(
                OpType::Tag(Tag::new(0, vec![bool_t().into()])),
                bool_read_op.outputs(),
            )
            .unwrap()
            .outputs_arr();
        let entry = entry.finish_with_outputs(tagged, []).unwrap();
        cfg.branch(&entry, 0, &cfg.exit_block()).unwrap();
        let cfg = cfg.finish_sub_container().unwrap();
        fb.finish_with_outputs(cfg.outputs()).unwrap();
        let mut h = mb.finish_hugr().unwrap();

        assert!(lowerer(&ext).run(&mut h).unwrap());

        let ext_ops = h.nodes().filter_map(|n| h.get_optype(n).as_extension_op());
        assert_eq!(
            ext_ops.map(|e| e.def().name()).sorted().collect_vec(),
            ["get", "itousize", "lowered_read_bool", "panic",]
        );
    }

    #[test]
    fn dfg_conditional_case() {
        let ext = ext();
        let coln = ext.get_type(PACKED_VEC).unwrap();
        let pv = |t: Type| Type::new_extension(coln.instantiate([t.into()]).unwrap());
        let sum_rows = [vec![pv(pv(bool_t())), i64_t()].into(), pv(i64_t()).into()];
        let mut dfb = DFGBuilder::new(inout_sig(
            vec![Type::new_sum(sum_rows.clone()), pv(bool_t()), pv(i64_t())],
            vec![pv(bool_t()), pv(i64_t())],
        ))
        .unwrap();
        let [sum, vb, vi] = dfb.input_wires_arr();
        let mut cb = dfb
            .conditional_builder(
                (sum_rows, sum),
                [(pv(bool_t()), vb), (pv(i64_t()), vi)],
                vec![pv(bool_t()), pv(i64_t())].into(),
            )
            .unwrap();
        let mut case0 = cb.case_builder(0).unwrap();
        let [vvb, i, _, vi0] = case0.input_wires_arr();
        let [vb0] = case0
            .add_dataflow_op(read_op(&ext, pv(bool_t())), [vvb, i])
            .unwrap()
            .outputs_arr();
        case0.finish_with_outputs([vb0, vi0]).unwrap();

        let case1 = cb.case_builder(1).unwrap();
        let [vi, vb1, _vi1] = case1.input_wires_arr();
        case1.finish_with_outputs([vb1, vi]).unwrap();
        let cond = cb.finish_sub_container().unwrap();
        let mut h = dfb.finish_hugr_with_outputs(cond.outputs()).unwrap();

        lowerer(&ext).run(&mut h).unwrap();

        let ext_ops = h
            .nodes()
            .filter_map(|n| h.get_optype(n).as_extension_op())
            .collect_vec();
        assert_eq!(
            ext_ops
                .iter()
                .map(|e| e.def().name())
                .sorted()
                .collect_vec(),
            ["get", "itousize", "panic"]
        );
        // The PackedVec<PackedVec<bool>> becomes an array<i64>
        let [array_get] = ext_ops
            .into_iter()
            .filter_map(|e| ArrayOp::from_extension_op(e).ok())
            .collect_array()
            .unwrap();
        assert_eq!(array_get, ArrayOpDef::get.to_concrete(i64_t(), 64));
    }

    #[test]
    fn loop_const() {
        let cu = |u| ConstUsize::new(u).into();
        let mut tl = TailLoopBuilder::new(
            list_type(usize_t()),
            list_type(bool_t()),
            list_type(usize_t()),
        )
        .unwrap();
        let [_, bools] = tl.input_wires_arr();
        let st = SumType::new(vec![list_type(usize_t()); 2]);
        let pred = tl.add_load_value(
            Value::sum(
                0,
                [ListValue::new(usize_t(), [cu(1), cu(3), cu(3), cu(7)]).into()],
                st,
            )
            .unwrap(),
        );
        tl.set_outputs(pred, [bools]).unwrap();
        let mut h = tl.finish_hugr().unwrap();

        // 1. Lower List<T> to Array<10, T> UNLESS T is usize_t() or bool_t - this should have no effect
        let mut lowerer = ReplaceTypes::default();
        lowerer.replace_parametrized_type(list_type_def(), |args| {
            let ty = just_elem_type(args);
            (![usize_t(), bool_t()].contains(ty)).then_some(array_type(10, ty.clone()))
        });
        let backup = h.clone();
        assert!(!lowerer.run(&mut h).unwrap());
        assert_eq!(h, backup);

        //2. Lower List<T> to Array<10, T> UNLESS T is usize_t() - this leaves the Const unchanged
        let mut lowerer = ReplaceTypes::default();
        lowerer.replace_parametrized_type(list_type_def(), |args| {
            let ty = just_elem_type(args);
            (usize_t() != *ty).then_some(array_type(10, ty.clone()))
        });
        assert!(lowerer.run(&mut h).unwrap());
        let sig = h.signature(h.root()).unwrap();
        assert_eq!(
            sig.input(),
            &TypeRow::from(vec![list_type(usize_t()), array_type(10, bool_t())])
        );
        assert_eq!(sig.input(), sig.output());

        // 3. Lower all List<T> to Array<4,T> so we can use List's handy CustomConst
        let mut h = backup;
        let mut lowerer = ReplaceTypes::default();
        lowerer.replace_parametrized_type(
            list_type_def(),
            Box::new(|args: &[TypeArg]| Some(array_type(4, just_elem_type(args).clone()))),
        );
        lowerer.replace_consts_parametrized(list_type_def(), |opaq| {
            let lv = opaq
                .value()
                .downcast_ref::<ListValue>()
                .expect("Only one constant in test");
            Some(ArrayValue::new(lv.get_element_type().clone(), lv.get_contents().to_vec()).into())
        });
        lowerer.run(&mut h).unwrap();

        assert_eq!(
            h.get_optype(pred.node())
                .as_load_constant()
                .map(|lc| lc.constant_type()),
            Some(&Type::new_sum(vec![
                Type::from(array_type(4, usize_t()));
                2
            ]))
        );
    }

    #[test]
    fn partial_replace() {
        let e = Extension::new_arc(
            IdentList::new_unchecked("NoBoundsCheck"),
            Version::new(0, 0, 0),
            |e, w| {
                let params = vec![TypeBound::Any.into()];
                let tv = Type::new_var_use(0, TypeBound::Any);
                let list_of_var = list_type(tv.clone());
                e.add_op(
                    READ.into(),
                    "Like List::get but without the option".to_string(),
                    PolyFuncType::new(params, Signature::new(vec![list_of_var, usize_t()], tv)),
                    w,
                )
                .unwrap();
            },
        );
        fn option_contents(ty: &Type) -> Option<Type> {
            let row = ty.as_sum()?.get_variant(1).unwrap().clone();
            let elem = row.into_owned().into_iter().exactly_one().unwrap();
            Some(elem.try_into_type().unwrap())
        }
        let i32_t = || INT_TYPES[5].to_owned();
        let opt_i32 = Type::from(option_type(i32_t()));
        let TypeEnum::Extension(i32_custom_t) = i32_t().as_type_enum().clone() else {
            panic!()
        };
        let mut dfb = DFGBuilder::new(inout_sig(
            vec![list_type(i32_t()), list_type(opt_i32.clone())],
            vec![i32_t(), opt_i32.clone()],
        ))
        .unwrap();
        let [l_i, l_oi] = dfb.input_wires_arr();
        let idx = dfb.add_load_value(ConstUsize::new(2));
        let [i] = dfb
            .add_dataflow_op(read_op(&e, i32_t()), [l_i, idx])
            .unwrap()
            .outputs_arr();
        let [oi] = dfb
            .add_dataflow_op(read_op(&e, opt_i32.clone()), [l_oi, idx])
            .unwrap()
            .outputs_arr();
        let mut h = dfb.finish_hugr_with_outputs([i, oi]).unwrap();

        let mut lowerer = ReplaceTypes::default();
        lowerer.replace_type(i32_custom_t, qb_t());
        // Lower list<option<x>> to list<x>
        lowerer.replace_parametrized_type(list_type_def(), |args| {
            option_contents(just_elem_type(args)).map(list_type)
        });
        // and read<option<x>> to get<x> - the latter has the expected option<x> return type
        lowerer.replace_parametrized_op(
            e.get_op(READ).unwrap().as_ref(),
            Box::new(|args: &[TypeArg]| {
                option_contents(just_elem_type(args)).map(|elem| {
                    OpReplacement::SingleOp(
                        ListOp::get
                            .with_type(elem)
                            .to_extension_op()
                            .unwrap()
                            .into(),
                    )
                })
            }),
        );
        assert!(lowerer.run(&mut h).unwrap());
        // list<usz>      -> read<usz>      -> usz just becomes list<qb> -> read<qb> -> qb
        // list<opt<usz>> -> read<opt<usz>> -> opt<usz> becomes list<qb> -> get<qb>  -> opt<qb>
        assert_eq!(
            h.root_type().dataflow_signature().unwrap().io(),
            (
                &vec![list_type(qb_t()); 2].into(),
                &vec![qb_t(), option_type(qb_t()).into()].into()
            )
        );
        assert_eq!(
            h.nodes()
                .filter_map(|n| h.get_optype(n).as_extension_op())
                .map(ExtensionOp::name)
                .sorted()
                .collect_vec(),
            ["NoBoundsCheck.read", "collections.list.get"]
        );
    }
}
