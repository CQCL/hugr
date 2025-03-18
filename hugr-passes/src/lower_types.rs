use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use thiserror::Error;

use hugr_core::extension::{ExtensionId, OpDef, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::ops::{
    AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock, ExtensionOp,
    FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpTrait, OpType, Output, Tag, TailLoop,
    Value, CFG, DFG,
};
use hugr_core::types::{CustomType, Transformable, Type, TypeArg, TypeTransformer};
use hugr_core::{Hugr, Node};

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

#[derive(Clone, Debug, PartialEq)]
pub enum OpReplacement {
    SingleOp(OpType),
    /// Defines a sub-Hugr to splice in place of the op.
    /// Note this will be of limited use before [monomorphization](super::monomorphize) because
    /// the sub-Hugrwill not be able to use type variables present in the op.
    // TODO: store also a vec<TypeParam>, and update Hugr::validate to take &[TypeParam]s
    // (defaulting to empty list) - see https://github.com/CQCL/hugr/issues/709
    CompoundOp(Box<Hugr>),
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

#[derive(Clone)]
pub struct LowerTypes {
    /// Handles simple cases like T1 -> T2.
    /// If T1 is Copyable and T2 Linear, then error will be raised if we find e.g.
    /// ArrayOfCopyables(T1). This would require an additional entry for that.
    type_map: HashMap<CustomType, Type>,
    /// Parametric types are handled by a function which receives the lowered typeargs.
    param_types: HashMap<ParametricType, Arc<dyn Fn(&[TypeArg]) -> Type>>,
    // Handles simple cases Op1 -> Op2.
    op_map: HashMap<OpHashWrapper, OpReplacement>,
    // Called after lowering typeargs; return None to use original OpDef
    param_ops: HashMap<ParametricOp, Arc<dyn Fn(&[TypeArg]) -> Option<OpReplacement>>>,
    // TODO should probably have a map, or two, here - from CustomType and from ParametricType.
    // Whereupon the closure should be given a callback to self.change_value, too, in case of nested
    // values for collections.
    const_fn: Arc<dyn Fn(&dyn CustomConst) -> Option<Value>>,
    check_sig: bool,
}

impl Default for LowerTypes {
    fn default() -> Self {
        Self {
            type_map: Default::default(),
            param_types: Default::default(),
            op_map: Default::default(),
            param_ops: Default::default(),
            const_fn: Arc::new(|_| None),
            check_sig: false,
        }
    }
}

impl TypeTransformer for LowerTypes {
    type Err = ChangeTypeError;

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
            Some(dest_fn(&nargs))
        } else {
            None
        })
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChangeTypeError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
}

impl LowerTypes {
    /// Configures this instance to change occurrences of `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* Type, this should only
    /// be used on *[monomorphize](super::monomorphize)d* Hugrs, because substitution
    /// (parametric polymorphism) happening later will not respect the lowering(s).
    pub fn lower_type(&mut self, src: CustomType, dest: Type) {
        // We could check that 'dest' is copyable or 'src' is linear, but since we can't
        // check that for parametric types, we'll be consistent and not check here either.
        self.type_map.insert(src, dest);
    }

    pub fn lower_parametric_type(
        &mut self,
        src: &TypeDef,
        dest_fn: Box<dyn Fn(&[TypeArg]) -> Type>,
    ) {
        // No way to check that dest_fn never produces a linear type.
        // We could require copy/discard-generators if src is Copyable, or *might be*
        // (depending on arguments - i.e. if src's TypeDefBound is anything other than
        // `TypeDefBound::Explicit(TypeBound::Copyable)`) but that seems an annoying
        // overapproximation.
        self.param_types.insert(src.into(), Arc::from(dest_fn));
    }

    pub fn lower_op(&mut self, src: &ExtensionOp, tgt: OpReplacement) {
        self.op_map.insert(OpHashWrapper::from(src), tgt);
    }

    pub fn lower_parametric_op(
        &mut self,
        src: &OpDef,
        dest_fn: Box<dyn Fn(&[TypeArg]) -> Option<OpReplacement>>,
    ) {
        self.param_ops.insert(src.into(), Arc::from(dest_fn));
    }

    pub fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            let expected_dfsig = if self.check_sig {
                let mut dfsig = hugr.get_optype(n).dataflow_signature().map(Cow::into_owned);
                if let Some(sig) = dfsig.as_mut() {
                    sig.transform(self)?;
                }
                Some(dfsig)
            } else {
                None
            };
            changed |= self.change_node(hugr, n)?;
            let new_dfsig = hugr.get_optype(n).dataflow_signature();
            // (If check_sig) then verify that the Signature still has the same arity/wires,
            // with only the expected changes to types within.
            if let Some(expected_sig) = expected_dfsig {
                assert_eq!(new_dfsig.as_ref().map(Cow::deref), expected_sig.as_ref());
            }
        }
        Ok(changed)
    }

    fn change_node(&self, hugr: &mut impl HugrMut, n: Node) -> Result<bool, ChangeTypeError> {
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
                        .instantiate(&type_args)
                        .map_err(ChangeTypeError::SignatureError)?;
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

    fn change_value(&self, value: &mut Value) -> Result<bool, ChangeTypeError> {
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
            Value::Extension { e } => {
                if let Some(new_const) = (self.const_fn)(e.value()) {
                    *value = new_const;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Value::Function { hugr } => self.run_no_validate(&mut **hugr),
        }
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, sync::Arc};

    use hugr_core::builder::{
        Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder,
        ModuleBuilder, SubContainer,
    };
    use hugr_core::extension::prelude::{bool_t, option_type, UnwrapBuilder};
    use hugr_core::extension::{TypeDefBound, Version};
    use hugr_core::ops::{ExtensionOp, OpType, Tag};
    use hugr_core::std_extensions::{
        arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES},
        collections::array::{array_type, ArrayOpDef},
    };
    use hugr_core::types::{PolyFuncType, Signature, Type, TypeArg, TypeBound};
    use hugr_core::{hugr::IdentList, type_row, Extension, HugrView};

    use crate::{MonomorphizePass, RemoveDeadFuncsPass};

    use super::{LowerTypes, OpReplacement};

    fn ext() -> Arc<Extension> {
        Extension::new_arc(
            IdentList::new("TestExt").unwrap(),
            Version::new(0, 0, 1),
            |ext, w| {
                let pv_of_var = ext
                    .add_type(
                        "PackedVec".into(),
                        vec![TypeBound::Any.into()],
                        String::new(),
                        TypeDefBound::from_params(vec![0]),
                        w,
                    )
                    .unwrap()
                    .instantiate(vec![Type::new_var_use(0, TypeBound::Copyable).into()])
                    .unwrap();
                ext.add_op(
                    "read".into(),
                    "".into(),
                    PolyFuncType::new(
                        vec![TypeBound::Copyable.into()],
                        Signature::new(
                            vec![pv_of_var.into(), INT_TYPES[6].to_owned()],
                            Type::new_var_use(0, TypeBound::Any),
                        ),
                    ),
                    w,
                )
                .unwrap();
                ext.add_op(
                    "lowered_read_bool".into(),
                    "".into(),
                    Signature::new(vec![INT_TYPES[6].to_owned(); 2], bool_t()),
                    w,
                )
                .unwrap();
            },
        )
    }

    fn lower_types(ext: &Extension) -> LowerTypes {
        fn lowered_read(args: &[TypeArg]) -> Option<OpReplacement> {
            let [TypeArg::Type { ty }] = args else {
                panic!("Illegal TypeArgs")
            };
            let mut dfb = DFGBuilder::new(Signature::new(
                vec![array_type(64, ty.clone()), INT_TYPES[6].to_owned()],
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
        let pv = ext.get_type("PackedVec").unwrap();
        let read = ext.get_op("read").unwrap();
        let mut lw = LowerTypes::default();
        lw.lower_type(
            pv.instantiate([bool_t().into()]).unwrap(),
            INT_TYPES[6].to_owned(),
        );
        lw.lower_parametric_type(
            pv,
            Box::new(|args: &[TypeArg]| {
                let [TypeArg::Type { ty }] = args else {
                    panic!("Illegal TypeArgs")
                };
                array_type(64, ty.clone())
            }),
        );
        lw.lower_op(
            &ExtensionOp::new(read.clone(), [bool_t().into()]).unwrap(),
            OpReplacement::SingleOp(
                ExtensionOp::new(ext.get_op("lowered_read_bool").unwrap().clone(), [])
                    .unwrap()
                    .into(),
            ),
        );
        lw.lower_parametric_op(read.as_ref(), Box::new(lowered_read));
        lw
    }

    #[test]
    fn module_func_dfg_cfg_call() {
        let ext = ext();
        let coln = ext.get_type("PackedVec").unwrap();

        let i64 = || INT_TYPES[6].to_owned();
        let c_int = Type::from(coln.instantiate([INT_TYPES[6].to_owned().into()]).unwrap());
        let c_bool = Type::from(coln.instantiate([bool_t().into()]).unwrap());
        let mut mb = ModuleBuilder::new();
        let read = {
            let read_op = ext.get_op("read").unwrap();
            let tv = Type::new_var_use(0, TypeBound::Copyable);
            let mut read_fn = mb
                .define_function(
                    "reader",
                    PolyFuncType::new(
                        [TypeBound::Copyable.into()],
                        Signature::new(
                            vec![coln.instantiate([tv.clone().into()]).unwrap().into(), i64()],
                            tv.clone(),
                        ),
                    ),
                )
                .unwrap();
            let res = read_fn
                .add_dataflow_op(
                    ExtensionOp::new(read_op.clone(), [tv.into()]).unwrap(),
                    read_fn.input_wires(),
                )
                .unwrap();
            read_fn.finish_with_outputs(res.outputs()).unwrap()
        };
        let mut fb = mb
            .define_function(
                "main",
                Signature::new(vec![i64(), c_int.clone(), c_bool.clone()], bool_t()),
            )
            .unwrap();
        let [idx, indices, bools] = fb.input_wires_arr();
        let mut dfb = fb
            .dfg_builder(Signature::new(vec![i64(), c_int], i64()), [idx, indices])
            .unwrap();
        let [idx, indices] = dfb.input_wires_arr();
        let int_read_op = dfb
            .call(read.handle(), &[i64().into()], [indices, idx])
            .unwrap();
        let [idx2] = dfb
            .finish_with_outputs(int_read_op.outputs())
            .unwrap()
            .outputs_arr();
        let mut cfg = fb
            .cfg_builder([(i64(), idx2), (c_bool, bools)], bool_t().into())
            .unwrap();
        let mut entry = cfg.entry_builder([bool_t().into()], type_row![]).unwrap();
        let [idx2, bools] = entry.input_wires_arr();
        let bool_read_op = entry
            .call(read.handle(), &[bool_t().into()], [bools, idx2])
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
        // Since we treat collection<bool> differently, we must monomorphize to catch all instantiations
        MonomorphizePass::default().run(&mut h).unwrap();
        RemoveDeadFuncsPass::default()
            .with_module_entry_points(h.children(h.root()).filter(|n| {
                h.get_optype(*n)
                    .as_func_defn()
                    .is_some_and(|fd| fd.name == "main")
            }))
            .run(&mut h)
            .unwrap();
        assert_eq!(
            h.nodes()
                .filter(|n| h.get_optype(*n).is_func_defn())
                .count(),
            3
        );

        assert!(lower_types(&ext).run_no_validate(&mut h).unwrap());
        h.validate().unwrap();

        let mut counts: HashMap<_, usize> = HashMap::new();
        for ext_op in h.nodes().filter_map(|n| h.get_optype(n).as_extension_op()) {
            *(counts.entry(ext_op.def().name().as_str()).or_default()) += 1;
        }
        assert_eq!(
            counts,
            HashMap::from([
                ("lowered_read_bool", 1),
                ("itousize", 1),
                ("get", 1),
                ("panic", 1)
            ])
        );
    }
}
