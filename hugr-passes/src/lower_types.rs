#![allow(clippy::type_complexity)]
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use hugr_core::builder::{BuildError, BuildHandle, Dataflow};
use hugr_core::ops::handle::DataflowOpID;
use itertools::Either;
use thiserror::Error;

use hugr_core::extension::{ExtensionId, OpDef, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{OpaqueValue, Sum};
use hugr_core::ops::{
    AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock, ExtensionOp,
    FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpTrait, OpType, Output, Tag, TailLoop,
    Value, CFG, DFG,
};
use hugr_core::types::{
    CustomType, Signature, Transformable, Type, TypeArg, TypeEnum, TypeTransformer,
};
use hugr_core::{Hugr, Node, Wire};

use crate::validation::{ValidatePassError, ValidationLevel};

mod linearize;
pub use linearize::{LinearizeError, Linearizer, copy_array, discard_array};

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
    fn add_hugr(self, hugr: &mut impl HugrMut, parent: Node) -> Node {
        match self {
            OpReplacement::SingleOp(op_type) => hugr.add_node_with_parent(parent, op_type),
            OpReplacement::CompoundOp(new_h) => hugr.insert_hugr(parent, *new_h).new_root,
        }
    }

    fn add(
        self,
        dfb: &mut impl Dataflow,
        inputs: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        match self {
            OpReplacement::SingleOp(opty) => dfb.add_dataflow_op(opty, inputs),
            OpReplacement::CompoundOp(h) => dfb.add_hugr_with_wires(*h, inputs),
        }
    }

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
pub struct LowerTypes {
    /// Handles simple cases like T1 -> T2.
    /// If T1 is Copyable and T2 Linear, then error will be raised if we find e.g.
    /// ArrayOfCopyables(T1). This would require an additional entry for that.
    type_map: HashMap<CustomType, Type>,
    /// Parametric types are handled by a function which receives the lowered typeargs.
    param_types: HashMap<ParametricType, Arc<dyn Fn(&[TypeArg]) -> Option<Type>>>,
    linearize: Linearizer,
    // Handles simple cases Op1 -> Op2.
    op_map: HashMap<OpHashWrapper, OpReplacement>,
    // Called after lowering typeargs; return None to use original OpDef
    param_ops: HashMap<ParametricOp, Arc<dyn Fn(&[TypeArg]) -> Option<OpReplacement>>>,
    consts: HashMap<Either<CustomType, ParametricType>, Arc<dyn Fn(&OpaqueValue) -> Option<Value>>>,
    check_sig: bool,
    validation: ValidationLevel,
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
            dest_fn(&nargs)
        } else {
            None
        })
    }
}

#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ChangeTypeError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    #[error("Lowering op {op} with original signature {old:?}\nExpected signature: {expected:?}\nBut got: {actual:?}")]
    SignatureMismatch {
        op: OpType,
        old: Option<Signature>,
        expected: Option<Signature>,
        actual: Option<Signature>,
    },
    #[error(transparent)]
    ValidationError(#[from] ValidatePassError),
    #[error(transparent)]
    LinearizeError(#[from] LinearizeError),
}

impl LowerTypes {
    /// Sets the validation level used before and after the pass is run.
    // Note the self -> Self style is consistent with other passes, but not the other methods here.
    // TODO change the others? But we are planning to drop validation_level in https://github.com/CQCL/hugr/pull/1895
    pub fn validation_level(mut self, level: ValidationLevel) -> Self {
        self.validation = level;
        self
    }

    /// Configures this instance to change occurrences of `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* Type, this should only
    /// be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as substitution
    /// (parametric polymorphism) happening later will not respect the lowering(s).
    ///
    /// This takes precedence over [Self::lower_parametric_type] where the `src`s overlap.
    pub fn lower_type(&mut self, src: CustomType, dest: Type) {
        // We could check that 'dest' is copyable or 'src' is linear, but since we can't
        // check that for parametric types, we'll be consistent and not check here either.
        self.type_map.insert(src, dest);
    }

    /// Configures this instance to change occurrences of a parametrized type `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been lowered (e.g. they may not
    /// fit the bounds of the original type).
    pub fn lower_parametric_type(
        &mut self,
        src: &TypeDef,
        dest_fn: Box<dyn Fn(&[TypeArg]) -> Option<Type>>,
    ) {
        // No way to check that dest_fn never produces a linear type.
        // We could require copy/discard-generators if src is Copyable, or *might be*
        // (depending on arguments - i.e. if src's TypeDefBound is anything other than
        // `TypeDefBound::Explicit(TypeBound::Copyable)`) but that seems an annoying
        // overapproximation. Moreover, these depend upon the *return type* of the Fn.
        // We could take an
        // `dyn Fn(&TypeArg) -> (Type, Fn(&Linearizer) -> OpReplacement, Fn(&Linearizer) -> OpReplacement))`
        // but that seems too awkward.
        self.param_types.insert(src.into(), Arc::from(dest_fn));
    }

    /// Configures this instance that, when an outport of type `src` has other than one connected
    /// inport, the specified `copy` and or `discard` ops should be used to wire it to those inports.
    /// (`copy` should have exactly one inport, of type `src`, and two outports, of same type;
    /// `discard` should have exactly one inport, of type 'src', and no outports.)
    ///
    /// To clarify, these are used if `src` is not [Copyable], but is (perhaps contained in) the
    /// result of lonering a type that was either copied or discarded in the input Hugr.
    ///
    /// [Copyable]: hugr_core::types::TypeBound::Copyable
    pub fn linearize(&mut self, src: Type, copy: OpReplacement, discard: OpReplacement) {
        // We could raise an error if src's bound is Copyable?
        self.linearize.register(src, copy, discard)
    }

    /// Configures this instance that when lowering produces an outport which
    /// * has type an instantiation of the parametric type `src`, and
    /// * is not [Copyable](hugr_core::types::TypeBound::Copyable), and
    /// * has other than one connected inport,
    ///
    /// ...then these functions should be used to generate `copy` or `discard` ops.
    ///
    /// (That is, this is the equivalent of [Self::linearize] but for parametric types.)
    ///
    /// The [Linearizer] is passed so that the callbacks can use this to generate
    /// `copy/`discard` ops for other types (e.g. the elements of a collection),
    /// as part of an [OpReplacement::CompoundOp].
    pub fn linearize_parametric(
        &mut self,
        src: &TypeDef,
        copy_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
        discard_fn: Box<dyn Fn(&[TypeArg], &Linearizer) -> Result<OpReplacement, LinearizeError>>,
    ) {
        // We could raise an error if src's TypeDefBound is explicit Copyable ?
        self.linearize.register_parametric(src, copy_fn, discard_fn)
    }

    /// Configures this instance to change occurrences of `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* [OpDef], this should only
    /// be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as substitution
    /// (parametric polymorphism) happening later will not respect the lowering(s).
    ///
    /// This takes precedence over [Self::lower_parametric_op] where the `src`s overlap.
    pub fn lower_op(&mut self, src: &ExtensionOp, dest: OpReplacement) {
        self.op_map.insert(OpHashWrapper::from(src), dest);
    }

    /// Configures this instance to change occurrences of a parametrized op `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been lowered (e.g. they may not
    /// fit the bounds of the original op).
    ///
    /// If the Callback returns None, the new typeargs will be applied to the original op.
    pub fn lower_parametric_op(
        &mut self,
        src: &OpDef,
        dest_fn: Box<dyn Fn(&[TypeArg]) -> Option<OpReplacement>>,
    ) {
        self.param_ops.insert(src.into(), Arc::from(dest_fn));
    }

    /// Configures this instance to change occurrences consts of type `src_ty`, using
    /// a callback given the value of the constant (of that type). (The callback may
    /// return `None` to indicate nothing has changed; we assume `Some` means something
    /// has changed when evaluating the `bool` result of [Self::run].)
    ///
    /// Note that if `src_ty` is an instance of a *parametrized* [TypeDef], this
    /// takes precedence over [Self::lower_consts_parametric] where the `src_ty`s overlap.
    pub fn lower_consts(
        &mut self,
        src_ty: &CustomType,
        const_fn: Box<dyn Fn(&OpaqueValue) -> Option<Value>>,
    ) {
        self.consts
            .insert(Either::Left(src_ty.clone()), Arc::from(const_fn));
    }

    /// Configures this instance to change occurrences consts of all types that
    /// are instances of a parametric typedef `src_ty`, using a callback given
    /// the value of the constant (the [OpaqueValue] contains the [TypeArg]s).
    pub fn lower_consts_parametric(
        &mut self,
        src_ty: &TypeDef,
        const_fn: Box<dyn Fn(&OpaqueValue) -> Option<Value>>,
    ) {
        self.consts
            .insert(Either::Right(src_ty.into()), Arc::from(const_fn));
    }

    /// Configures this instance to check signatures of ops lowered following [Self::lower_op]
    /// and [Self::lower_parametric_op] are as expected, i.e. match the signatures of the
    /// original op modulo the required type substitutions. (If signatures are incorrect,
    /// it is likely that the wires in the Hugr will be invalid, so this gives an early warning
    /// by instead raising [ChangeTypeError::SignatureMismatch].)
    pub fn check_signatures(&mut self, check_sig: bool) {
        self.check_sig = check_sig;
    }

    /// Run the pass using specified configuration.
    pub fn run<H: HugrMut>(&self, hugr: &mut H) -> Result<bool, ChangeTypeError> {
        self.validation
            .run_validated_pass(hugr, |hugr: &mut H, _| self.run_no_validate(hugr))
    }

    fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            let maybe_check_sig = if self.check_sig {
                Some(
                    if let Some(old_sig) = hugr.get_optype(n).dataflow_signature() {
                        let old_sig = old_sig.into_owned();
                        let mut expected_sig = old_sig.clone();
                        expected_sig.transform(self)?;
                        Some((old_sig, expected_sig))
                    } else {
                        None
                    },
                )
            } else {
                None
            };
            changed |= self.change_node(hugr, n)?;
            let new_dfsig = hugr.get_optype(n).dataflow_signature();
            // (If check_sig) then verify that the Signature still has the same arity/wires,
            // with only the expected changes to types within.
            if let Some(old_and_expected) = maybe_check_sig {
                match (&old_and_expected, &new_dfsig) {
                    (None, None) => (),
                    (Some((_, exp)), Some(act))
                        if exp.input == act.input && exp.output == act.output => {}
                    _ => {
                        let (old, expected) = old_and_expected.unzip();
                        return Err(ChangeTypeError::SignatureMismatch {
                            op: hugr.get_optype(n).clone(),
                            old,
                            expected,
                            actual: new_dfsig.map(Cow::into_owned),
                        });
                    }
                };
            }
            if let Some(new_sig) = (changed && n != hugr.root())
                .then_some(new_dfsig)
                .flatten()
                .map(Cow::into_owned)
            {
                for outp in new_sig.output_ports() {
                    if !new_sig.out_port_type(outp).unwrap().copyable() {
                        let targets = hugr.linked_inputs(n, outp).collect::<Vec<_>>();
                        if targets.len() != 1 {
                            hugr.disconnect(n, outp);
                            let typ = new_sig.out_port_type(outp).unwrap();
                            self.linearize
                                .insert_copy_discard(hugr, n, outp, typ, &targets)?;
                        }
                    }
                }
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
                        .instantiate(type_args)
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
            Value::Extension { e } => Ok('changed: {
                if let TypeEnum::Extension(exty) = e.get_type().as_type_enum() {
                    if let Some(const_fn) = self
                        .consts
                        .get(&Either::Left(exty.clone()))
                        .or(self.consts.get(&Either::Right(exty.into())))
                    {
                        if let Some(new_const) = const_fn(e) {
                            *value = new_const;
                            break 'changed true;
                        }
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
    use hugr_core::extension::prelude::{bool_t, option_type, usize_t, ConstUsize, UnwrapBuilder};
    use hugr_core::extension::simple_op::MakeExtensionOp;
    use hugr_core::extension::{TypeDefBound, Version};

    use hugr_core::ops::{ExtensionOp, OpType, Tag, Value};

    use hugr_core::std_extensions::arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES};
    use hugr_core::std_extensions::collections::array::{
        array_type, ArrayOp, ArrayOpDef, ArrayValue,
    };
    use hugr_core::std_extensions::collections::list::{list_type, list_type_def, ListValue};
    use hugr_core::types::{PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeRow};
    use hugr_core::{hugr::IdentList, type_row, Extension, HugrView};
    use itertools::Itertools;

    use super::{LowerTypes, OpReplacement};

    const PACKED_VEC: &str = "PackedVec";
    fn i64_t() -> Type {
        INT_TYPES[6].clone()
    }

    fn read_op(ext: &Arc<Extension>, t: Type) -> ExtensionOp {
        ExtensionOp::new(ext.get_op("read").unwrap().clone(), [t.into()]).unwrap()
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
                    "read".into(),
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

    fn lowerer(ext: &Arc<Extension>) -> LowerTypes {
        fn lowered_read(args: &[TypeArg]) -> Option<OpReplacement> {
            let [TypeArg::Type { ty }] = args else {
                panic!("Illegal TypeArgs")
            };
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
        let mut lw = LowerTypes::default();
        lw.lower_type(pv.instantiate([bool_t().into()]).unwrap(), i64_t());
        lw.lower_parametric_type(
            pv,
            Box::new(|args: &[TypeArg]| {
                let [TypeArg::Type { ty }] = args else {
                    panic!("Illegal TypeArgs")
                };
                Some(array_type(64, ty.clone()))
            }),
        );
        lw.lower_op(
            &read_op(ext, bool_t()),
            OpReplacement::SingleOp(
                ExtensionOp::new(ext.get_op("lowered_read_bool").unwrap().clone(), [])
                    .unwrap()
                    .into(),
            ),
        );
        lw.lower_parametric_op(ext.get_op("read").unwrap().as_ref(), Box::new(lowered_read));
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
        let mut lowerer = LowerTypes::default();
        lowerer.lower_parametric_type(
            list_type_def(),
            Box::new(|args| {
                let [TypeArg::Type { ty }] = args else {
                    panic!("Expected elem type")
                };
                (![usize_t(), bool_t()].contains(ty)).then_some(array_type(10, ty.clone()))
            }),
        );
        let backup = h.clone();
        assert!(!lowerer.run(&mut h).unwrap());
        assert_eq!(h, backup);

        //2. Lower List<T> to Array<10, T> UNLESS T is usize_t() - this leaves the Const unchanged
        let mut lowerer = LowerTypes::default();
        lowerer.lower_parametric_type(
            list_type_def(),
            Box::new(|args| {
                let [TypeArg::Type { ty }] = args else {
                    panic!("Expected elem type")
                };
                (usize_t() != *ty).then_some(array_type(10, ty.clone()))
            }),
        );
        assert!(lowerer.run(&mut h).unwrap());
        let sig = h.signature(h.root()).unwrap();
        assert_eq!(
            sig.input(),
            &TypeRow::from(vec![list_type(usize_t()), array_type(10, bool_t())])
        );
        assert_eq!(sig.input(), sig.output());

        // 3. Lower all List<T> to Array<4,T> so we can use List's handy CustomConst
        let mut h = backup;
        let mut lowerer = LowerTypes::default();
        lowerer.lower_parametric_type(
            list_type_def(),
            Box::new(|args: &[TypeArg]| {
                let [TypeArg::Type { ty }] = args else {
                    panic!("Expected elem type")
                };
                Some(array_type(4, ty.clone()))
            }),
        );
        lowerer.lower_consts_parametric(
            list_type_def(),
            Box::new(|opaq| {
                let lv = opaq
                    .value()
                    .downcast_ref::<ListValue>()
                    .expect("Only one constant in test");
                Some(
                    ArrayValue::new(lv.get_element_type().clone(), lv.get_contents().to_vec())
                        .into(),
                )
            }),
        );
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
}
