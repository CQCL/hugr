#![allow(clippy::type_complexity)]
#![warn(missing_docs)]
//! Replace types with other types across the Hugr. See [ReplaceTypes] and [Linearizer].
//!
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use hugr_core::builder::{BuildError, BuildHandle, Dataflow};
use hugr_core::extension::{ExtensionId, OpDef, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{OpaqueValue, Sum};
use hugr_core::ops::handle::DataflowOpID;
use hugr_core::ops::{
    AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock, ExtensionOp,
    FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpTrait, OpType, Output, Tag, TailLoop,
    Value, CFG, DFG,
};
use hugr_core::types::{
    CustomType, Signature, Transformable, Type, TypeArg, TypeEnum, TypeTransformer,
};
use hugr_core::{Hugr, HugrView, Node, Wire};

use crate::ComposablePass;

mod linearize;
pub use linearize::{CallbackHandler, DelegatingLinearizer, LinearizeError, Linearizer};

/// A recipe for creating a dataflow Node - as a new child of a [DataflowParent]
/// or in order to replace an existing node.
///
/// [DataflowParent]: hugr_core::ops::OpTag::DataflowParent
#[derive(Clone, Debug, PartialEq)]
pub enum NodeTemplate {
    /// A single node - so if replacing an existing node, change only the op
    SingleOp(OpType),
    /// Defines a sub-Hugr to insert, whose root becomes (or replaces) the desired Node.
    /// The root must be a [CFG], [Conditional], [DFG] or [TailLoop].
    // Not a FuncDefn, nor Case/DataflowBlock
    /// Note this will be of limited use before [monomorphization](super::monomorphize())
    /// because the new subtree will not be able to use type variables present in the
    /// parent Hugr or previous op.
    // TODO: store also a vec<TypeParam>, and update Hugr::validate to take &[TypeParam]s
    // (defaulting to empty list) - see https://github.com/CQCL/hugr/issues/709
    CompoundOp(Box<Hugr>),
    // TODO allow also Call to a Node in the existing Hugr
    // (can't see any other way to achieve multiple calls to the same decl.
    // So client should add the functions before replacement, then remove unused ones afterwards.)
}

impl NodeTemplate {
    /// Adds this instance to the specified [HugrMut] as a new node or subtree under a
    /// given parent, returning the unique new child (of that parent) thus created
    pub fn add_hugr(self, hugr: &mut impl HugrMut, parent: Node) -> Node {
        match self {
            NodeTemplate::SingleOp(op_type) => hugr.add_node_with_parent(parent, op_type),
            NodeTemplate::CompoundOp(new_h) => hugr.insert_hugr(parent, *new_h).new_root,
        }
    }

    /// Adds this instance to the specified [Dataflow] builder as a new node or subtree
    pub fn add(
        self,
        dfb: &mut impl Dataflow,
        inputs: impl IntoIterator<Item = Wire>,
    ) -> Result<BuildHandle<DataflowOpID>, BuildError> {
        match self {
            NodeTemplate::SingleOp(opty) => dfb.add_dataflow_op(opty, inputs),
            NodeTemplate::CompoundOp(h) => dfb.add_hugr_with_wires(*h, inputs),
        }
    }

    fn replace(&self, hugr: &mut impl HugrMut, n: Node) {
        assert_eq!(hugr.children(n).count(), 0);
        let new_optype = match self.clone() {
            NodeTemplate::SingleOp(op_type) => op_type,
            NodeTemplate::CompoundOp(new_h) => {
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

    fn signature(&self) -> Option<Cow<'_, Signature>> {
        match self {
            NodeTemplate::SingleOp(op_type) => op_type,
            NodeTemplate::CompoundOp(hugr) => hugr.root_type(),
        }
        .dataflow_signature()
    }
}

/// A configuration of what types, ops, and constants should be replaced with what.
/// May be applied to a Hugr via [Self::run].
///
/// Parametrized types and ops will be reparametrized taking into account the
/// replacements, but any ops taking/returning the replaced types *not* as a result of
/// parametrization, will also need to be replaced - see [Self::replace_op].
/// Similarly [Const]s.
///
/// Types that are [Copyable](hugr_core::types::TypeBound::Copyable) may also be replaced
/// with types that are not, see [Linearizer].
///
/// Note that although this pass may be used before [monomorphization], there are some
/// limitations (that do not apply if done after [monomorphization]):
/// * [NodeTemplate::CompoundOp] only works for operations that do not use type variables
/// * "Overrides" of specific instantiations of polymorphic types will not be detected if
///   the instantiations are created inside polymorphic functions. For example, suppose
///   we [Self::replace_type] type `A` with `X`, [Self::replace_parametrized_type]
///   container `MyList` with `List`, and [Self::replace_type] `MyList<A>` with
///   `SpecialListOfXs`. If a function `foo` polymorphic over a type variable `T` dealing
///   with `MyList<T>`s, that is called with type argument `A`, then `foo<T>` will be
///   updated to deal with `List<T>`s and the call `foo<A>` updated to `foo<X>`, but this
///   will still result in using `List<X>` rather than `SpecialListOfXs`. (However this
///   would be fine *after* [monomorphization]: the monomorphic definition of `foo_A`
///   would use `SpecialListOfXs`.)
/// * See also limitations noted for [Linearizer].
///
/// [monomorphization]: super::monomorphize()
#[derive(Clone, Default)]
pub struct ReplaceTypes {
    type_map: HashMap<CustomType, Type>,
    param_types: HashMap<ParametricType, Arc<dyn Fn(&[TypeArg]) -> Option<Type>>>,
    linearize: DelegatingLinearizer,
    op_map: HashMap<OpHashWrapper, NodeTemplate>,
    param_ops: HashMap<ParametricOp, Arc<dyn Fn(&[TypeArg]) -> Option<NodeTemplate>>>,
    consts: HashMap<
        CustomType,
        Arc<dyn Fn(&OpaqueValue, &ReplaceTypes) -> Result<Value, ReplaceTypesError>>,
    >,
    param_consts: HashMap<
        ParametricType,
        Arc<dyn Fn(&OpaqueValue, &ReplaceTypes) -> Result<Option<Value>, ReplaceTypesError>>,
    >,
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

/// An error produced by the [ReplaceTypes] pass
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ReplaceTypesError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    #[error(transparent)]
    LinearizeError(#[from] LinearizeError),
}

impl ReplaceTypes {
    /// Configures this instance to replace occurrences of type `src` with `dest`.
    /// Note that if `src` is an instance of a *parametrized* [TypeDef], this takes
    /// precedence over [Self::replace_parametrized_type] where the `src`s overlap. Thus, this
    /// should only be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as
    /// substitution (parametric polymorphism) happening later will not respect this replacement.
    ///
    /// If there are any [LoadConstant]s of this type, callers should also call [Self::replace_consts]
    /// (or [Self::replace_consts_parametrized]) as the [LoadConstant]s will be reparametrized
    /// (and this will break the edge from [Const] to [LoadConstant]).
    ///
    /// Note that if `src` is Copyable and `dest` is Linear, then (besides linearity violations)
    /// [SignatureError] will be raised if this leads to an impossible type e.g. ArrayOfCopyables(src).
    /// (This can be overridden by an additional [Self::replace_type].)
    pub fn replace_type(&mut self, src: CustomType, dest: Type) {
        // We could check that 'dest' is copyable or 'src' is linear, but since we can't
        // check that for parametrized types, we'll be consistent and not check here either.
        self.type_map.insert(src, dest);
    }

    /// Configures this instance to change occurrences of a parametrized type `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been updated (e.g. they may not
    /// fit the bounds of the original type). The callback may return `None` to indicate
    /// no change (in which case the supplied TypeArgs will be given to `src`).
    ///
    /// If there are any [LoadConstant]s of any of these types, callers should also call
    /// [Self::replace_consts_parametrized] (or [Self::replace_consts]) as the
    /// [LoadConstant]s will be reparametrized (and this will break the edge from [Const] to
    /// [LoadConstant]).
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
        // It would be too awkward to require:
        // dest_fn: impl Fn(&TypeArg) -> (Type,
        //                                Fn(&Linearizer) -> NodeTemplate, // copy
        //                                Fn(&Linearizer) -> NodeTemplate)` // discard
        self.param_types.insert(src.into(), Arc::new(dest_fn));
    }

    /// Allows to configure how to deal with types/wires that were [Copyable]
    /// but have become linear as a result of type-changing. Specifically,
    /// the [Linearizer] is used whenever lowering produces an outport which both
    /// * has a non-[Copyable] type - perhaps a direct substitution, or perhaps e.g.
    ///   as a result of changing the element type of a collection such as an [`array`]
    /// * has other than one connected inport,
    ///
    /// [Copyable]: hugr_core::types::TypeBound::Copyable
    /// [`array`]: hugr_core::std_extensions::collections::array::array_type
    pub fn linearizer(&mut self) -> &mut DelegatingLinearizer {
        &mut self.linearize
    }

    /// Configures this instance to change occurrences of `src` to `dest`.
    /// Note that if `src` is an instance of a *parametrized* [OpDef], this takes
    /// precedence over [Self::replace_parametrized_op] where the `src`s overlap. Thus,
    /// this should only be used on already-*[monomorphize](super::monomorphize())d*
    /// Hugrs, as substitution (parametric polymorphism) happening later will not respect
    /// this replacement.
    pub fn replace_op(&mut self, src: &ExtensionOp, dest: NodeTemplate) {
        self.op_map.insert(OpHashWrapper::from(src), dest);
    }

    /// Configures this instance to change occurrences of a parametrized op `src`
    /// via a callback that builds the replacement type given the [TypeArg]s.
    /// Note that the TypeArgs will already have been updated (e.g. they may not
    /// fit the bounds of the original op).
    ///
    /// If the Callback returns None, the new typeargs will be applied to the original op.
    pub fn replace_parametrized_op(
        &mut self,
        src: &OpDef,
        dest_fn: impl Fn(&[TypeArg]) -> Option<NodeTemplate> + 'static,
    ) {
        self.param_ops.insert(src.into(), Arc::new(dest_fn));
    }

    /// Configures this instance to change [Const]s of type `src_ty`, using
    /// a callback that is passed the value of the constant (of that type).
    ///
    /// Note that if `src_ty` is an instance of a *parametrized* [TypeDef],
    /// this takes precedence over [Self::replace_consts_parametrized] where
    /// the `src_ty`s overlap.
    pub fn replace_consts(
        &mut self,
        src_ty: CustomType,
        const_fn: impl Fn(&OpaqueValue, &ReplaceTypes) -> Result<Value, ReplaceTypesError> + 'static,
    ) {
        self.consts.insert(src_ty, Arc::new(const_fn));
    }

    /// Configures this instance to change [Const]s of all types that are instances
    /// of a parametrized typedef `src_ty`, using a callback that is passed the
    /// value of the constant (the [OpaqueValue] contains the [TypeArg]s). The
    /// callback may return `None` to indicate no change to the constant.
    pub fn replace_consts_parametrized(
        &mut self,
        src_ty: &TypeDef,
        const_fn: impl Fn(&OpaqueValue, &ReplaceTypes) -> Result<Option<Value>, ReplaceTypesError>
            + 'static,
    ) {
        self.param_consts.insert(src_ty.into(), Arc::new(const_fn));
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

    /// Modifies the specified Value in-place according to current configuration.
    /// Returns whether the value has changed (conservative over-approximation).
    pub fn change_value(&self, value: &mut Value) -> Result<bool, ReplaceTypesError> {
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
            Value::Extension { e } => Ok({
                let new_const = match e.get_type().as_type_enum() {
                    TypeEnum::Extension(exty) => match self.consts.get(exty) {
                        Some(const_fn) => Some(const_fn(e, self)),
                        None => self
                            .param_consts
                            .get(&exty.into())
                            .and_then(|const_fn| const_fn(e, self).transpose()),
                    },
                    _ => None,
                };
                if let Some(new_const) = new_const {
                    *value = new_const?;
                    true
                } else {
                    false
                }
            }),
            Value::Function { hugr } => self.run(&mut **hugr),
        }
    }
}

impl ComposablePass for ReplaceTypes {
    type Error = ReplaceTypesError;
    type Result = bool;

    fn run(&self, hugr: &mut impl HugrMut) -> Result<bool, ReplaceTypesError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            changed |= self.change_node(hugr, n)?;
            let new_dfsig = hugr.get_optype(n).dataflow_signature();
            if let Some(new_sig) = new_dfsig
                .filter(|_| changed && n != hugr.root())
                .map(Cow::into_owned)
            {
                for outp in new_sig.output_ports() {
                    if !new_sig.out_port_type(outp).unwrap().copyable() {
                        let targets = hugr.linked_inputs(n, outp).collect::<Vec<_>>();
                        if targets.len() != 1 {
                            hugr.disconnect(n, outp);
                            let src = Wire::new(n, outp);
                            self.linearize.insert_copy_discard(hugr, src, &targets)?;
                        }
                    }
                }
            }
        }
        Ok(changed)
    }
}
pub mod handlers {
    //! Callbacks for use with [ReplaceTypes::replace_consts_parametrized]
    use hugr_core::ops::{constant::OpaqueValue, Value};
    use hugr_core::std_extensions::collections::list::ListValue;
    use hugr_core::types::Transformable;

    use super::{ReplaceTypes, ReplaceTypesError};

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
    use hugr_core::std_extensions::arithmetic::int_types::ConstInt;
    use hugr_core::std_extensions::arithmetic::{conversions::ConvertOpDef, int_types::INT_TYPES};
    use hugr_core::std_extensions::collections::array::{
        array_type, ArrayOp, ArrayOpDef, ArrayValue,
    };
    use hugr_core::std_extensions::collections::list::{
        list_type, list_type_def, ListOp, ListValue,
    };

    use hugr_core::types::{PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeRow};
    use hugr_core::{hugr::IdentList, type_row, Extension, HugrView};
    use itertools::Itertools;

    use crate::ComposablePass;

    use super::{handlers::list_const, NodeTemplate, ReplaceTypes};

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
        fn lowered_read(args: &[TypeArg]) -> Option<NodeTemplate> {
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
            Some(NodeTemplate::CompoundOp(Box::new(
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
            NodeTemplate::SingleOp(
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
        let backup = tl.finish_hugr().unwrap();

        let mut lowerer = ReplaceTypes::default();
        // Recursively descend into lists
        lowerer.replace_consts_parametrized(list_type_def(), list_const);

        // 1. Lower List<T> to Array<10, T> UNLESS T is usize_t() or i64_t
        lowerer.replace_parametrized_type(list_type_def(), |args| {
            let ty = just_elem_type(args);
            (![usize_t(), i64_t()].contains(ty)).then_some(array_type(10, ty.clone()))
        });
        {
            let mut h = backup.clone();
            assert_eq!(lowerer.run(&mut h), Ok(true));
            let sig = h.signature(h.root()).unwrap();
            assert_eq!(
                sig.input(),
                &TypeRow::from(vec![list_type(usize_t()), array_type(10, bool_t())])
            );
            assert_eq!(sig.input(), sig.output());
        }

        // 2. Now we'll also change usize's to i64_t's
        let usize_custom_t = usize_t().as_extension().unwrap().clone();
        lowerer.replace_type(usize_custom_t.clone(), i64_t());
        lowerer.replace_consts(usize_custom_t, |opaq, _| {
            Ok(ConstInt::new_u(
                6,
                opaq.value().downcast_ref::<ConstUsize>().unwrap().value(),
            )
            .unwrap()
            .into())
        });
        {
            let mut h = backup.clone();
            assert_eq!(lowerer.run(&mut h), Ok(true));
            let sig = h.signature(h.root()).unwrap();
            assert_eq!(
                sig.input(),
                &TypeRow::from(vec![list_type(i64_t()), array_type(10, bool_t())])
            );
            assert_eq!(sig.input(), sig.output());
            // This will have to update inside the Const
            let cst = h
                .nodes()
                .filter_map(|n| h.get_optype(n).as_const())
                .exactly_one()
                .ok()
                .unwrap();
            assert_eq!(cst.get_type(), Type::new_sum(vec![list_type(i64_t()); 2]));
        }

        // 3. Lower all List<T> to Array<4,T>
        let mut h = backup;
        lowerer.replace_parametrized_type(
            list_type_def(),
            Box::new(|args: &[TypeArg]| Some(array_type(4, just_elem_type(args).clone()))),
        );
        lowerer.replace_consts_parametrized(list_type_def(), |opaq, repl| {
            // First recursively transform the contents
            let Some(Value::Extension { e: opaq }) = list_const(opaq, repl)? else {
                panic!("Expected list value to stay a list value");
            };
            let lv = opaq.value().downcast_ref::<ListValue>().unwrap();

            Ok(Some(
                ArrayValue::new(lv.get_element_type().clone(), lv.get_contents().to_vec()).into(),
            ))
        });
        lowerer.run(&mut h).unwrap();

        assert_eq!(
            h.get_optype(pred.node())
                .as_load_constant()
                .map(|lc| lc.constant_type()),
            Some(&Type::new_sum(vec![Type::from(array_type(4, i64_t())); 2]))
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
        let i32_custom_t = i32_t().as_extension().unwrap().clone();
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
                    NodeTemplate::SingleOp(
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
