#![allow(clippy::type_complexity)]
//! Replace types with other types across the Hugr. See [`ReplaceTypes`] and [Linearizer].
//!
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use handlers::list_const;
use hugr_core::std_extensions::collections::array::array_type_def;
use hugr_core::std_extensions::collections::list::list_type_def;
use hugr_core::std_extensions::collections::value_array::value_array_type_def;
use thiserror::Error;

use hugr_core::builder::{BuildError, BuildHandle, Dataflow};
use hugr_core::extension::{ExtensionId, OpDef, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{OpaqueValue, Sum};
use hugr_core::ops::handle::{DataflowOpID, FuncID};
use hugr_core::ops::{
    AliasDefn, CFG, Call, CallIndirect, Case, Conditional, Const, DFG, DataflowBlock, ExitBlock,
    ExtensionOp, Input, LoadConstant, LoadFunction, OpTrait, OpType, Output, Tag, TailLoop, Value,
};
use hugr_core::types::{
    ConstTypeError, CustomType, Signature, Transformable, Type, TypeArg, TypeEnum, TypeRow,
    TypeTransformer,
};
use hugr_core::{Direction, Hugr, HugrView, Node, PortIndex, Wire};

use crate::ComposablePass;

mod linearize;
pub use linearize::{CallbackHandler, DelegatingLinearizer, LinearizeError, Linearizer};

/// A recipe for creating a dataflow Node - as a new child of a [`DataflowParent`]
/// or in order to replace an existing node.
///
/// [`DataflowParent`]: hugr_core::ops::OpTag::DataflowParent
#[derive(Clone, Debug, PartialEq)]
pub enum NodeTemplate {
    /// A single node - so if replacing an existing node, change only the op
    SingleOp(OpType),
    /// Defines a sub-Hugr to insert, whose root becomes (or replaces) the desired Node.
    /// The root must be a [CFG], [Conditional], [DFG] or [`TailLoop`].
    // Not a FuncDefn, nor Case/DataflowBlock
    /// Note this will be of limited use before [monomorphization](super::monomorphize())
    /// because the new subtree will not be able to use type variables present in the
    /// parent Hugr or previous op.
    CompoundOp(Box<Hugr>),
    /// A Call to an existing function.
    Call(Node, Vec<TypeArg>),
}

impl NodeTemplate {
    /// Adds this instance to the specified [`HugrMut`] as a new node or subtree under a
    /// given parent, returning the unique new child (of that parent) thus created
    ///
    /// # Panics
    ///
    /// * If `parent` is not in the `hugr`
    ///
    /// # Errors
    ///
    /// * If `self` is a [`Self::Call`] and the target Node either
    ///    * is neither a [`FuncDefn`] nor a [`FuncDecl`]
    ///    * has a [`signature`] which the type-args of the [`Self::Call`] do not match
    ///
    /// [`signature`]: hugr_core::types::PolyFuncType
    /// [`FuncDecl`]: hugr_core::ops::FuncDecl
    /// [`FuncDefn`]: hugr_core::ops::FuncDefn
    pub fn add_hugr(
        self,
        hugr: &mut impl HugrMut<Node = Node>,
        parent: Node,
    ) -> Result<Node, BuildError> {
        match self {
            NodeTemplate::SingleOp(op_type) => Ok(hugr.add_node_with_parent(parent, op_type)),
            NodeTemplate::CompoundOp(new_h) => {
                Ok(hugr.insert_hugr(parent, *new_h).inserted_entrypoint)
            }
            NodeTemplate::Call(target, type_args) => {
                let c = call(hugr, target, type_args)?;
                let tgt_port = c.called_function_port();
                let n = hugr.add_node_with_parent(parent, c);
                hugr.connect(target, 0, n, tgt_port);
                Ok(n)
            }
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
            // Really we should check whether func points at a FuncDecl or FuncDefn and create
            // the appropriate variety of FuncID but it doesn't matter for the purpose of making a Call.
            NodeTemplate::Call(func, type_args) => {
                if !dfb.hugr().contains_node(func) {
                    return Err(BuildError::NodeNotFound { node: func });
                }
                dfb.call(&FuncID::<true>::from(func), &type_args, inputs)
            }
        }
    }

    fn replace(self, hugr: &mut impl HugrMut<Node = Node>, n: Node) -> Result<(), BuildError> {
        assert_eq!(hugr.children(n).count(), 0);
        let new_optype = match self {
            NodeTemplate::SingleOp(op_type) => op_type,
            NodeTemplate::CompoundOp(new_h) => {
                let new_entrypoint = hugr.insert_hugr(n, *new_h).inserted_entrypoint;
                let children = hugr.children(new_entrypoint).collect::<Vec<_>>();
                let root_opty = hugr.remove_node(new_entrypoint);
                for ch in children {
                    hugr.set_parent(ch, n);
                }
                root_opty
            }
            NodeTemplate::Call(func, type_args) => {
                let c = call(hugr, func, type_args)?;
                let static_inport = c.called_function_port();
                // insert an input for the Call static input
                hugr.insert_ports(n, Direction::Incoming, static_inport.index(), 1);
                // connect the function to (what will be) the call
                hugr.connect(func, 0, n, static_inport);
                c.into()
            }
        };
        *hugr.optype_mut(n) = new_optype;
        Ok(())
    }

    fn check_signature(
        &self,
        inputs: &TypeRow,
        outputs: &TypeRow,
    ) -> Result<(), Option<Signature>> {
        let sig = match self {
            NodeTemplate::SingleOp(op_type) => op_type,
            NodeTemplate::CompoundOp(hugr) => hugr.entrypoint_optype(),
            NodeTemplate::Call(_, _) => return Ok(()), // no way to tell
        }
        .dataflow_signature();
        if sig.as_deref().map(Signature::io) == Some((inputs, outputs)) {
            Ok(())
        } else {
            Err(sig.map(Cow::into_owned))
        }
    }
}

fn call<H: HugrView<Node = Node>>(
    h: &H,
    func: Node,
    type_args: Vec<TypeArg>,
) -> Result<Call, BuildError> {
    let func_sig = match h.get_optype(func) {
        OpType::FuncDecl(fd) => fd.signature().clone(),
        OpType::FuncDefn(fd) => fd.signature().clone(),
        _ => {
            return Err(BuildError::UnexpectedType {
                node: func,
                op_desc: "func defn/decl",
            });
        }
    };
    Ok(Call::try_new(func_sig, type_args)?)
}

/// Options for how the replacement for an op is processed.
///
/// May be specified by [ReplaceTypes::replace_op_with] and [ReplaceTypes::replace_parametrized_op_with].
/// Otherwise (the default), replacements are inserted as is (without further processing).
#[derive(Clone, Default, PartialEq, Eq)] // More derives might inhibit future extension
pub struct ReplacementOptions {
    linearize: bool,
}

impl ReplacementOptions {
    /// Specifies that all operations within the replacement should have their
    /// output ports linearized.
    pub fn with_linearization(mut self, lin: bool) -> Self {
        self.linearize = lin;
        self
    }
}

/// A configuration of what types, ops, and constants should be replaced with what.
/// May be applied to a Hugr via [`Self::run`].
///
/// Parametrized types and ops will be reparametrized taking into account the
/// replacements, but any ops taking/returning the replaced types *not* as a result of
/// parametrization, will also need to be replaced - see [`Self::replace_op`].
/// Similarly [Const]s.
///
/// Types that are [Copyable](hugr_core::types::TypeBound::Copyable) may also be replaced
/// with types that are not, see [Linearizer].
///
/// Note that although this pass may be used before [monomorphization], there are some
/// limitations (that do not apply if done after [monomorphization]):
/// * [`NodeTemplate::CompoundOp`] only works for operations that do not use type variables
/// * "Overrides" of specific instantiations of polymorphic types will not be detected if
///   the instantiations are created inside polymorphic functions. For example, suppose
///   we [`Self::replace_type`] type `A` with `X`, [`Self::replace_parametrized_type`]
///   container `MyList` with `List`, and [`Self::replace_type`] `MyList<A>` with
///   `SpecialListOfXs`. If a function `foo` polymorphic over a type variable `T` dealing
///   with `MyList<T>`s, that is called with type argument `A`, then `foo<T>` will be
///   updated to deal with `List<T>`s and the call `foo<A>` updated to `foo<X>`, but this
///   will still result in using `List<X>` rather than `SpecialListOfXs`. (However this
///   would be fine *after* [monomorphization]: the monomorphic definition of `foo_A`
///   would use `SpecialListOfXs`.)
/// * See also limitations noted for [Linearizer].
///
/// [monomorphization]: super::monomorphize()
#[derive(Clone)]
pub struct ReplaceTypes {
    type_map: HashMap<CustomType, Type>,
    param_types: HashMap<ParametricType, Arc<dyn Fn(&[TypeArg]) -> Option<Type>>>,
    linearize: DelegatingLinearizer,
    op_map: HashMap<OpHashWrapper, (NodeTemplate, ReplacementOptions)>,
    param_ops: HashMap<
        ParametricOp,
        (
            Arc<dyn Fn(&[TypeArg]) -> Option<NodeTemplate>>,
            ReplacementOptions,
        ),
    >,
    consts: HashMap<
        CustomType,
        Arc<dyn Fn(&OpaqueValue, &ReplaceTypes) -> Result<Value, ReplaceTypesError>>,
    >,
    param_consts: HashMap<
        ParametricType,
        Arc<dyn Fn(&OpaqueValue, &ReplaceTypes) -> Result<Option<Value>, ReplaceTypesError>>,
    >,
}

impl Default for ReplaceTypes {
    fn default() -> Self {
        let mut res = Self::new_empty();
        res.linearize = DelegatingLinearizer::default();
        res.replace_consts_parametrized(array_type_def(), handlers::array_const);
        res.replace_consts_parametrized(value_array_type_def(), handlers::value_array_const);
        res.replace_consts_parametrized(list_type_def(), list_const);
        res
    }
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

/// An error produced by the [`ReplaceTypes`] pass
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum ReplaceTypesError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    #[error(transparent)]
    ConstError(#[from] ConstTypeError),
    #[error(transparent)]
    LinearizeError(#[from] LinearizeError),
    #[error("Replacement op for {0} could not be added because {1}")]
    AddTemplateError(Node, Box<BuildError>),
}

impl ReplaceTypes {
    /// Makes a new instance. Unlike [`Self::default`], this does not understand
    /// any extension types, even those in the prelude.
    #[must_use]
    pub fn new_empty() -> Self {
        Self {
            type_map: Default::default(),
            param_types: Default::default(),
            linearize: DelegatingLinearizer::new_empty(),
            op_map: Default::default(),
            param_ops: Default::default(),
            consts: Default::default(),
            param_consts: Default::default(),
        }
    }

    /// Configures this instance to replace occurrences of type `src` with `dest`.
    /// Note that if `src` is an instance of a *parametrized* [`TypeDef`], this takes
    /// precedence over [`Self::replace_parametrized_type`] where the `src`s overlap. Thus, this
    /// should only be used on already-*[monomorphize](super::monomorphize())d* Hugrs, as
    /// substitution (parametric polymorphism) happening later will not respect this replacement.
    ///
    /// If there are any [`LoadConstant`]s of this type, callers should also call [`Self::replace_consts`]
    /// (or [`Self::replace_consts_parametrized`]) as the [`LoadConstant`]s will be reparametrized
    /// (and this will break the edge from [Const] to [`LoadConstant`]).
    ///
    /// Note that if `src` is Copyable and `dest` is Linear, then (besides linearity violations)
    /// [`SignatureError`] will be raised if this leads to an impossible type e.g. ArrayOfCopyables(src).
    /// (This can be overridden by an additional [`Self::replace_type`].)
    pub fn replace_type(&mut self, src: CustomType, dest: Type) {
        // We could check that 'dest' is copyable or 'src' is linear, but since we can't
        // check that for parametrized types, we'll be consistent and not check here either.
        self.type_map.insert(src, dest);
    }

    /// Configures this instance to change occurrences of a parametrized type `src`
    /// via a callback that builds the replacement type given the [`TypeArg`]s.
    /// Note that the `TypeArgs` will already have been updated (e.g. they may not
    /// fit the bounds of the original type). The callback may return `None` to indicate
    /// no change (in which case the supplied `TypeArgs` will be given to `src`).
    ///
    /// If there are any [`LoadConstant`]s of any of these types, callers should also call
    /// [`Self::replace_consts_parametrized`] (or [`Self::replace_consts`]) as the
    /// [`LoadConstant`]s will be reparametrized (and this will break the edge from [Const] to
    /// [`LoadConstant`]).
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
    /// Equivalent to [Self::replace_op_with] with default [ReplacementOptions].
    pub fn replace_op(&mut self, src: &ExtensionOp, dest: NodeTemplate) {
        self.replace_op_with(src, dest, ReplacementOptions::default())
    }

    /// Configures this instance to change occurrences of `src` to `dest`.
    ///
    /// Note that if `src` is an instance of a *parametrized* [`OpDef`], this takes
    /// precedence over [`Self::replace_parametrized_op`] where the `src`s overlap. Thus,
    /// this should only be used on already-*[monomorphize](super::monomorphize())d*
    /// Hugrs, as substitution (parametric polymorphism) happening later will not respect
    /// this replacement.
    pub fn replace_op_with(
        &mut self,
        src: &ExtensionOp,
        dest: NodeTemplate,
        opts: ReplacementOptions,
    ) {
        self.op_map.insert(OpHashWrapper::from(src), (dest, opts));
    }

    /// Configures this instance to change occurrences of a parametrized op `src`
    /// via a callback that builds the replacement type given the [`TypeArg`]s.
    /// Equivalent to [Self::replace_parametrized_op_with] with default [ReplacementOptions].
    pub fn replace_parametrized_op(
        &mut self,
        src: &OpDef,
        dest_fn: impl Fn(&[TypeArg]) -> Option<NodeTemplate> + 'static,
    ) {
        self.replace_parametrized_op_with(src, dest_fn, ReplacementOptions::default())
    }

    /// Configures this instance to change occurrences of a parametrized op `src`
    /// via a callback that builds the replacement type given the [`TypeArg`]s.
    /// Note that the `TypeArgs` will already have been updated (e.g. they may not
    /// fit the bounds of the original op).
    ///
    /// If the Callback returns None, the new typeargs will be applied to the original op.
    pub fn replace_parametrized_op_with(
        &mut self,
        src: &OpDef,
        dest_fn: impl Fn(&[TypeArg]) -> Option<NodeTemplate> + 'static,
        opts: ReplacementOptions,
    ) {
        self.param_ops.insert(src.into(), (Arc::new(dest_fn), opts));
    }

    /// Configures this instance to change [Const]s of type `src_ty`, using
    /// a callback that is passed the value of the constant (of that type).
    ///
    /// Note that if `src_ty` is an instance of a *parametrized* [`TypeDef`],
    /// this takes precedence over [`Self::replace_consts_parametrized`] where
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
    /// value of the constant (the [`OpaqueValue`] contains the [`TypeArg`]s). The
    /// callback may return `None` to indicate no change to the constant.
    pub fn replace_consts_parametrized(
        &mut self,
        src_ty: &TypeDef,
        const_fn: impl Fn(&OpaqueValue, &ReplaceTypes) -> Result<Option<Value>, ReplaceTypesError>
        + 'static,
    ) {
        self.param_consts.insert(src_ty.into(), Arc::new(const_fn));
    }

    fn change_node(
        &self,
        hugr: &mut impl HugrMut<Node = Node>,
        n: Node,
    ) -> Result<bool, ReplaceTypesError> {
        match hugr.optype_mut(n) {
            OpType::FuncDefn(fd) => fd.signature_mut().body_mut().transform(self),
            OpType::FuncDecl(fd) => fd.signature_mut().body_mut().transform(self),
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
            OpType::ExtensionOp(ext_op) => Ok({
                let def = ext_op.def_arc();
                let mut changed = false;
                let replacement = match self.op_map.get(&OpHashWrapper::from(&*ext_op)) {
                    r @ Some(_) => r.cloned(),
                    None => {
                        let mut args = ext_op.args().to_vec();
                        changed = args.transform(self)?;
                        let r2 = self
                            .param_ops
                            .get(&def.as_ref().into())
                            .and_then(|(rep_fn, opts)| rep_fn(&args).map(|nt| (nt, opts.clone())));
                        if r2.is_none() && changed {
                            *ext_op = ExtensionOp::new(def.clone(), args)?;
                        }
                        r2
                    }
                };
                if let Some((replacement, opts)) = replacement {
                    replacement
                        .replace(hugr, n)
                        .map_err(|e| ReplaceTypesError::AddTemplateError(n, Box::new(e)))?;
                    if opts.linearize {
                        for d in hugr.descendants(n).collect::<Vec<_>>() {
                            if d != n {
                                self.linearize_outputs(hugr, d)?;
                            }
                        }
                    }
                    true
                } else {
                    changed
                }
            }),

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

    fn linearize_outputs<H: HugrMut<Node = Node>>(
        &self,
        hugr: &mut H,
        n: H::Node,
    ) -> Result<(), LinearizeError> {
        if let Some(new_sig) = hugr.get_optype(n).dataflow_signature() {
            let new_sig = new_sig.into_owned();
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
        Ok(())
    }
}

impl<H: HugrMut<Node = Node>> ComposablePass<H> for ReplaceTypes {
    type Error = ReplaceTypesError;
    type Result = bool;

    fn run(&self, hugr: &mut H) -> Result<bool, ReplaceTypesError> {
        let mut changed = false;
        for n in hugr.entry_descendants().collect::<Vec<_>>() {
            changed |= self.change_node(hugr, n)?;
            if n != hugr.entrypoint() && changed {
                self.linearize_outputs(hugr, n)?;
            }
        }
        Ok(changed)
    }
}

pub mod handlers;

#[derive(Clone, Hash, PartialEq, Eq)]
struct OpHashWrapper {
    op_name: String, // Only because SmolStr not in hugr-passes yet
    args: Vec<TypeArg>,
}

impl From<&ExtensionOp> for OpHashWrapper {
    fn from(op: &ExtensionOp) -> Self {
        Self {
            op_name: op.qualified_id().to_string(),
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

    use crate::replace_types::handlers::generic_array_const;
    use hugr_core::builder::{
        BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
        FunctionBuilder, HugrBuilder, ModuleBuilder, SubContainer, TailLoopBuilder, inout_sig,
    };
    use hugr_core::extension::prelude::{
        ConstUsize, UnwrapBuilder, bool_t, option_type, qb_t, usize_t,
    };
    use hugr_core::extension::{TypeDefBound, Version, simple_op::MakeExtensionOp};
    use hugr_core::hugr::hugrmut::HugrMut;
    use hugr_core::hugr::{IdentList, ValidationError};
    use hugr_core::ops::constant::CustomConst;
    use hugr_core::ops::constant::OpaqueValue;
    use hugr_core::ops::{ExtensionOp, OpTrait, OpType, Tag, Value};
    use hugr_core::std_extensions::arithmetic::conversions::ConvertOpDef;
    use hugr_core::std_extensions::arithmetic::int_types::{ConstInt, INT_TYPES};
    use hugr_core::std_extensions::collections::array::{Array, ArrayKind, GenericArrayValue};
    use hugr_core::std_extensions::collections::list::{
        ListOp, ListValue, list_type, list_type_def,
    };
    use hugr_core::std_extensions::collections::value_array::{
        VArrayOp, VArrayOpDef, VArrayValue, ValueArray, value_array_type,
    };

    use hugr_core::types::{PolyFuncType, Signature, SumType, Type, TypeArg, TypeBound, TypeRow};
    use hugr_core::{Extension, HugrView, type_row};
    use itertools::Itertools;
    use rstest::rstest;

    use crate::ComposablePass;

    use super::{NodeTemplate, ReplaceTypes, handlers::list_const};

    const PACKED_VEC: &str = "PackedVec";
    const READ: &str = "read";

    fn i64_t() -> Type {
        INT_TYPES[6].clone()
    }

    fn read_op(ext: &Arc<Extension>, t: Type) -> ExtensionOp {
        ExtensionOp::new(ext.get_op(READ).unwrap().clone(), [t.into()]).unwrap()
    }

    fn just_elem_type(args: &[TypeArg]) -> &Type {
        let [TypeArg::Runtime(ty)] = args else {
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
                        vec![TypeBound::Linear.into()],
                        String::new(),
                        TypeDefBound::from_params(vec![0]),
                        w,
                    )
                    .unwrap()
                    .instantiate(vec![Type::new_var_use(0, TypeBound::Copyable).into()])
                    .unwrap();
                ext.add_op(
                    READ.into(),
                    String::new(),
                    PolyFuncType::new(
                        vec![TypeBound::Copyable.into()],
                        Signature::new(
                            vec![pv_of_var.into(), i64_t()],
                            Type::new_var_use(0, TypeBound::Linear),
                        ),
                    ),
                    w,
                )
                .unwrap();
                ext.add_op(
                    "lowered_read_bool".into(),
                    String::new(),
                    Signature::new(vec![i64_t(); 2], bool_t()),
                    w,
                )
                .unwrap();
            },
        )
    }

    fn lowered_read<T: Container + Dataflow>(
        elem_ty: Type,
        new: impl Fn(Signature) -> Result<T, BuildError>,
    ) -> T {
        let mut dfb = new(Signature::new(
            vec![value_array_type(64, elem_ty.clone()), i64_t()],
            elem_ty.clone(),
        ))
        .unwrap();
        let [val, idx] = dfb.input_wires_arr();
        let [idx] = dfb
            .add_dataflow_op(ConvertOpDef::itousize.without_log_width(), [idx])
            .unwrap()
            .outputs_arr();
        let [opt, _] = dfb
            .add_dataflow_op(
                VArrayOpDef::get.to_concrete(elem_ty.clone(), 64),
                [val, idx],
            )
            .unwrap()
            .outputs_arr();
        let [res] = dfb
            .build_unwrap_sum(1, option_type(Type::from(elem_ty)), opt)
            .unwrap();
        dfb.set_outputs([res]).unwrap();
        dfb
    }

    fn lowerer(ext: &Arc<Extension>) -> ReplaceTypes {
        let pv = ext.get_type(PACKED_VEC).unwrap();
        let mut lw = ReplaceTypes::default();
        lw.replace_type(pv.instantiate([bool_t().into()]).unwrap(), i64_t());
        lw.replace_parametrized_type(
            pv,
            Box::new(|args: &[TypeArg]| Some(value_array_type(64, just_elem_type(args).clone()))),
        );
        lw.replace_op(
            &read_op(ext, bool_t()),
            NodeTemplate::SingleOp(
                ExtensionOp::new(ext.get_op("lowered_read_bool").unwrap().clone(), [])
                    .unwrap()
                    .into(),
            ),
        );
        lw.replace_parametrized_op(ext.get_op(READ).unwrap().as_ref(), |type_args| {
            Some(NodeTemplate::CompoundOp(Box::new(
                lowered_read(just_elem_type(type_args).clone(), DFGBuilder::new)
                    .finish_hugr()
                    .unwrap(),
            )))
        });
        lw
    }

    #[test]
    fn module_func_cfg_call() {
        let ext = ext();
        let coln = ext.get_type(PACKED_VEC).unwrap();
        let c_int = Type::from(coln.instantiate([i64_t().into()]).unwrap());
        let c_bool = Type::from(coln.instantiate([bool_t().into()]).unwrap());
        let mut mb = ModuleBuilder::new();
        let sig = Signature::new_endo(Type::new_var_use(0, TypeBound::Linear));
        let fb = mb
            .define_function("id", PolyFuncType::new([TypeBound::Linear.into()], sig))
            .unwrap();
        let inps = fb.input_wires();
        let id = fb.finish_with_outputs(inps).unwrap();

        let sig = Signature::new(vec![i64_t(), c_int.clone(), c_bool.clone()], bool_t());
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

        let ext_ops = h
            .entry_descendants()
            .filter_map(|n| h.get_optype(n).as_extension_op());
        assert_eq!(
            ext_ops
                .map(hugr_core::ops::ExtensionOp::unqualified_id)
                .sorted()
                .collect_vec(),
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
            .entry_descendants()
            .filter_map(|n| h.get_optype(n).as_extension_op())
            .collect_vec();
        assert_eq!(
            ext_ops
                .iter()
                .map(|x| x.unqualified_id())
                .sorted()
                .collect_vec(),
            ["get", "itousize", "panic"]
        );
        // The PackedVec<PackedVec<bool>> becomes an array<i64>
        let [array_get] = ext_ops
            .into_iter()
            .filter_map(|e| VArrayOp::from_extension_op(e).ok())
            .collect_array()
            .unwrap();
        assert_eq!(array_get, VArrayOpDef::get.to_concrete(i64_t(), 64));
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

        // 1. Lower List<T> to Array<10, T> UNLESS T is usize_t() or i64_t
        lowerer.replace_parametrized_type(list_type_def(), |args| {
            let ty = just_elem_type(args);
            (![usize_t(), i64_t()].contains(ty)).then_some(value_array_type(10, ty.clone()))
        });
        {
            let mut h = backup.clone();
            assert_eq!(lowerer.run(&mut h), Ok(true));
            let sig = h.signature(h.entrypoint()).unwrap();
            assert_eq!(
                sig.input(),
                &TypeRow::from(vec![list_type(usize_t()), value_array_type(10, bool_t())])
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
            let sig = h.signature(h.entrypoint()).unwrap();
            assert_eq!(
                sig.input(),
                &TypeRow::from(vec![list_type(i64_t()), value_array_type(10, bool_t())])
            );
            assert_eq!(sig.input(), sig.output());
            // This will have to update inside the Const
            let cst = h
                .entry_descendants()
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
            Box::new(|args: &[TypeArg]| Some(value_array_type(4, just_elem_type(args).clone()))),
        );
        lowerer.replace_consts_parametrized(list_type_def(), |opaq, repl| {
            // First recursively transform the contents
            let Some(Value::Extension { e: opaq }) = list_const(opaq, repl)? else {
                panic!("Expected list value to stay a list value");
            };
            let lv = opaq.value().downcast_ref::<ListValue>().unwrap();

            Ok(Some(
                VArrayValue::new(lv.get_element_type().clone(), lv.get_contents().to_vec()).into(),
            ))
        });
        lowerer.run(&mut h).unwrap();

        assert_eq!(
            h.get_optype(pred.node())
                .as_load_constant()
                .map(hugr_core::ops::LoadConstant::constant_type),
            Some(&Type::new_sum(vec![
                Type::from(value_array_type(4, i64_t()));
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
                let params = vec![TypeBound::Linear.into()];
                let tv = Type::new_var_use(0, TypeBound::Linear);
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
        let i32_t = || INT_TYPES[5].clone();
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
            h.entrypoint_optype().dataflow_signature().unwrap().io(),
            (
                &vec![list_type(qb_t()); 2].into(),
                &vec![qb_t(), option_type(qb_t()).into()].into()
            )
        );
        assert_eq!(
            h.entry_descendants()
                .filter_map(|n| h.get_optype(n).as_extension_op())
                .map(hugr_core::ops::ExtensionOp::qualified_id)
                .sorted()
                .collect_vec(),
            ["NoBoundsCheck.read", "collections.list.get"]
        );
    }

    #[rstest]
    #[case(&[], Array)]
    #[case(&[], ValueArray)]
    #[case(&[3], Array)]
    #[case(&[3], ValueArray)]
    #[case(&[5,7,11,13,17,19], Array)]
    #[case(&[5,7,11,13,17,19], ValueArray)]
    fn array_const<AK: ArrayKind>(#[case] vals: &[u64], #[case] _kind: AK)
    where
        GenericArrayValue<AK>: CustomConst,
    {
        let mut dfb =
            DFGBuilder::new(inout_sig(type_row![], AK::ty(vals.len() as _, usize_t()))).unwrap();
        let c = dfb.add_load_value(GenericArrayValue::<AK>::new(
            usize_t(),
            vals.iter().map(|u| ConstUsize::new(*u).into()),
        ));
        let backup = dfb.finish_hugr_with_outputs([c]).unwrap();

        let mut repl = ReplaceTypes::new_empty();
        let usize_custom_t = usize_t().as_extension().unwrap().clone();
        repl.replace_type(usize_custom_t.clone(), INT_TYPES[6].clone());
        repl.replace_consts(usize_custom_t, |cst: &OpaqueValue, _| {
            let cu = cst.value().downcast_ref::<ConstUsize>().unwrap();
            Ok(ConstInt::new_u(6, cu.value())?.into())
        });

        let mut h = backup.clone();
        repl.run(&mut h).unwrap(); // No validation here
        assert!(
            matches!(h.validate(), Err(ValidationError::IncompatiblePorts {from, to, ..})
             if backup.get_optype(from).is_const() && to == c.node())
        );
        repl.replace_consts_parametrized(AK::type_def(), generic_array_const::<AK>);
        let mut h = backup;
        repl.run(&mut h).unwrap();
        h.validate().unwrap();
    }

    #[test]
    fn op_to_call() {
        let e = ext();
        let pv = e.get_type(PACKED_VEC).unwrap();
        let inner = pv.instantiate([usize_t().into()]).unwrap();
        let outer = pv
            .instantiate([Type::new_extension(inner.clone()).into()])
            .unwrap();
        let mut dfb = DFGBuilder::new(inout_sig(vec![outer.into(), i64_t()], usize_t())).unwrap();
        let [outer, idx] = dfb.input_wires_arr();
        let [inner] = dfb
            .add_dataflow_op(read_op(&e, inner.clone().into()), [outer, idx])
            .unwrap()
            .outputs_arr();
        let res = dfb
            .add_dataflow_op(read_op(&e, usize_t()), [inner, idx])
            .unwrap();
        let mut h = dfb.finish_hugr_with_outputs(res.outputs()).unwrap();
        let read_func = h
            .insert_hugr(
                h.entrypoint(),
                lowered_read(Type::new_var_use(0, TypeBound::Copyable), |sig| {
                    FunctionBuilder::new(
                        "lowered_read",
                        PolyFuncType::new([TypeBound::Copyable.into()], sig),
                    )
                })
                .finish_hugr()
                .unwrap(),
            )
            .inserted_entrypoint;

        let mut lw = lowerer(&e);
        lw.replace_parametrized_op(e.get_op(READ).unwrap().as_ref(), move |args| {
            Some(NodeTemplate::Call(read_func, args.to_owned()))
        });
        lw.run(&mut h).unwrap();

        assert_eq!(h.output_neighbours(read_func).count(), 2);
        let ext_op_names = h
            .entry_descendants()
            .filter_map(|n| h.get_optype(n).as_extension_op())
            .map(hugr_core::ops::ExtensionOp::unqualified_id)
            .sorted()
            .collect_vec();
        assert_eq!(ext_op_names, ["get", "itousize", "panic",]);
    }
}
