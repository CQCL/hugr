use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use thiserror::Error;

use hugr_core::extension::{ExtensionId, SignatureError, TypeDef};
use hugr_core::hugr::hugrmut::HugrMut;
use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::ops::{
    AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock, ExtensionOp,
    FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpTrait, OpType, Output, Tag, TailLoop,
    Value, CFG, DFG,
};
use hugr_core::types::{CustomType, Transformable, Type, TypeArg, TypeEnum, TypeTransformer};
use hugr_core::{Hugr, IncomingPort, Node, OutgoingPort};

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

#[derive(Clone, Debug, PartialEq)]
pub enum OpReplacement {
    SingleOp(OpType),
    CompoundOp(Box<Hugr>),
}

impl OpReplacement {
    fn add(&self, hugr: &mut impl HugrMut, parent: Node) -> Node {
        match self.clone() {
            OpReplacement::SingleOp(op_type) => hugr.add_node_with_parent(parent, op_type),
            OpReplacement::CompoundOp(new_h) => hugr.insert_hugr(parent, *new_h).new_root,
        }
    }

    // n must be non-root. I mean, it's an ExtensionOp...
    fn replace(&self, hugr: &mut impl HugrMut, n: Node) {
        let new_optype = match self.clone() {
            OpReplacement::SingleOp(op_type) => op_type,
            OpReplacement::CompoundOp(new_h) => {
                let new_root = hugr
                    .insert_hugr(hugr.get_parent(n).unwrap(), *new_h)
                    .new_root;
                for ch in hugr.children(new_root).collect::<Vec<_>>() {
                    hugr.set_parent(ch, n);
                }
                hugr.remove_node(new_root)
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
    param_types: HashMap<(ExtensionId, String), Arc<dyn Fn(&[TypeArg]) -> Type>>,
    // Keyed by lowered type, as only needed when there is an op outputting such
    copy_discard: HashMap<Type, (OpReplacement, OpReplacement)>,
    // Copy/discard of parametric types handled by a function that receives the new/lowered type.
    // We do not allow linearization to "parametrized" non-extension types, at least not
    // in one step. We could do that using a trait, but it seems enough of a corner case.
    // Instead that can be achieved by *firstly* lowering to a custom linear type, with copy/dup
    // inserted; *secondly* by lowering that to the desired non-extension linear type,
    // including lowering of the copy/dup operations to...whatever.
    copy_discard_parametric: HashMap<
        (ExtensionId, String),
        // TODO should pass &LowerTypes, or at least some way to call copy_op / discard_op, to these
        (
            Arc<dyn Fn(&[TypeArg]) -> OpReplacement>,
            Arc<dyn Fn(&[TypeArg]) -> OpReplacement>,
        ),
    >,
    // Handles simple cases Op1 -> Op2. TODO handle parametric ops
    op_map: HashMap<OpHashWrapper, OpReplacement>,
    //        1. is input op always a single OpType, or a schema/predicate?
    //        2. output might not be an op - might be a node with children
    //        3. do we need checking BEFORE reparametrization as well as after? (after only if not reparametrized?)
    const_fn: Arc<dyn Fn(&dyn CustomConst) -> Option<Value>>,
    check_sig: bool,
}

impl TypeTransformer for LowerTypes {
    type Err = ChangeTypeError;

    fn apply_custom(&self, ct: &CustomType) -> Result<Option<Type>, Self::Err> {
        Ok(if let Some(res) = self.type_map.get(ct) {
            Some(res.clone())
        } else if let Some(dest_fn) = self
            .param_types
            .get(&(ct.extension().clone(), ct.name().to_string()))
        {
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

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ChangeTypeError {
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
}

impl LowerTypes {
    pub fn lower_type(&mut self, src: CustomType, dest: Type) {
        self.type_map.insert(src, dest);
    }

    pub fn lower_parametric_type(
        &mut self,
        src: TypeDef,
        dest_fn: Box<dyn Fn(&[TypeArg]) -> Type>,
    ) {
        // No way to check that dest_fn never produces a linear type.
        // We could require copy/discard-generators if src is Copyable, or *might be*
        // (depending on arguments - i.e. if src's TypeDefBound is anything other than
        // `TypeDefBound::Explicit(TypeBound::Copyable)`) but that seems an annoying
        // overapproximation. We could just require copy/discard-generators in *all cases*
        // (e.g. funcs that just panic!)...
        self.param_types.insert(
            (src.extension_id().clone(), src.name().to_string()),
            Arc::from(dest_fn),
        );
    }

    pub fn linearize(&mut self, src: Type, copy: OpReplacement, discard: OpReplacement) {
        self.copy_discard.insert(src, (copy, discard));
    }

    pub fn linearize_parametric(
        &mut self,
        src: TypeDef,
        copy_fn: Box<dyn Fn(&[TypeArg]) -> OpReplacement>,
        discard_fn: Box<dyn Fn(&[TypeArg]) -> OpReplacement>,
    ) {
        self.copy_discard_parametric.insert(
            (src.extension_id().clone(), src.name().to_string()),
            (Arc::from(copy_fn), Arc::from(discard_fn)),
        );
    }

    pub fn lower_op(&mut self, src: &ExtensionOp, tgt: OpReplacement) {
        self.op_map.insert(OpHashWrapper::from(src), tgt);
    }

    pub fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            let expected_sig = if self.check_sig {
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
            if let Some(dfsig) = expected_sig {
                assert_eq!(new_dfsig.as_ref().map(Cow::deref), dfsig.as_ref());
            }
            let Some(new_sig) = new_dfsig.filter(|_| changed) else {
                continue;
            };
            let new_sig = new_sig.into_owned();
            for outp in new_sig.output_ports() {
                if new_sig.out_port_type(outp).unwrap().copyable() {
                    continue;
                };
                let targets = hugr.linked_inputs(n, outp).collect::<Vec<_>>();
                if targets.len() == 1 {
                    continue;
                };
                hugr.disconnect(n, outp);
                let sig = hugr.get_optype(n).dataflow_signature().unwrap();
                let typ = sig.out_port_type(outp).unwrap();
                if targets.len() == 0 {
                    let discard = self
                        .discard_op(typ)
                        .expect("Don't know how to discard {typ:?}"); // TODO return error

                    let disc = discard.add(hugr, hugr.get_parent(n).unwrap());
                    hugr.connect(n, outp, disc, 0);
                } else {
                    // TODO return error
                    let copy = self.copy_op(typ).expect("Don't know how to copy {typ:?}");
                    self.do_copy_chain(hugr, n, outp, copy, &targets)
                }
            }
        }
        Ok(changed)
    }

    fn do_copy_chain(
        &self,
        hugr: &mut impl HugrMut,
        mut src_node: Node,
        mut src_port: OutgoingPort,
        copy: OpReplacement,
        inps: &[(Node, IncomingPort)],
    ) {
        assert!(inps.len() > 1);
        // Could sanity-check signature here?
        for (tgt_node, tgt_port) in &inps[..inps.len() - 1] {
            let n = copy.add(hugr, hugr.get_parent(src_node).unwrap());
            hugr.connect(src_node, src_port, n, 0);
            hugr.connect(n, 0, *tgt_node, *tgt_port);
            (src_node, src_port) = (n, 1.into());
        }
        let (tgt_node, tgt_port) = inps.last().unwrap();
        hugr.connect(src_node, src_port, *tgt_node, *tgt_port)
    }

    pub fn copy_op(&self, typ: &Type) -> Option<OpReplacement> {
        if let Some((copy, _)) = self.copy_discard.get(typ) {
            return Some(copy.clone());
        }
        let TypeEnum::Extension(exty) = typ.as_type_enum() else {
            return None;
        };
        self.copy_discard_parametric
            .get(&(exty.extension().clone(), exty.name().to_string()))
            .map(|(copy_fn, _)| copy_fn(exty.args()))
    }

    pub fn discard_op(&self, typ: &Type) -> Option<OpReplacement> {
        if let Some((_, discard)) = self.copy_discard.get(typ) {
            return Some(discard.clone());
        }
        let TypeEnum::Extension(exty) = typ.as_type_enum() else {
            return None;
        };
        self.copy_discard_parametric
            .get(&(exty.extension().clone(), exty.name().to_string()))
            .map(|(_, discard_fn)| discard_fn(exty.args()))
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
            OpType::ExtensionOp(ext_op) => {
                if let Some(replacement) = self.op_map.get(&OpHashWrapper::from(&*ext_op)) {
                    replacement.replace(hugr, n); // Copy/discard insertion done by caller
                    Ok(true)
                } else {
                    let def = ext_op.def_arc();
                    let mut args = ext_op.args().to_vec();
                    Ok(args.transform(self)? && {
                        *ext_op = ExtensionOp::new(def.clone(), args)?;
                        true
                    })
                }
            }
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
                if let Some(new_const) = self.subst_custom_const(e.value())? {
                    *value = new_const;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Value::Function { hugr } => self.run_no_validate(&mut **hugr),
        }
    }

    fn subst_custom_const(&self, _cst: &dyn CustomConst) -> Result<Option<Value>, ChangeTypeError> {
        todo!()
    }
}
