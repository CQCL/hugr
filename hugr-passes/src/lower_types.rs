use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::ops::OpTrait;
use hugr_core::types::{Transformable, TypeEnum, TypeTransformer};
use hugr_core::{Hugr, IncomingPort, OutgoingPort};
use thiserror::Error;

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use hugr_core::{
    extension::SignatureError,
    hugr::hugrmut::HugrMut,
    ops::{
        AliasDefn, Call, CallIndirect, Case, Conditional, Const, DataflowBlock, ExitBlock,
        ExtensionOp, FuncDecl, FuncDefn, Input, LoadConstant, LoadFunction, OpType, Output, Tag,
        TailLoop, Value, CFG, DFG,
    },
    types::{CustomType, Type, TypeBound},
    Node,
};

#[derive(Clone, Debug, PartialEq)]
enum OpReplacement {
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
    /// No support yet for mapping parametrically.
    type_map: HashMap<CustomType, Type>,
    copy_discard: HashMap<CustomType, (OpReplacement, OpReplacement)>, // TODO what about e.g. arrays that have gone from copyable to linear because their elements have?!
    op_map: HashMap<ExtensionOp, OpReplacement>,
    //        1. is input op always a single OpType, or a schema/predicate?
    //        2. output might not be an op - might be a node with children
    //        3. do we need checking BEFORE reparametrization as well as after? (after only if not reparametrized?)
    const_fn: Arc<dyn Fn(&dyn CustomConst) -> Option<Value>>,
    check_sig: bool,
}

impl TypeTransformer for LowerTypes {
    type Err = ChangeTypeError;

    fn apply_custom(&self, ct: &CustomType) -> Result<Option<Type>, Self::Err> {
        Ok(self.type_map.get(ct).cloned())
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
        if src.bound() == TypeBound::Copyable && !dest.copyable() {
            // Of course we could try, and fail only if we encounter outports that are not singly-used!
            panic!("Cannot lower copyable type to linear without copy/dup - use lower_type_linearize instead");
        }
        self.type_map.insert(src, dest);
    }

    pub fn lower_type_linearize(
        &mut self,
        src: CustomType,
        dest: Type,
        copy: OpReplacement,
        discard: OpReplacement,
    ) {
        self.type_map.insert(src.clone(), dest);
        self.copy_discard.insert(src, (copy, discard));
    }

    pub fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            let expected_sig = self.check_sig.then(|| {
                let mut dfsig = hugr.get_optype(n).dataflow_signature().map(Cow::into_owned);
                if let Some(sig) = dfsig.as_mut() {
                    sig.transform(self);
                }
                dfsig
            });
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
                if targets.len() == 0 {
                    self.do_discard(hugr, n, outp)
                } else {
                    self.do_copy_chain(hugr, n, outp, &targets)
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
        inps: &[(Node, IncomingPort)],
    ) {
        assert!(inps.len() > 1);
        for (tgt_node, tgt_port) in &inps[..inps.len() - 1] {
            (src_node, src_port) = self.do_copy(hugr, src_node, src_port, *tgt_node, *tgt_port);
        }
        let (tgt_node, tgt_port) = inps.last().unwrap();
        hugr.connect(src_node, src_port, *tgt_node, *tgt_port)
    }

    fn do_copy(
        &self,
        hugr: &mut impl HugrMut,
        src_node: Node,
        src_port: OutgoingPort,
        tgt_node: Node,
        tgt_port: IncomingPort,
    ) -> (Node, OutgoingPort) {
        let sig = hugr.get_optype(src_node).dataflow_signature().unwrap();
        let typ = sig.out_port_type(src_port).unwrap();
        if let TypeEnum::Extension(exty) = typ.as_type_enum() {
            if let Some((copy, _)) = self.copy_discard.get(exty) {
                let n = copy.add(hugr, hugr.get_parent(src_node).unwrap());
                hugr.connect(src_node, src_port, n, 0);
                hugr.connect(n, 0, tgt_node, tgt_port);
                return (n, 1.into());
            }
        }
        todo!("Containers/arrays/etc.")
    }

    fn do_discard(&self, hugr: &mut impl HugrMut, src_node: Node, src_port: OutgoingPort) {
        let sig = hugr.get_optype(src_node).dataflow_signature().unwrap();
        let typ = sig.out_port_type(src_port).unwrap();
        if let TypeEnum::Extension(exty) = typ.as_type_enum() {
            if let Some((_, discard)) = self.copy_discard.get(exty) {
                let n = discard.add(hugr, hugr.get_parent(src_node).unwrap());
                hugr.connect(src_node, src_port, n, 0);
                return;
            }
        }
        todo!("Containers/arrays/etc.")
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
                if let Some(replacement) = self.op_map.get(ext_op) {
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
