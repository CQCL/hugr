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

#[derive(Clone, Debug, PartialEq)]
pub enum OpReplacement {
    SingleOp(OpType),
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
    param_types: HashMap<(ExtensionId, String), Arc<dyn Fn(&[TypeArg]) -> Type>>,
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
        // overapproximation.
        self.param_types.insert(
            (src.extension_id().clone(), src.name().to_string()),
            Arc::from(dest_fn),
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
