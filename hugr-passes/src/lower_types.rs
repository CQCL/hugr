use hugr_core::ops::constant::{CustomConst, Sum};
use hugr_core::types::{Transformable, TypeTransformer};
use thiserror::Error;

use std::collections::HashMap;
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

#[derive(Clone)]
pub struct LowerTypes {
    type_fn: Arc<dyn Fn(&CustomType) -> Option<Type>>,
    type_map: HashMap<CustomType, Type>,
    copy_dup: HashMap<CustomType, (OpType, OpType)>, // TODO what about e.g. arrays that have gone from copyable to linear because their elements have?!
    //op_map: HashMap<OpType, OpType>
    //        1. is input op always a single OpType, or a schema/predicate?
    //        2. output might not be an op - might be a node with children
    //        3. do we need checking BEFORE reparametrization as well as after? (after only if not reparametrized?)
    #[allow(unused)]
    const_fn: Arc<dyn Fn(&dyn CustomConst) -> Option<Value>>,
}

impl TypeTransformer for LowerTypes {
    type Err = ChangeTypeError;

    fn apply_custom(&self, ct: &CustomType) -> Result<Option<Type>, Self::Err> {
        Ok(if let Some(r) = (self.type_fn)(ct) {
            Some(r)
        } else if let Some(r) = self.type_map.get(ct) {
            Some(r.clone())
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
        if src.bound() == TypeBound::Copyable && !dest.copyable() {
            // Of course we could try, and fail only if we encounter outports that are not singly-used!
            panic!("Cannot lower copyable type to linear without copy/dup - use lower_type_linearize instead");
        }
        self.type_map.insert(src, dest);
    }

    pub fn lower_type_linearize(&mut self, src: CustomType, dest: Type, copy: OpType, dup: OpType) {
        self.type_map.insert(src.clone(), dest);
        self.copy_dup.insert(src, (copy, dup));
    }

    pub fn run_no_validate(&self, hugr: &mut impl HugrMut) -> Result<bool, ChangeTypeError> {
        let mut changed = false;
        for n in hugr.nodes().collect::<Vec<_>>() {
            changed |= self.change_node(hugr, n)?;
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
                let def = ext_op.def_arc();
                let mut args = ext_op.args().to_vec();
                if args.transform(self)? {
                    *ext_op = ExtensionOp::new(def.clone(), args)?;
                }
                // let params = ext_op_params[node].to_owned();
                todo!("Also check whether we should lower op")
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
